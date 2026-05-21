#include "tblur_cuda.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdarg>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <atomic>
#include <mutex>
#include <string>
#include <thread>

// ---------------------------------------------------------------------------
// Error handling
//
// Errors are written to a per-user log AND stashed in a thread-safe global so
// the Nuke node can surface them via Op::error().
// ---------------------------------------------------------------------------

struct SpinLock {
  std::atomic_flag flag = ATOMIC_FLAG_INIT;

  void lock() {
    while (flag.test_and_set(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  }

  void unlock() { flag.clear(std::memory_order_release); }
};

static SpinLock g_last_error_mutex;
static std::string g_last_error;

static void set_last_error(const char* msg) {
  std::lock_guard<SpinLock> lock(g_last_error_mutex);
  g_last_error = msg ? msg : "";
}

static void cuda_logf(const char* fmt, ...) {
  char msg[2048];
  va_list args;
  va_start(args, fmt);
  vsnprintf(msg, sizeof(msg), fmt, args);
  va_end(args);

  set_last_error(msg);

  const char* home = std::getenv("USERPROFILE");
  if (home && *home) {
    std::string path(home);
    path += "\\.nuke\\TBlur\\cuda_runtime.log";
    if (FILE* fp = std::fopen(path.c_str(), "a")) {
      std::fprintf(fp, "%s\n", msg);
      std::fclose(fp);
    }
  }
}

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      cuda_logf("TBlur CUDA error: %s (%s:%d)",                                \
              cudaGetErrorString(err), __FILE__, __LINE__);                    \
      return false;                                                            \
    }                                                                          \
  } while (0)

#define CUDA_CHECK_KERNEL()                                                    \
  do {                                                                         \
    cudaError_t err = cudaGetLastError();                                      \
    if (err != cudaSuccess) {                                                  \
      cuda_logf("TBlur CUDA kernel error: %s (%s:%d)",                         \
              cudaGetErrorString(err), __FILE__, __LINE__);                    \
      return false;                                                            \
    }                                                                          \
  } while (0)

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ float d_clamp01(float v) {
  return fmaxf(0.0f, fminf(1.0f, v));
}

__device__ __forceinline__ float d_finite_or(float v, float fallback) {
  return isfinite(v) ? v : fallback;
}

__device__ __forceinline__ float d_luminance(float r, float g, float b) {
  return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

__device__ __forceinline__ float d_edge_detect_luma(float value) {
  float v = d_clamp01(value);
  return log2f(1.0f + v * 31.0f) * 0.2f;
}

static inline float iteration_sigma_scale_host(int iter_index, int iterations) {
  const int n = max(1, iterations);
  const int i = max(0, min(iter_index, n - 1));
  const float num = sqrtf(3.0f) * powf(2.0f, static_cast<float>(n - i - 1));
  const float den = sqrtf(powf(4.0f, static_cast<float>(n)) - 1.0f);
  return (den > 1e-6f) ? (num / den) : 1.0f;
}

__device__ __forceinline__ float d_detection_gain_curve(float value, float gain) {
  float v = d_clamp01(value);
  if (gain <= 1.0001f) {
    return v;
  }
  float denom = v + (1.0f - v) / gain;
  return (denom > 1e-6f) ? d_clamp01(v / denom) : v;
}

__device__ __forceinline__ void d_detection_preboost_rgb(
    float r, float g, float b, float& out_r, float& out_g, float& out_b) {
  float rc = d_clamp01(r);
  float gc = d_clamp01(g);
  float bc = d_clamp01(b);
  float lum = d_clamp01(d_luminance(rc, gc, bc));
  constexpr float kShadowPivot = 0.42f;
  constexpr float kShadowPower = 1.30f;
  constexpr float kShadowGainMax = 2.40f;
  float shadow_w = powf(
      d_clamp01((kShadowPivot - lum) / fmaxf(1e-6f, kShadowPivot)),
      kShadowPower);
  float gain = 1.0f + (kShadowGainMax - 1.0f) * shadow_w;
  out_r = d_detection_gain_curve(rc, gain);
  out_g = d_detection_gain_curve(gc, gain);
  out_b = d_detection_gain_curve(bc, gain);
}

__device__ __forceinline__ float d_lab_f(float t) {
  constexpr float kDelta = 6.0f / 29.0f;
  constexpr float kDelta2 = kDelta * kDelta;
  constexpr float kDelta3 = kDelta * kDelta * kDelta;
  if (t <= 0.0f)
    return 4.0f / 29.0f;
  if (t <= kDelta3)
    return t / (3.0f * kDelta2) + 4.0f / 29.0f;
  return cbrtf(t);
}

__device__ void d_rgb_to_lab(float r, float g, float b, float &L, float &A,
                             float &B) {
  float rc = d_clamp01(r);
  float gc = d_clamp01(g);
  float bc = d_clamp01(b);

  float x = 0.4124564f * rc + 0.3575761f * gc + 0.1804375f * bc;
  float y = 0.2126729f * rc + 0.7151522f * gc + 0.0721750f * bc;
  float z = 0.0193339f * rc + 0.1191920f * gc + 0.9503041f * bc;

  constexpr float kXn = 0.95047f;
  constexpr float kYn = 1.0f;
  constexpr float kZn = 1.08883f;

  float fx = d_lab_f(fmaxf(0.0f, x / kXn));
  float fy = d_lab_f(fmaxf(0.0f, y / kYn));
  float fz = d_lab_f(fmaxf(0.0f, z / kZn));

  L = 116.0f * fy - 16.0f;
  A = 500.0f * (fx - fy);
  B = 200.0f * (fy - fz);
}

// ---------------------------------------------------------------------------
// Kernel 1: Edge map  (block 16x16, shared memory 18x18 x 4 floats)
// Uses fast luma/chroma gradient approximation (Y,U,V) for speed.
// ---------------------------------------------------------------------------

__global__ void edge_map_kernel(const float *__restrict__ source_rgba,
                                const float *__restrict__ guide_luma_edge,
                                float *__restrict__ guide_edge, int width,
                                int height, float hard_stop_mix,
                                float edge_gate, float edge_norm_scale,
                                float blur_edge_t) {
  // Shared memory: (bw+2)*(bh+2) pixels, each storing {L, a, b, Y_orig}
  extern __shared__ float s_data[];

  const int bw = blockDim.x; // 16
  const int bh = blockDim.y; // 16
  const int sw = bw + 2;
  const int sh = bh + 2;
  const int tid = threadIdx.y * bw + threadIdx.x;
  const int total_threads = bw * bh;
  const int total_smem = sw * sh;

  // Cooperatively load the tile + 1-pixel halo into shared memory
  for (int i = tid; i < total_smem; i += total_threads) {
    int sx = i % sw;
    int sy = i / sw;
    int gx = static_cast<int>(blockIdx.x) * bw + sx - 1;
    int gy = static_cast<int>(blockIdx.y) * bh + sy - 1;
    gx = max(0, min(width - 1, gx));
    gy = max(0, min(height - 1, gy));

    const int pidx = (gy * width + gx) * 4;
    float r = source_rgba[pidx + 0];
    float g = source_rgba[pidx + 1];
    float b = source_rgba[pidx + 2];
    float rc = d_clamp01(r);
    float gc = d_clamp01(g);
    float bc = d_clamp01(b);
    float lum_orig = d_luminance(rc, gc, bc);
    float dr = rc;
    float dg = gc;
    float db = bc;
    d_detection_preboost_rgb(rc, gc, bc, dr, dg, db);
    float L = 0.0f;
    float A = 0.0f;
    float B = 0.0f;
    d_rgb_to_lab(dr, dg, db, L, A, B);

    int base = i * 4;
    s_data[base + 0] = L;
    s_data[base + 1] = A;
    s_data[base + 2] = B;
    s_data[base + 3] = lum_orig;
  }

  __syncthreads();

  // Compute edge for this pixel
  int gx = static_cast<int>(blockIdx.x) * bw + static_cast<int>(threadIdx.x);
  int gy = static_cast<int>(blockIdx.y) * bh + static_cast<int>(threadIdx.y);
  if (gx >= width || gy >= height)
    return;

  // Shared-memory indices (offset +1 for halo)
  int sc = ((threadIdx.y + 1) * sw + (threadIdx.x + 1)) * 4; // center
  int sl = ((threadIdx.y + 1) * sw + (threadIdx.x)) * 4;     // left
  int sr = ((threadIdx.y + 1) * sw + (threadIdx.x + 2)) * 4; // right
  int su = ((threadIdx.y) * sw + (threadIdx.x + 1)) * 4;     // up
  int sd = ((threadIdx.y + 2) * sw + (threadIdx.x + 1)) * 4; // down

  // Lab gradient: max of horizontal and vertical pair distance
  float dl_lr = s_data[sl + 0] - s_data[sr + 0];
  float da_lr = s_data[sl + 1] - s_data[sr + 1];
  float db_lr = s_data[sl + 2] - s_data[sr + 2];
  float delta_lr2 = dl_lr * dl_lr + da_lr * da_lr + db_lr * db_lr;

  float dl_ud = s_data[su + 0] - s_data[sd + 0];
  float da_ud = s_data[su + 1] - s_data[sd + 1];
  float db_ud = s_data[su + 2] - s_data[sd + 2];
  float delta_ud2 = dl_ud * dl_ud + da_ud * da_ud + db_ud * db_ud;

  constexpr float kEdgeFloor = 1.1f;
  constexpr float kEdgeGain = 0.09f;
  constexpr float kEdgeGamma = 0.6f;
  constexpr float kHardStopEdgeValue = 48.0f;
  constexpr float kDarkSpeckleLuma = 0.06f;
  constexpr float kDarkSpeckleContrast = 0.03f;
  constexpr float kDarkSpeckleStrength = 0.90f;
  constexpr float kGuideEdgeGamma = 0.72f;
  constexpr float kGuideFloor1 = 0.0035f;
  constexpr float kGuideGain1 = 8.5f;
  constexpr float kGuideFloor2 = 0.0027f;
  constexpr float kGuideGain2 = 7.5f;
  constexpr float kGuideFloor4 = 0.0019f;
  constexpr float kGuideGain4 = 6.5f;
  constexpr float kGuideFloor8 = 0.0013f;
  constexpr float kGuideGain8 = 5.4f;
  constexpr float kGuideRelFloor = 0.045f;
  constexpr float kGuideRelGain = 1.05f;
  constexpr float kGuideRelGamma = 0.80f;
  constexpr float kDarkBoostLuma = 0.24f;
  constexpr float kDarkBoostStrength = 0.38f;
  const float guide_w1 = 1.0f - 0.45f * blur_edge_t;
  const float guide_w2 = 0.22f + 0.40f * blur_edge_t;
  const float guide_w4 = 0.05f + 0.58f * blur_edge_t;
  const float guide_w8 = 0.00f + 0.62f * blur_edge_t;

  float raw_edge = sqrtf(fmaxf(delta_lr2, delta_ud2)) - kEdgeFloor;
  float source_edge =
      d_clamp01(powf(fmaxf(0.0f, raw_edge) * kEdgeGain, kEdgeGamma));
  float lum_center = d_clamp01(s_data[sc + 3]);
  float lum_neighbors =
      (d_clamp01(s_data[sl + 3]) + d_clamp01(s_data[sr + 3]) +
       d_clamp01(s_data[su + 3]) + d_clamp01(s_data[sd + 3])) * 0.25f;
  const int idx = gy * width + gx;
  const int x_l = max(0, gx - 1);
  const int x_r = min(width - 1, gx + 1);
  const int y_u = max(0, gy - 1);
  const int y_d = min(height - 1, gy + 1);
  const int x_l2 = max(0, gx - 2);
  const int x_r2 = min(width - 1, gx + 2);
  const int y_u2 = max(0, gy - 2);
  const int y_d2 = min(height - 1, gy + 2);
  const int x_l4 = max(0, gx - 4);
  const int x_r4 = min(width - 1, gx + 4);
  const int y_u4 = max(0, gy - 4);
  const int y_d4 = min(height - 1, gy + 4);
  const int x_l8 = max(0, gx - 8);
  const int x_r8 = min(width - 1, gx + 8);
  const int y_u8 = max(0, gy - 8);
  const int y_d8 = min(height - 1, gy + 8);
  const int i_l = gy * width + x_l;
  const int i_r = gy * width + x_r;
  const int i_u = y_u * width + gx;
  const int i_d = y_d * width + gx;
  const int i_l2 = gy * width + x_l2;
  const int i_r2 = gy * width + x_r2;
  const int i_u2 = y_u2 * width + gx;
  const int i_d2 = y_d2 * width + gx;
  const int i_l4 = gy * width + x_l4;
  const int i_r4 = gy * width + x_r4;
  const int i_u4 = y_u4 * width + gx;
  const int i_d4 = y_d4 * width + gx;
  const int i_l8 = gy * width + x_l8;
  const int i_r8 = gy * width + x_r8;
  const int i_u8 = y_u8 * width + gx;
  const int i_d8 = y_d8 * width + gx;
  const float g1 = fmaxf(
      fabsf(guide_luma_edge[i_l] - guide_luma_edge[i_r]),
      fabsf(guide_luma_edge[i_u] - guide_luma_edge[i_d]));
  const float g2 = fmaxf(
      fabsf(guide_luma_edge[i_l2] - guide_luma_edge[i_r2]),
      fabsf(guide_luma_edge[i_u2] - guide_luma_edge[i_d2]));
  const float g4 = fmaxf(
      fabsf(guide_luma_edge[i_l4] - guide_luma_edge[i_r4]),
      fabsf(guide_luma_edge[i_u4] - guide_luma_edge[i_d4]));
  const float g8 = fmaxf(
      fabsf(guide_luma_edge[i_l8] - guide_luma_edge[i_r8]),
      fabsf(guide_luma_edge[i_u8] - guide_luma_edge[i_d8]));
  const float e1 = d_clamp01(
      powf(fmaxf(0.0f, g1 - kGuideFloor1) * kGuideGain1, kGuideEdgeGamma));
  const float e2 = d_clamp01(
      powf(fmaxf(0.0f, g2 - kGuideFloor2) * kGuideGain2, kGuideEdgeGamma));
  const float e4 = d_clamp01(
      powf(fmaxf(0.0f, g4 - kGuideFloor4) * kGuideGain4, kGuideEdgeGamma));
  const float e8 = d_clamp01(
      powf(fmaxf(0.0f, g8 - kGuideFloor8) * kGuideGain8, kGuideEdgeGamma));
  float guide_edge_term =
      fmaxf(fmaxf(guide_w1 * e1, guide_w2 * e2),
            fmaxf(guide_w4 * e4, guide_w8 * e8));
  guide_edge_term = d_clamp01(guide_edge_term);
  const float noise_floor =
      0.0018f + 0.0065f * powf(1.0f - lum_center, 1.35f);
  const float noise_gate = d_clamp01((g1 - noise_floor) / 0.03f);
  guide_edge_term *= (0.30f + 0.70f * noise_gate);
  const float guide_rel = g1 / fmaxf(0.010f, lum_neighbors + 0.03f);
  const float guide_rel_edge = d_clamp01(
      powf(fmaxf(0.0f, guide_rel - kGuideRelFloor) * kGuideRelGain, kGuideRelGamma));
  const float dark_zone =
      d_clamp01((kDarkBoostLuma - lum_center) / fmaxf(1e-6f, kDarkBoostLuma));
  const float dark_boost_gate = d_clamp01((g1 - 0.0015f) / 0.025f);
  const float dark_boost =
      1.0f + kDarkBoostStrength * dark_zone * dark_boost_gate;

  source_edge = fmaxf(source_edge, guide_edge_term);
  source_edge = fmaxf(source_edge, guide_rel_edge * dark_zone);

  float dark_amount =
      d_clamp01((kDarkSpeckleLuma - lum_center) /
                fmaxf(1e-6f, kDarkSpeckleLuma));
  float isolate_amount =
      d_clamp01((lum_neighbors - lum_center - kDarkSpeckleContrast) / 0.22f);
  float dark_speckle = dark_amount * isolate_amount;
  if (dark_speckle > 1e-5f) {
    const float preserve_edges = d_clamp01(source_edge * 1.45f);
    const float speckle_strength =
        kDarkSpeckleStrength * (1.0f - 0.80f * preserve_edges);
    source_edge *= fmaxf(0.12f, 1.0f - speckle_strength * dark_speckle);
  }
  source_edge = d_clamp01(source_edge * dark_boost);

  // Hard-stop edge gate
  if (hard_stop_mix > 1e-4f && dark_speckle < 0.45f &&
      source_edge > edge_gate) {
    float stop_value = 1.0f + (kHardStopEdgeValue - 1.0f) * hard_stop_mix;
    source_edge = stop_value;
  } else {
    source_edge = fminf(source_edge * edge_norm_scale, 1.0f);
  }

  guide_edge[idx] = source_edge;
}

__global__ void guide_luma_edge_kernel(const float *__restrict__ guide_luma,
                                       float *__restrict__ guide_luma_edge,
                                       int pixel_count) {
  const int i = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
  if (i >= pixel_count) {
    return;
  }
  guide_luma_edge[i] = d_edge_detect_luma(guide_luma[i]);
}

__global__ void source_luma_kernel(const float *__restrict__ source_rgba,
                                   float *__restrict__ guide_luma,
                                   int pixel_count) {
  const int i = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
  if (i >= pixel_count) {
    return;
  }
  const float4 rgba = reinterpret_cast<const float4*>(source_rgba)[i];
  guide_luma[i] = d_luminance(rgba.x, rgba.y, rgba.z);
}

__global__ void guide_luma_mix_kernel(const float *__restrict__ source_rgba,
                                      const float *__restrict__ guide_input,
                                      float *__restrict__ guide_luma,
                                      int pixel_count,
                                      int guide_components,
                                      int guide_mode_rgb,
                                      float guide_mix) {
  const int i = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
  if (i >= pixel_count) {
    return;
  }

  const float4 rgba = reinterpret_cast<const float4*>(source_rgba)[i];
  const float source_luma = d_luminance(rgba.x, rgba.y, rgba.z);
  if (!guide_input || guide_components <= 0 || guide_mix <= 1e-6f) {
    guide_luma[i] = source_luma;
    return;
  }

  const float* guide_px = guide_input + static_cast<size_t>(i) * static_cast<size_t>(guide_components);
  float guide_value = 0.0f;
  if (guide_mode_rgb != 0 && guide_components >= 3) {
    guide_value = d_luminance(guide_px[0], guide_px[1], guide_px[2]);
  } else {
    guide_value = guide_px[0];
  }

  if (guide_mix >= 0.999f) {
    guide_luma[i] = guide_value;
  } else {
    guide_luma[i] = source_luma + (guide_value - source_luma) * guide_mix;
  }
}

__global__ void mask_extract_kernel(const float *__restrict__ mask_input,
                                    float *__restrict__ mask_output,
                                    int pixel_count,
                                    int mask_components,
                                    int mask_red_chan,
                                    int mask_alpha_chan,
                                    int invert_mask) {
  const int i = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
  if (i >= pixel_count) {
    return;
  }

  float mv = 1.0f;
  if (!mask_input || mask_components <= 0) {
    mv = 1.0f;
  } else {
    const float* mask_px =
        mask_input + static_cast<size_t>(i) * static_cast<size_t>(mask_components);
    if (mask_alpha_chan >= 0 && mask_alpha_chan < mask_components) {
      mv = mask_px[mask_alpha_chan];
    } else if (mask_red_chan >= 0 && mask_red_chan < mask_components) {
      mv = mask_px[mask_red_chan];
    } else {
      mv = 0.0f;
    }
  }

  mv = d_clamp01(mv);
  if (invert_mask != 0) {
    mv = 1.0f - mv;
  }
  mask_output[i] = mv;
}

// ---------------------------------------------------------------------------
// Kernel 2a: Despeckle horizontal accumulation
// ---------------------------------------------------------------------------

__global__ void despeckle_h_kernel(const float *__restrict__ guide_edge,
                                   float *__restrict__ h_sum,
                                   float *__restrict__ h_w, int width,
                                   int height) {
  int gx = blockIdx.x * blockDim.x + threadIdx.x;
  int gy = blockIdx.y * blockDim.y + threadIdx.y;
  if (gx >= width || gy >= height)
    return;

  int i = gy * width + gx;
  float center = fminf(guide_edge[i], 1.0f);
  float sum = 2.0f * center;
  float w = 2.0f;
  if (gx > 0) {
    sum += fminf(guide_edge[i - 1], 1.0f);
    w += 1.0f;
  }
  if (gx + 1 < width) {
    sum += fminf(guide_edge[i + 1], 1.0f);
    w += 1.0f;
  }
  h_sum[i] = sum;
  h_w[i] = w;
}

// ---------------------------------------------------------------------------
// Kernel 2b: Despeckle vertical accumulation
// ---------------------------------------------------------------------------

__global__ void despeckle_v_kernel(const float *__restrict__ h_sum,
                                   const float *__restrict__ h_w,
                                   const float *__restrict__ guide_edge,
                                   float *__restrict__ neigh, int width,
                                   int height) {
  int gx = blockIdx.x * blockDim.x + threadIdx.x;
  int gy = blockIdx.y * blockDim.y + threadIdx.y;
  if (gx >= width || gy >= height)
    return;

  int i = gy * width + gx;
  float sum = 2.0f * h_sum[i];
  float w = 2.0f * h_w[i];
  if (gy > 0) {
    int iu = i - width;
    sum += h_sum[iu];
    w += h_w[iu];
  }
  if (gy + 1 < height) {
    int id = i + width;
    sum += h_sum[id];
    w += h_w[id];
  }
  neigh[i] = (w > 0.0f) ? (sum / w) : fminf(guide_edge[i], 1.0f);
}

// ---------------------------------------------------------------------------
// Kernel 2c: Despeckle blend
// ---------------------------------------------------------------------------

__global__ void despeckle_blend_kernel(float *__restrict__ guide_edge,
                                       const float *__restrict__ neigh,
                                       float edge_despeckle_mix, int width,
                                       int height) {
  int gx = blockIdx.x * blockDim.x + threadIdx.x;
  int gy = blockIdx.y * blockDim.y + threadIdx.y;
  if (gx >= width || gy >= height)
    return;

  int i = gy * width + gx;
  float center = guide_edge[i];
  if (center > 1.0f)
    return; // preserve hard-stop edges

  float n = neigh[i];
  float cleaned = center + (n - center) * edge_despeckle_mix;
  if (center > n + 0.22f) {
    cleaned = fminf(cleaned, n + 0.08f);
  }
  guide_edge[i] = d_clamp01(cleaned);
}

// ---------------------------------------------------------------------------
// Kernel 3: Domain-transform filter (windowed, matches WGSL shader)
// ---------------------------------------------------------------------------

template<bool kHorizontal, bool kKeepAlpha>
__global__ void
filter_kernel(const float *__restrict__ src, float *__restrict__ dst,
              const float *__restrict__ guide_luma,
              const float *__restrict__ guide_edge, int width, int height,
              int radius, int sample_step, float inv2_sig_t, float k,
              float edge_weight) {
  int gx = blockIdx.x * blockDim.x + threadIdx.x;
  int gy = blockIdx.y * blockDim.y + threadIdx.y;
  if (gx >= width || gy >= height)
    return;

  int center = gy * width + gx;
  const float4* src4 = reinterpret_cast<const float4*>(src);
  float4 center_rgba = src4[center];

  float sum0 = center_rgba.x;
  float sum1 = center_rgba.y;
  float sum2 = center_rgba.z;
  float sum3 = 0.0f;
  if constexpr (!kKeepAlpha) {
    sum3 = center_rgba.w;
  }
  float wsum = 1.0f;

  float base_luma = guide_luma[center];

  // Forward direction
  {
    float t = 0.0f;
    float prev_luma = base_luma;
    int d = 1;
    if (sample_step <= 1) {
      while (d <= radius) {
        int sx, sy;
        if constexpr (kHorizontal) {
          sx = min(width - 1, gx + d);
          sy = gy;
        } else {
          sx = gx;
          sy = min(height - 1, gy + d);
        }

        const int sidx = sy * width + sx;
        const float cur_luma = guide_luma[sidx];
        const float cur_edge = guide_edge[sidx];

        t += 1.0f + (k * fabsf(cur_luma - prev_luma) + edge_weight * cur_edge);
        prev_luma = cur_luma;

        const float w = __expf(-(t * t) * inv2_sig_t);
        if (w < 1e-4f) {
          break;
        }

        const float4 srgba = src4[sidx];
        sum0 += srgba.x * w;
        sum1 += srgba.y * w;
        sum2 += srgba.z * w;
        if constexpr (!kKeepAlpha) {
          sum3 += srgba.w * w;
        }
        wsum += w;
        ++d;
      }
    } else {
      const int dense_limit = min(radius, 7);
      while (d <= dense_limit) {
        int sx, sy;
        if constexpr (kHorizontal) {
          sx = min(width - 1, gx + d);
          sy = gy;
        } else {
          sx = gx;
          sy = min(height - 1, gy + d);
        }

        const int sidx = sy * width + sx;
        const float cur_luma = guide_luma[sidx];
        const float cur_edge = guide_edge[sidx];

        t += 1.0f + (k * fabsf(cur_luma - prev_luma) + edge_weight * cur_edge);
        prev_luma = cur_luma;

        const float w = __expf(-(t * t) * inv2_sig_t);
        if (w < 1e-4f) {
          d = radius + 1;
          break;
        }

        const float4 srgba = src4[sidx];
        sum0 += srgba.x * w;
        sum1 += srgba.y * w;
        sum2 += srgba.z * w;
        if constexpr (!kKeepAlpha) {
          sum3 += srgba.w * w;
        }
        wsum += w;
        ++d;
      }

      if (d <= radius) {
        const float stride_f = static_cast<float>(sample_step);
        const float sparse_comp = 1.0f + 0.5f * (stride_f - 1.0f);
        while (d <= radius) {
          int sx, sy;
          if constexpr (kHorizontal) {
            sx = min(width - 1, gx + d);
            sy = gy;
          } else {
            sx = gx;
            sy = min(height - 1, gy + d);
          }

          const int sidx = sy * width + sx;
          const float cur_luma = guide_luma[sidx];
          const float cur_edge = guide_edge[sidx];

          t += stride_f +
               (k * fabsf(cur_luma - prev_luma) + edge_weight * cur_edge) * stride_f;
          prev_luma = cur_luma;

          float w = __expf(-(t * t) * inv2_sig_t);
          if (w < 1e-4f) {
            break;
          }
          w *= sparse_comp;

          const float4 srgba = src4[sidx];
          sum0 += srgba.x * w;
          sum1 += srgba.y * w;
          sum2 += srgba.z * w;
          if constexpr (!kKeepAlpha) {
            sum3 += srgba.w * w;
          }
          wsum += w;
          d += sample_step;
        }
      }
    }
  }

  // Backward direction
  {
    float t = 0.0f;
    float prev_luma = base_luma;
    int d = 1;
    if (sample_step <= 1) {
      while (d <= radius) {
        int sx, sy;
        if constexpr (kHorizontal) {
          sx = (gx >= d) ? (gx - d) : 0;
          sy = gy;
        } else {
          sx = gx;
          sy = (gy >= d) ? (gy - d) : 0;
        }

        const int sidx = sy * width + sx;
        const float cur_luma = guide_luma[sidx];
        const float cur_edge = guide_edge[sidx];

        t += 1.0f + (k * fabsf(cur_luma - prev_luma) + edge_weight * cur_edge);
        prev_luma = cur_luma;

        const float w = __expf(-(t * t) * inv2_sig_t);
        if (w < 1e-4f) {
          break;
        }

        const float4 srgba = src4[sidx];
        sum0 += srgba.x * w;
        sum1 += srgba.y * w;
        sum2 += srgba.z * w;
        if constexpr (!kKeepAlpha) {
          sum3 += srgba.w * w;
        }
        wsum += w;
        ++d;
      }
    } else {
      const int dense_limit = min(radius, 7);
      while (d <= dense_limit) {
        int sx, sy;
        if constexpr (kHorizontal) {
          sx = (gx >= d) ? (gx - d) : 0;
          sy = gy;
        } else {
          sx = gx;
          sy = (gy >= d) ? (gy - d) : 0;
        }

        const int sidx = sy * width + sx;
        const float cur_luma = guide_luma[sidx];
        const float cur_edge = guide_edge[sidx];

        t += 1.0f + (k * fabsf(cur_luma - prev_luma) + edge_weight * cur_edge);
        prev_luma = cur_luma;

        const float w = __expf(-(t * t) * inv2_sig_t);
        if (w < 1e-4f) {
          d = radius + 1;
          break;
        }

        const float4 srgba = src4[sidx];
        sum0 += srgba.x * w;
        sum1 += srgba.y * w;
        sum2 += srgba.z * w;
        if constexpr (!kKeepAlpha) {
          sum3 += srgba.w * w;
        }
        wsum += w;
        ++d;
      }

      if (d <= radius) {
        const float stride_f = static_cast<float>(sample_step);
        const float sparse_comp = 1.0f + 0.5f * (stride_f - 1.0f);
        while (d <= radius) {
          int sx, sy;
          if constexpr (kHorizontal) {
            sx = (gx >= d) ? (gx - d) : 0;
            sy = gy;
          } else {
            sx = gx;
            sy = (gy >= d) ? (gy - d) : 0;
          }

          const int sidx = sy * width + sx;
          const float cur_luma = guide_luma[sidx];
          const float cur_edge = guide_edge[sidx];

          t += stride_f +
               (k * fabsf(cur_luma - prev_luma) + edge_weight * cur_edge) * stride_f;
          prev_luma = cur_luma;

          float w = __expf(-(t * t) * inv2_sig_t);
          if (w < 1e-4f) {
            break;
          }
          w *= sparse_comp;

          const float4 srgba = src4[sidx];
          sum0 += srgba.x * w;
          sum1 += srgba.y * w;
          sum2 += srgba.z * w;
          if constexpr (!kKeepAlpha) {
            sum3 += srgba.w * w;
          }
          wsum += w;
          d += sample_step;
        }
      }
    }
  }

  float inv_w = 1.0f / fmaxf(wsum, 1e-8f);
  float4 outv;
  outv.x = sum0 * inv_w;
  outv.y = sum1 * inv_w;
  outv.z = sum2 * inv_w;
  if constexpr (kKeepAlpha) {
    outv.w = center_rgba.w;
  } else {
    outv.w = sum3 * inv_w;
  }
  reinterpret_cast<float4*>(dst)[center] = outv;
}

template<bool kHorizontal>
static inline void launch_filter_kernel(
    cudaStream_t stream,
    dim3 grid,
    dim3 block,
    const float* src,
    float* dst,
    const float* guide_luma,
    const float* guide_edge,
    int width,
    int height,
    int radius,
    int sample_step,
    float inv2_sig_t,
    float k,
    float edge_weight,
    int keep_alpha) {
  if (keep_alpha != 0) {
    filter_kernel<kHorizontal, true><<<grid, block, 0, stream>>>(
        src, dst, guide_luma, guide_edge, width, height, radius, sample_step,
        inv2_sig_t, k, edge_weight);
  } else {
    filter_kernel<kHorizontal, false><<<grid, block, 0, stream>>>(
        src, dst, guide_luma, guide_edge, width, height, radius, sample_step,
        inv2_sig_t, k, edge_weight);
  }
}

// ---------------------------------------------------------------------------
// Kernel 4: Blend
// ---------------------------------------------------------------------------

__global__ void blend_kernel(const float *__restrict__ source,
                             const float *__restrict__ filtered,
                             const float *__restrict__ mask,
                             float *__restrict__ output, float mix_val,
                             int width, int height, int keep_alpha) {
  int gx = blockIdx.x * blockDim.x + threadIdx.x;
  int gy = blockIdx.y * blockDim.y + threadIdx.y;
  if (gx >= width || gy >= height)
    return;

  const int i = gy * width + gx;
  const float mv = mask ? mask[i] : 1.0f;
  const float blend = d_clamp01(mix_val * mv);

  const float4 src4 = reinterpret_cast<const float4*>(source)[i];
  const float4 flt4 = reinterpret_cast<const float4*>(filtered)[i];

  float4 outv;
  outv.x = src4.x + (flt4.x - src4.x) * blend;
  outv.y = src4.y + (flt4.y - src4.y) * blend;
  outv.z = src4.z + (flt4.z - src4.z) * blend;
  if (keep_alpha != 0) {
    outv.w = src4.w;
  } else {
    outv.w = src4.w + (flt4.w - src4.w) * blend;
  }
  reinterpret_cast<float4*>(output)[i] = outv;
}

__global__ void guide_edge_preview_kernel(const float *__restrict__ guide_edge,
                                          const float *__restrict__ source,
                                          float *__restrict__ output,
                                          int width,
                                          int height,
                                          int keep_alpha) {
  const int gx = blockIdx.x * blockDim.x + threadIdx.x;
  const int gy = blockIdx.y * blockDim.y + threadIdx.y;
  if (gx >= width || gy >= height) {
    return;
  }
  const int i = gy * width + gx;
  const float v = d_clamp01(guide_edge[i]);
  float4 outv;
  outv.x = v;
  outv.y = v;
  outv.z = v;
  if (keep_alpha != 0 && source) {
    outv.w = reinterpret_cast<const float4*>(source)[i].w;
  } else {
    outv.w = 1.0f;
  }
  reinterpret_cast<float4*>(output)[i] = outv;
}

__global__ void organic_cleanup_kernel(const float *__restrict__ input,
                                       float *__restrict__ output,
                                       int width,
                                       int height,
                                       int channel_count,
                                       float blend) {
  const int gx = blockIdx.x * blockDim.x + threadIdx.x;
  const int gy = blockIdx.y * blockDim.y + threadIdx.y;
  if (gx >= width || gy >= height) {
    return;
  }

  const int xl = max(0, gx - 1);
  const int xr = min(width - 1, gx + 1);
  const int yu = max(0, gy - 1);
  const int yd = min(height - 1, gy + 1);

  const int c = gy * width + gx;
  const int l = gy * width + xl;
  const int r = gy * width + xr;
  const int u = yu * width + gx;
  const int d = yd * width + gx;
  const int ul = yu * width + xl;
  const int ur = yu * width + xr;
  const int dl = yd * width + xl;
  const int dr = yd * width + xr;

  const float4 center = reinterpret_cast<const float4*>(input)[c];
  const float4 left = reinterpret_cast<const float4*>(input)[l];
  const float4 right = reinterpret_cast<const float4*>(input)[r];
  const float4 up = reinterpret_cast<const float4*>(input)[u];
  const float4 down = reinterpret_cast<const float4*>(input)[d];
  const float4 up_left = reinterpret_cast<const float4*>(input)[ul];
  const float4 up_right = reinterpret_cast<const float4*>(input)[ur];
  const float4 down_left = reinterpret_cast<const float4*>(input)[dl];
  const float4 down_right = reinterpret_cast<const float4*>(input)[dr];

  float4 blurred;
  blurred.x =
      (4.0f * center.x +
       2.0f * (left.x + right.x + up.x + down.x) +
       (up_left.x + up_right.x + down_left.x + down_right.x)) * (1.0f / 16.0f);
  blurred.y =
      (4.0f * center.y +
       2.0f * (left.y + right.y + up.y + down.y) +
       (up_left.y + up_right.y + down_left.y + down_right.y)) * (1.0f / 16.0f);
  blurred.z =
      (4.0f * center.z +
       2.0f * (left.z + right.z + up.z + down.z) +
       (up_left.z + up_right.z + down_left.z + down_right.z)) * (1.0f / 16.0f);
  blurred.w =
      (4.0f * center.w +
       2.0f * (left.w + right.w + up.w + down.w) +
       (up_left.w + up_right.w + down_left.w + down_right.w)) * (1.0f / 16.0f);

  float4 outv = center;
  outv.x = center.x + (blurred.x - center.x) * blend;
  outv.y = center.y + (blurred.y - center.y) * blend;
  outv.z = center.z + (blurred.z - center.z) * blend;
  if (channel_count >= 4) {
    outv.w = center.w + (blurred.w - center.w) * blend;
  } else {
    outv.w = center.w;
  }

  reinterpret_cast<float4*>(output)[c] = outv;
}

// ---------------------------------------------------------------------------
// Deinterleave kernels — replace host-side scatter for non-packed planes.
//
// Plane buffer is read with arbitrary `row_stride` and `col_stride` (in floats),
// channels selected by `chan_*` indices (-1 = absent → fallback). Bounds are
// always checked so a malformed plane can't read out of bounds.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float fetch_plane(
    const float* plane, int plane_size, int idx, float fallback) {
  return (idx >= 0 && idx < plane_size) ? d_finite_or(plane[idx], fallback) : fallback;
}

__global__ void deinterleave_rgba_kernel(
    const float* __restrict__ plane_data, int plane_size,
    float* __restrict__ d_source_rgba,
    int width, int height,
    int row_stride, int col_stride,
    int chan_r, int chan_g, int chan_b, int chan_a) {
  int gx = blockIdx.x * blockDim.x + threadIdx.x;
  int gy = blockIdx.y * blockDim.y + threadIdx.y;
  if (gx >= width || gy >= height) return;

  const int src_base = gy * row_stride + gx * col_stride;
  const int dst = (gy * width + gx) * 4;

  d_source_rgba[dst + 0] = (chan_r >= 0)
      ? fetch_plane(plane_data, plane_size, src_base + chan_r, 0.0f) : 0.0f;
  d_source_rgba[dst + 1] = (chan_g >= 0)
      ? fetch_plane(plane_data, plane_size, src_base + chan_g, 0.0f) : 0.0f;
  d_source_rgba[dst + 2] = (chan_b >= 0)
      ? fetch_plane(plane_data, plane_size, src_base + chan_b, 0.0f) : 0.0f;
  d_source_rgba[dst + 3] = (chan_a >= 0)
      ? fetch_plane(plane_data, plane_size, src_base + chan_a, 1.0f) : 1.0f;
}

__global__ void deinterleave_rgb_kernel(
    const float* __restrict__ plane_data, int plane_size,
    float* __restrict__ d_guide_rgb,
    int width, int height,
    int row_stride, int col_stride,
    int chan_r, int chan_g, int chan_b) {
  int gx = blockIdx.x * blockDim.x + threadIdx.x;
  int gy = blockIdx.y * blockDim.y + threadIdx.y;
  if (gx >= width || gy >= height) return;

  const int src_base = gy * row_stride + gx * col_stride;
  const int dst = (gy * width + gx) * 3;

  // If chan_g/chan_b are missing, replicate red (matches host behaviour).
  const float r = (chan_r >= 0)
      ? fetch_plane(plane_data, plane_size, src_base + chan_r, 0.0f) : 0.0f;
  const float g = (chan_g >= 0)
      ? fetch_plane(plane_data, plane_size, src_base + chan_g, 0.0f) : r;
  const float b = (chan_b >= 0)
      ? fetch_plane(plane_data, plane_size, src_base + chan_b, 0.0f) : r;

  d_guide_rgb[dst + 0] = r;
  d_guide_rgb[dst + 1] = g;
  d_guide_rgb[dst + 2] = b;
}

__global__ void deinterleave_scalar_kernel(
    const float* __restrict__ plane_data, int plane_size,
    float* __restrict__ d_out,
    int width, int height,
    int row_stride, int col_stride,
    int chan_pick) {
  int gx = blockIdx.x * blockDim.x + threadIdx.x;
  int gy = blockIdx.y * blockDim.y + threadIdx.y;
  if (gx >= width || gy >= height) return;

  const int src_base = gy * row_stride + gx * col_stride;
  const int dst = gy * width + gx;
  d_out[dst] = (chan_pick >= 0)
      ? fetch_plane(plane_data, plane_size, src_base + chan_pick, 0.0f) : 0.0f;
}

// Two-component (R+A) plane → packed [R, A] interleaved.
__global__ void deinterleave_mask2_kernel(
    const float* __restrict__ plane_data, int plane_size,
    float* __restrict__ d_out,
    int width, int height,
    int row_stride, int col_stride,
    int chan_r, int chan_a) {
  int gx = blockIdx.x * blockDim.x + threadIdx.x;
  int gy = blockIdx.y * blockDim.y + threadIdx.y;
  if (gx >= width || gy >= height) return;

  const int src_base = gy * row_stride + gx * col_stride;
  const int dst = (gy * width + gx) * 2;
  d_out[dst + 0] = (chan_r >= 0)
      ? fetch_plane(plane_data, plane_size, src_base + chan_r, 0.0f) : 0.0f;
  d_out[dst + 1] = (chan_a >= 0)
      ? fetch_plane(plane_data, plane_size, src_base + chan_a, 0.0f) : 0.0f;
}

// Sanitize an already-packed RGBA buffer (NaN/Inf → 0/0/0/1). Replaces the
// host-side `sanitize_rgba_buffer_in_place`.
__global__ void sanitize_rgba_kernel(float* __restrict__ rgba, int pixel_count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= pixel_count) return;
  const int p = i * 4;
  rgba[p + 0] = d_finite_or(rgba[p + 0], 0.0f);
  rgba[p + 1] = d_finite_or(rgba[p + 1], 0.0f);
  rgba[p + 2] = d_finite_or(rgba[p + 2], 0.0f);
  rgba[p + 3] = d_finite_or(rgba[p + 3], 1.0f);
}

// ---------------------------------------------------------------------------
// Host-side backend state
// ---------------------------------------------------------------------------

// Pinned host buffer that grows on demand. Allocated with cudaHostAlloc to
// enable real async H2D / D2H transfers (pageable memory falls back to a
// driver bounce-buffer copy that's effectively synchronous).
struct PinnedBuf {
  float* ptr = nullptr;
  size_t capacity_floats = 0;
};

// Device-side scratch buffer that grows on demand (used for planar plane
// uploads when sizes vary).
struct DeviceScratch {
  float* ptr = nullptr;
  size_t capacity_floats = 0;
};

struct CudaBackend {
  bool initialized = false;
  int device_id = -1;
  int width = 0;
  int height = 0;

  // Fixed W*H device buffers (re-allocated when (W,H) changes).
  float* d_source_rgba = nullptr;   // 4ch
  float* d_guide_input = nullptr;   // 3ch (RGB) or 1ch (luma) staging
  float* d_guide_luma = nullptr;    // 1ch — used by edge map and filter
  float* d_mask_input = nullptr;    // 2ch staging
  float* d_mask = nullptr;          // 1ch — used by blend
  float* d_guide_edge = nullptr;    // 1ch — edge map
  float* d_edge_temp_a = nullptr;   // despeckle scratch
  float* d_edge_temp_b = nullptr;   // despeckle scratch
  float* d_filter_a = nullptr;      // 4ch ping
  float* d_filter_b = nullptr;      // 4ch pong / blend output

  // Variable-size device scratch for raw planar uploads.
  DeviceScratch d_planar_source;
  DeviceScratch d_planar_guide;
  DeviceScratch d_planar_mask;

  // Pinned host staging — separate per role so concurrent uploads on the same
  // stream don't clobber each other's source memory before the H2D fires.
  PinnedBuf h_source_pinned;
  PinnedBuf h_guide_pinned;
  PinnedBuf h_mask_pinned;
  PinnedBuf h_output_pinned;

  cudaStream_t stream = nullptr;
  SpinLock mutex;
};

static CudaBackend g_backend;

static bool activate_device(const CudaBackend& b) {
  if (b.device_id < 0) return false;
  return cudaSetDevice(b.device_id) == cudaSuccess;
}

static bool ensure_pinned(PinnedBuf& b, size_t needed_floats) {
  if (b.capacity_floats >= needed_floats) return true;
  if (b.ptr) {
    cudaFreeHost(b.ptr);
    b.ptr = nullptr;
    b.capacity_floats = 0;
  }
  // Round up to 256K floats (1 MiB) to amortize reallocs.
  constexpr size_t kAlignFloats = 256 * 1024;
  const size_t alloc =
      ((needed_floats + kAlignFloats - 1) / kAlignFloats) * kAlignFloats;
  void* p = nullptr;
  cudaError_t err = cudaHostAlloc(&p, alloc * sizeof(float), cudaHostAllocDefault);
  if (err != cudaSuccess) {
    cuda_logf("TBlur cudaHostAlloc(%zu floats) failed: %s",
              alloc, cudaGetErrorString(err));
    return false;
  }
  b.ptr = static_cast<float*>(p);
  b.capacity_floats = alloc;
  return true;
}

static bool ensure_device_scratch(DeviceScratch& s, size_t needed_floats) {
  if (s.capacity_floats >= needed_floats) return true;
  if (s.ptr) {
    cudaFree(s.ptr);
    s.ptr = nullptr;
    s.capacity_floats = 0;
  }
  constexpr size_t kAlignFloats = 256 * 1024;
  const size_t alloc =
      ((needed_floats + kAlignFloats - 1) / kAlignFloats) * kAlignFloats;
  cudaError_t err = cudaMalloc(&s.ptr, alloc * sizeof(float));
  if (err != cudaSuccess) {
    cuda_logf("TBlur cudaMalloc(scratch %zu floats) failed: %s",
              alloc, cudaGetErrorString(err));
    return false;
  }
  s.capacity_floats = alloc;
  return true;
}

static void free_buffers(CudaBackend& b) {
  if (b.device_id >= 0) {
    (void)cudaSetDevice(b.device_id);
  }
  if (b.d_source_rgba) { cudaFree(b.d_source_rgba); b.d_source_rgba = nullptr; }
  if (b.d_guide_input) { cudaFree(b.d_guide_input); b.d_guide_input = nullptr; }
  if (b.d_guide_luma)  { cudaFree(b.d_guide_luma);  b.d_guide_luma = nullptr; }
  if (b.d_mask_input)  { cudaFree(b.d_mask_input);   b.d_mask_input = nullptr; }
  if (b.d_mask)        { cudaFree(b.d_mask);         b.d_mask = nullptr; }
  if (b.d_guide_edge)  { cudaFree(b.d_guide_edge);   b.d_guide_edge = nullptr; }
  if (b.d_edge_temp_a) { cudaFree(b.d_edge_temp_a);  b.d_edge_temp_a = nullptr; }
  if (b.d_edge_temp_b) { cudaFree(b.d_edge_temp_b);  b.d_edge_temp_b = nullptr; }
  if (b.d_filter_a)    { cudaFree(b.d_filter_a);     b.d_filter_a = nullptr; }
  if (b.d_filter_b)    { cudaFree(b.d_filter_b);     b.d_filter_b = nullptr; }
  if (b.d_planar_source.ptr) { cudaFree(b.d_planar_source.ptr); b.d_planar_source = {}; }
  if (b.d_planar_guide.ptr)  { cudaFree(b.d_planar_guide.ptr);  b.d_planar_guide = {}; }
  if (b.d_planar_mask.ptr)   { cudaFree(b.d_planar_mask.ptr);   b.d_planar_mask = {}; }
  if (b.h_source_pinned.ptr) { cudaFreeHost(b.h_source_pinned.ptr); b.h_source_pinned = {}; }
  if (b.h_guide_pinned.ptr)  { cudaFreeHost(b.h_guide_pinned.ptr);  b.h_guide_pinned = {}; }
  if (b.h_mask_pinned.ptr)   { cudaFreeHost(b.h_mask_pinned.ptr);   b.h_mask_pinned = {}; }
  if (b.h_output_pinned.ptr) { cudaFreeHost(b.h_output_pinned.ptr); b.h_output_pinned = {}; }
  b.width = 0;
  b.height = 0;
}

static bool init_backend(CudaBackend& b) {
  if (b.initialized) return (b.device_id >= 0);
  b.initialized = true;

  const char* visible = std::getenv("CUDA_VISIBLE_DEVICES");
  if (visible && *visible) {
    cuda_logf("TBlur CUDA_VISIBLE_DEVICES=%s", visible);
  }

  cudaError_t init_err = cudaFree(nullptr);
  if (init_err != cudaSuccess) {
    cuda_logf("TBlur cudaFree(0) init failed: %d (%s)",
              static_cast<int>(init_err), cudaGetErrorString(init_err));
    return false;
  }

  int device_count = 0;
  cudaError_t count_err = cudaGetDeviceCount(&device_count);
  if (count_err != cudaSuccess) {
    cuda_logf("TBlur cudaGetDeviceCount failed: %d (%s)",
              static_cast<int>(count_err), cudaGetErrorString(count_err));
    return false;
  }
  if (device_count == 0) {
    cuda_logf("TBlur cudaGetDeviceCount returned 0 devices");
    return false;
  }

  int best = -1;
  size_t best_mem = 0;
  for (int i = 0; i < device_count; i++) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
      cuda_logf("TBlur CUDA device %d: %s, cc=%d.%d, vram=%llu MB",
                i, prop.name, prop.major, prop.minor,
                static_cast<unsigned long long>(prop.totalGlobalMem / (1024ull * 1024ull)));
      if (prop.totalGlobalMem > best_mem) {
        best = i;
        best_mem = prop.totalGlobalMem;
      }
    }
  }
  if (best < 0) return false;

  cudaError_t set_err = cudaSetDevice(best);
  if (set_err != cudaSuccess) {
    cuda_logf("TBlur cudaSetDevice(%d) failed: %d (%s)",
              best, static_cast<int>(set_err), cudaGetErrorString(set_err));
    return false;
  }
  cudaError_t stream_err = cudaStreamCreate(&b.stream);
  if (stream_err != cudaSuccess) {
    cuda_logf("TBlur cudaStreamCreate failed: %d (%s)",
              static_cast<int>(stream_err), cudaGetErrorString(stream_err));
    return false;
  }

  b.device_id = best;
  cuda_logf("TBlur CUDA init ok (device=%d)", best);
  return true;
}

static bool ensure_buffers(CudaBackend& b, int w, int h) {
  if (!activate_device(b)) return false;
  if (b.width == w && b.height == h && b.d_source_rgba != nullptr) return true;

  // Drop only the (W,H)-sized buffers; keep the variable-size scratches and
  // pinned bufs since they grow on demand independently.
  if (b.d_source_rgba) { cudaFree(b.d_source_rgba); b.d_source_rgba = nullptr; }
  if (b.d_guide_input) { cudaFree(b.d_guide_input); b.d_guide_input = nullptr; }
  if (b.d_guide_luma)  { cudaFree(b.d_guide_luma);  b.d_guide_luma = nullptr; }
  if (b.d_mask_input)  { cudaFree(b.d_mask_input);   b.d_mask_input = nullptr; }
  if (b.d_mask)        { cudaFree(b.d_mask);         b.d_mask = nullptr; }
  if (b.d_guide_edge)  { cudaFree(b.d_guide_edge);   b.d_guide_edge = nullptr; }
  if (b.d_edge_temp_a) { cudaFree(b.d_edge_temp_a);  b.d_edge_temp_a = nullptr; }
  if (b.d_edge_temp_b) { cudaFree(b.d_edge_temp_b);  b.d_edge_temp_b = nullptr; }
  if (b.d_filter_a)    { cudaFree(b.d_filter_a);     b.d_filter_a = nullptr; }
  if (b.d_filter_b)    { cudaFree(b.d_filter_b);     b.d_filter_b = nullptr; }

  size_t pixels = static_cast<size_t>(w) * static_cast<size_t>(h);

  if (cudaMalloc(&b.d_source_rgba, pixels * 4 * sizeof(float)) != cudaSuccess) goto fail;
  if (cudaMalloc(&b.d_guide_input, pixels * 3 * sizeof(float)) != cudaSuccess) goto fail;
  if (cudaMalloc(&b.d_guide_luma,  pixels *     sizeof(float)) != cudaSuccess) goto fail;
  if (cudaMalloc(&b.d_mask_input,  pixels * 2 * sizeof(float)) != cudaSuccess) goto fail;
  if (cudaMalloc(&b.d_mask,        pixels *     sizeof(float)) != cudaSuccess) goto fail;
  if (cudaMalloc(&b.d_guide_edge,  pixels *     sizeof(float)) != cudaSuccess) goto fail;
  if (cudaMalloc(&b.d_edge_temp_a, pixels *     sizeof(float)) != cudaSuccess) goto fail;
  if (cudaMalloc(&b.d_edge_temp_b, pixels *     sizeof(float)) != cudaSuccess) goto fail;
  if (cudaMalloc(&b.d_filter_a,    pixels * 4 * sizeof(float)) != cudaSuccess) goto fail;
  if (cudaMalloc(&b.d_filter_b,    pixels * 4 * sizeof(float)) != cudaSuccess) goto fail;

  // Pre-size pinned host bufs to typical per-frame upload (W*H*max-components).
  if (!ensure_pinned(b.h_source_pinned, pixels * 4)) goto fail;
  if (!ensure_pinned(b.h_guide_pinned,  pixels * 3)) goto fail;
  if (!ensure_pinned(b.h_mask_pinned,   pixels * 2)) goto fail;
  if (!ensure_pinned(b.h_output_pinned, pixels * 4)) goto fail;

  b.width = w;
  b.height = h;
  return true;

fail:
  free_buffers(b);
  return false;
}

// ---------------------------------------------------------------------------
// Upload helpers
// ---------------------------------------------------------------------------

// Upload a packed buffer of `n_floats` floats from `host_src` to `d_target`,
// staging through `pinned`.
static bool upload_packed(
    cudaStream_t stream, PinnedBuf& pinned, float* d_target,
    const float* host_src, size_t n_floats) {
  if (!ensure_pinned(pinned, n_floats)) return false;
  std::memcpy(pinned.ptr, host_src, n_floats * sizeof(float));
  CUDA_CHECK(cudaMemcpyAsync(d_target, pinned.ptr, n_floats * sizeof(float),
                             cudaMemcpyHostToDevice, stream));
  return true;
}

// Stage a planar plane in pinned host memory, async-copy to device scratch,
// then return a pointer to the scratch buffer for the caller to launch a
// deinterleave kernel from.
static bool stage_planar(
    cudaStream_t stream, PinnedBuf& pinned, DeviceScratch& scratch,
    const float* plane_data, size_t n_floats,
    float** out_d_scratch) {
  if (n_floats == 0) return false;
  if (!ensure_pinned(pinned, n_floats)) return false;
  if (!ensure_device_scratch(scratch, n_floats)) return false;
  std::memcpy(pinned.ptr, plane_data, n_floats * sizeof(float));
  CUDA_CHECK(cudaMemcpyAsync(scratch.ptr, pinned.ptr,
                             n_floats * sizeof(float),
                             cudaMemcpyHostToDevice, stream));
  *out_d_scratch = scratch.ptr;
  return true;
}

// Upload source. desc.packed_data → direct H2D into d_source_rgba; planar →
// stage in scratch then deinterleave into d_source_rgba.
static bool upload_source(
    CudaBackend& b, const TBlurInputDesc& desc, int width, int height) {
  const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
  if (desc.packed_data) {
    return upload_packed(b.stream, b.h_source_pinned, b.d_source_rgba,
                         desc.packed_data, pixels * 4);
  }
  if (!desc.planar_data || desc.plane_size_floats <= 0) return false;
  float* d_scratch = nullptr;
  if (!stage_planar(b.stream, b.h_source_pinned, b.d_planar_source,
                    desc.planar_data,
                    static_cast<size_t>(desc.plane_size_floats),
                    &d_scratch)) {
    return false;
  }
  dim3 block(16, 16);
  dim3 grid((width + 15) / 16, (height + 15) / 16);
  deinterleave_rgba_kernel<<<grid, block, 0, b.stream>>>(
      d_scratch, desc.plane_size_floats, b.d_source_rgba,
      width, height, desc.row_stride, desc.col_stride,
      desc.chan_a, desc.chan_b, desc.chan_c, desc.chan_d);
  CUDA_CHECK_KERNEL();
  return true;
}

// Upload guide as 1-channel directly into d_guide_luma.
static bool upload_guide_luma_direct(
    CudaBackend& b, const TBlurInputDesc& desc, int width, int height) {
  const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
  if (desc.packed_data) {
    return upload_packed(b.stream, b.h_guide_pinned, b.d_guide_luma,
                         desc.packed_data, pixels);
  }
  if (!desc.planar_data || desc.plane_size_floats <= 0) return false;
  float* d_scratch = nullptr;
  if (!stage_planar(b.stream, b.h_guide_pinned, b.d_planar_guide,
                    desc.planar_data,
                    static_cast<size_t>(desc.plane_size_floats),
                    &d_scratch)) {
    return false;
  }
  dim3 block(16, 16);
  dim3 grid((width + 15) / 16, (height + 15) / 16);
  deinterleave_scalar_kernel<<<grid, block, 0, b.stream>>>(
      d_scratch, desc.plane_size_floats, b.d_guide_luma,
      width, height, desc.row_stride, desc.col_stride, desc.chan_a);
  CUDA_CHECK_KERNEL();
  return true;
}

// Upload guide into d_guide_input as either 3ch (RGB) or 1ch (luma) for later
// mixing by guide_luma_mix_kernel.
static bool upload_guide_input(
    CudaBackend& b, const TBlurInputDesc& desc, int width, int height,
    int components) {
  const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
  if (desc.packed_data) {
    return upload_packed(b.stream, b.h_guide_pinned, b.d_guide_input,
                         desc.packed_data,
                         pixels * static_cast<size_t>(components));
  }
  if (!desc.planar_data || desc.plane_size_floats <= 0) return false;
  float* d_scratch = nullptr;
  if (!stage_planar(b.stream, b.h_guide_pinned, b.d_planar_guide,
                    desc.planar_data,
                    static_cast<size_t>(desc.plane_size_floats),
                    &d_scratch)) {
    return false;
  }
  dim3 block(16, 16);
  dim3 grid((width + 15) / 16, (height + 15) / 16);
  if (components >= 3) {
    deinterleave_rgb_kernel<<<grid, block, 0, b.stream>>>(
        d_scratch, desc.plane_size_floats, b.d_guide_input,
        width, height, desc.row_stride, desc.col_stride,
        desc.chan_a, desc.chan_b, desc.chan_c);
  } else {
    deinterleave_scalar_kernel<<<grid, block, 0, b.stream>>>(
        d_scratch, desc.plane_size_floats, b.d_guide_input,
        width, height, desc.row_stride, desc.col_stride, desc.chan_a);
  }
  CUDA_CHECK_KERNEL();
  return true;
}

// Upload mask into d_mask (1ch direct) or d_mask_input (2ch staging).
static bool upload_mask(
    CudaBackend& b, const TBlurInputDesc& desc, int width, int height,
    int components, bool direct_into_mask) {
  const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
  float* d_target = direct_into_mask ? b.d_mask : b.d_mask_input;
  if (desc.packed_data) {
    return upload_packed(b.stream, b.h_mask_pinned, d_target,
                         desc.packed_data,
                         pixels * static_cast<size_t>(components));
  }
  if (!desc.planar_data || desc.plane_size_floats <= 0) return false;
  float* d_scratch = nullptr;
  if (!stage_planar(b.stream, b.h_mask_pinned, b.d_planar_mask,
                    desc.planar_data,
                    static_cast<size_t>(desc.plane_size_floats),
                    &d_scratch)) {
    return false;
  }
  dim3 block(16, 16);
  dim3 grid((width + 15) / 16, (height + 15) / 16);
  if (direct_into_mask) {
    deinterleave_scalar_kernel<<<grid, block, 0, b.stream>>>(
        d_scratch, desc.plane_size_floats, d_target,
        width, height, desc.row_stride, desc.col_stride, desc.chan_a);
  } else {
    deinterleave_mask2_kernel<<<grid, block, 0, b.stream>>>(
        d_scratch, desc.plane_size_floats, d_target,
        width, height, desc.row_stride, desc.col_stride,
        desc.chan_a, desc.chan_b);
  }
  CUDA_CHECK_KERNEL();
  return true;
}

// ---------------------------------------------------------------------------`r`n// Public C API
// ---------------------------------------------------------------------------

extern "C" int cuda_is_available(void) {
  std::lock_guard<SpinLock> lock(g_backend.mutex);
  return init_backend(g_backend) ? 1 : 0;
}

extern "C" int cuda_get_device_name(char* out_name, int out_size) {
  std::lock_guard<SpinLock> lock(g_backend.mutex);
  if (!out_name || out_size <= 1) return 0;
  out_name[0] = '\0';
  if (!init_backend(g_backend)) return 0;
  if (!activate_device(g_backend)) return 0;
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, g_backend.device_id) != cudaSuccess) return 0;
  std::snprintf(out_name, static_cast<size_t>(out_size), "%s", prop.name);
  out_name[out_size - 1] = '\0';
  return 1;
}

extern "C" int cuda_prepare(int width, int height) {
  std::lock_guard<SpinLock> lock(g_backend.mutex);
  if (width <= 0 || height <= 0) return 0;
  if (!init_backend(g_backend)) return 0;
  if (!activate_device(g_backend)) return 0;
  if (!ensure_buffers(g_backend, width, height)) return 0;
  return 1;
}

extern "C" int cuda_get_last_error(char* out_msg, int out_size) {
  if (!out_msg || out_size <= 0) return 0;
  std::lock_guard<SpinLock> lock(g_last_error_mutex);
  std::snprintf(out_msg, static_cast<size_t>(out_size), "%s", g_last_error.c_str());
  out_msg[out_size - 1] = '\0';
  return static_cast<int>(g_last_error.size());
}

extern "C" int cuda_process(const TBlurDispatch* d, float* out_rgba) {
  std::lock_guard<SpinLock> lock(g_backend.mutex);

  if (!d || !out_rgba) return 0;
  const int width = d->width;
  const int height = d->height;
  if (width <= 0 || height <= 0) return 0;
  if (!d->source.packed_data && !d->source.planar_data) return 0;
  if (d->iterations <= 0 || d->radius_x <= 0 || d->radius_y <= 0) return 0;
  if (d->sample_step_x < 1 || d->sample_step_y < 1) return 0;
  if (!std::isfinite(d->inv2_sig_t_x) || !std::isfinite(d->inv2_sig_t_y) ||
      !std::isfinite(d->k) || !std::isfinite(d->edge_weight) ||
      !std::isfinite(d->edge_despeckle_mix) || !std::isfinite(d->hard_stop_mix) ||
      !std::isfinite(d->edge_gate) || !std::isfinite(d->edge_norm_scale) ||
      !std::isfinite(d->mix) || !std::isfinite(d->organic_cleanup_strength) ||
      !std::isfinite(d->guide_mix) ) return 0;

  if (!init_backend(g_backend)) return 0;
  if (!activate_device(g_backend)) return 0;
  if (!ensure_buffers(g_backend, width, height)) return 0;

  CudaBackend& b = g_backend;

  const int sample_step_x = max(1, min(d->sample_step_x, 8));
  const int sample_step_y = max(1, min(d->sample_step_y, 8));
  const int radius_x = max(1, min(d->radius_x, 1024));
  const int radius_y = max(1, min(d->radius_y, 1024));
  const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
  const int pixel_count = width * height;
  const int scalar_block = 256;
  const int scalar_grid = (pixel_count + scalar_block - 1) / scalar_block;

  const float blur_proxy = static_cast<float>(
      max(radius_x * sample_step_x, radius_y * sample_step_y));
  const float blur_log2 = log2f(1.0f + fmaxf(0.0f, blur_proxy));
  const float blur_edge_t =
      fminf(1.0f, fmaxf(0.0f, (blur_log2 - 1.5f) / 10.0f));

  // --- Source upload (always: ghost-frame-free) -------------------------
  if (!upload_source(b, d->source, width, height)) return 0;

  // --- Guide-luma assembly ---------------------------------------------
  // Three modes:
  //   A) no guide / mix==0  → derive luma from source
  //   B) guide is 1ch luma at full mix → upload directly into d_guide_luma
  //   C) otherwise → upload into d_guide_input + run guide_luma_mix_kernel
  const bool guide_present = d->guide_present != 0;
  const int guide_components = max(1, min(d->guide_components, 3));
  const float guide_mix = d->guide_mix;
  const bool guide_mode_rgb = d->guide_mode_rgb != 0;

  if (guide_present && guide_components == 1 && !guide_mode_rgb &&
      guide_mix >= 0.999f) {
    if (!upload_guide_luma_direct(b, d->guide, width, height)) return 0;
  } else if (guide_present && guide_mix > 1e-6f) {
    if (!upload_guide_input(b, d->guide, width, height, guide_components)) return 0;
    guide_luma_mix_kernel<<<scalar_grid, scalar_block, 0, b.stream>>>(
        b.d_source_rgba, b.d_guide_input, b.d_guide_luma,
        pixel_count, guide_components, guide_mode_rgb ? 1 : 0, guide_mix);
    CUDA_CHECK_KERNEL();
  } else {
    source_luma_kernel<<<scalar_grid, scalar_block, 0, b.stream>>>(
        b.d_source_rgba, b.d_guide_luma, pixel_count);
    CUDA_CHECK_KERNEL();
  }

  // --- Mask upload ------------------------------------------------------
  // mask_components==1 with chan_a==0 and chan_b<0 and !invert → upload directly
  // into d_mask. Otherwise upload into d_mask_input and run mask_extract_kernel.
  const bool mask_present = d->mask_present != 0;
  const int mask_components = max(1, min(d->mask_components, 2));
  const bool mask_direct =
      mask_present && mask_components == 1 &&
      d->mask.chan_a == 0 && d->mask.chan_b < 0 && d->invert_mask == 0;

  if (mask_direct) {
    if (!upload_mask(b, d->mask, width, height, 1, /*direct_into_mask=*/true)) return 0;
  } else if (mask_present) {
    if (!upload_mask(b, d->mask, width, height, mask_components,
                     /*direct_into_mask=*/false)) return 0;
    mask_extract_kernel<<<scalar_grid, scalar_block, 0, b.stream>>>(
        b.d_mask_input, b.d_mask, pixel_count, mask_components,
        d->mask.chan_a, d->mask.chan_b, d->invert_mask);
    CUDA_CHECK_KERNEL();
  }
  // --- Sanitize source NaN/Inf in place (replaces host-side sanitize) --
  sanitize_rgba_kernel<<<scalar_grid, scalar_block, 0, b.stream>>>(
      b.d_source_rgba, pixel_count);
  CUDA_CHECK_KERNEL();

  // --- Edge map --------------------------------------------------------
  {
    const int edge_block = 256;
    const int edge_grid = (pixel_count + edge_block - 1) / edge_block;
    guide_luma_edge_kernel<<<edge_grid, edge_block, 0, b.stream>>>(
        b.d_guide_luma, b.d_edge_temp_a, pixel_count);
    CUDA_CHECK_KERNEL();

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    size_t smem = static_cast<size_t>(18) * 18 * 4 * sizeof(float);
    edge_map_kernel<<<grid, block, smem, b.stream>>>(
        b.d_source_rgba, b.d_edge_temp_a, b.d_guide_edge,
        width, height, d->hard_stop_mix, d->edge_gate, d->edge_norm_scale,
        blur_edge_t);
    CUDA_CHECK_KERNEL();
  }

  // --- Despeckle (3 passes) -------------------------------------------
  if (d->edge_despeckle_mix > 1e-4f) {
    dim3 block(32, 8);
    dim3 grid((width + 31) / 32, (height + 7) / 8);

    despeckle_h_kernel<<<grid, block, 0, b.stream>>>(
        b.d_guide_edge, b.d_edge_temp_a, b.d_edge_temp_b, width, height);
    CUDA_CHECK_KERNEL();

    // d_filter_b is W*H*4 floats; we only write W*H scalar neigh values here.
    // Safe: the filter stage will fully overwrite d_filter_b before reading.
    float* d_neigh = b.d_filter_b;
    despeckle_v_kernel<<<grid, block, 0, b.stream>>>(
        b.d_edge_temp_a, b.d_edge_temp_b, b.d_guide_edge, d_neigh, width, height);
    CUDA_CHECK_KERNEL();

    dim3 block2(16, 16);
    dim3 grid2((width + 15) / 16, (height + 15) / 16);
    despeckle_blend_kernel<<<grid2, block2, 0, b.stream>>>(
        b.d_guide_edge, d_neigh, d->edge_despeckle_mix, width, height);
    CUDA_CHECK_KERNEL();
  }

  // --- show_guide_edge: bypass filtering, output edge preview ---------
  if (d->show_guide_edge != 0) {
    dim3 preview_block(16, 16);
    dim3 preview_grid((width + 15) / 16, (height + 15) / 16);
    guide_edge_preview_kernel<<<preview_grid, preview_block, 0, b.stream>>>(
        b.d_guide_edge, b.d_source_rgba, b.d_filter_a,
        width, height, d->keep_alpha);
    CUDA_CHECK_KERNEL();

    if (!ensure_pinned(b.h_output_pinned, pixels * 4)) return 0;
    CUDA_CHECK(cudaMemcpyAsync(b.h_output_pinned.ptr, b.d_filter_a,
                               pixels * 4 * sizeof(float),
                               cudaMemcpyDeviceToHost, b.stream));
    CUDA_CHECK(cudaStreamSynchronize(b.stream));
    std::memcpy(out_rgba, b.h_output_pinned.ptr, pixels * 4 * sizeof(float));
    return 1;
  }

  // --- Domain-transform filter (iterations × 2 passes) ----------------
  float* d_filtered = b.d_source_rgba;
  {
    dim3 block(32, 8);
    dim3 grid((width + 31) / 32, (height + 7) / 8);
    const float sigma_x_base = sqrtf(0.5f / fmaxf(d->inv2_sig_t_x, 1e-6f));
    const float sigma_y_base = sqrtf(0.5f / fmaxf(d->inv2_sig_t_y, 1e-6f));

    float* d_ping = b.d_source_rgba;
    float* d_pong = b.d_filter_a;
    for (int iter = 0; iter < d->iterations; iter++) {
      const float sigma_scale = iteration_sigma_scale_host(iter, d->iterations);
      const float sigma_x_iter = fmaxf(0.5f, sigma_x_base * sigma_scale);
      const float sigma_y_iter = fmaxf(0.5f, sigma_y_base * sigma_scale);
      const float inv2_x_iter = 1.0f / (2.0f * sigma_x_iter * sigma_x_iter);
      const float inv2_y_iter = 1.0f / (2.0f * sigma_y_iter * sigma_y_iter);
      const bool hv_order = ((iter & 1) == 0);

      if (hv_order) {
        launch_filter_kernel<true>(
            b.stream, grid, block, d_ping, d_pong, b.d_guide_luma,
            b.d_guide_edge, width, height, radius_x, sample_step_x,
            inv2_x_iter, d->k, d->edge_weight, d->keep_alpha);
        CUDA_CHECK_KERNEL();
        launch_filter_kernel<false>(
            b.stream, grid, block, d_pong, b.d_filter_b, b.d_guide_luma,
            b.d_guide_edge, width, height, radius_y, sample_step_y,
            inv2_y_iter, d->k, d->edge_weight, d->keep_alpha);
        CUDA_CHECK_KERNEL();
      } else {
        launch_filter_kernel<false>(
            b.stream, grid, block, d_ping, d_pong, b.d_guide_luma,
            b.d_guide_edge, width, height, radius_y, sample_step_y,
            inv2_y_iter, d->k, d->edge_weight, d->keep_alpha);
        CUDA_CHECK_KERNEL();
        launch_filter_kernel<true>(
            b.stream, grid, block, d_pong, b.d_filter_b, b.d_guide_luma,
            b.d_guide_edge, width, height, radius_x, sample_step_x,
            inv2_x_iter, d->k, d->edge_weight, d->keep_alpha);
        CUDA_CHECK_KERNEL();
      }
      d_ping = b.d_filter_b;
      d_pong = b.d_filter_a;
    }
    d_filtered = d_ping;
  }

  // --- Blend with source through mask × mix ---------------------------
  const bool need_blend = mask_present || (d->mix < 0.9999f);
  float* d_output = d_filtered;
  if (need_blend) {
    d_output = b.d_filter_a;
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    blend_kernel<<<grid, block, 0, b.stream>>>(
        b.d_source_rgba, d_filtered, mask_present ? b.d_mask : nullptr,
        d_output, d->mix, width, height, d->keep_alpha);
    CUDA_CHECK_KERNEL();
  }

  // --- Optional 3×3 organic cleanup ------------------------------------
  const float cleanup_blend =
      fminf(1.0f, fmaxf(0.0f, d->organic_cleanup_strength));
  if (cleanup_blend > 1e-4f) {
    const int cleanup_channels = (d->keep_alpha != 0) ? 3 : 4;
    float* d_cleanup_output =
        (d_output == b.d_filter_a) ? b.d_filter_b : b.d_filter_a;
    dim3 cleanup_block(16, 16);
    dim3 cleanup_grid((width + 15) / 16, (height + 15) / 16);
    organic_cleanup_kernel<<<cleanup_grid, cleanup_block, 0, b.stream>>>(
        d_output, d_cleanup_output, width, height, cleanup_channels, cleanup_blend);
    CUDA_CHECK_KERNEL();
    d_output = d_cleanup_output;
  }
  // --- D2H readback through pinned staging ----------------------------
  if (!ensure_pinned(b.h_output_pinned, pixels * 4)) return 0;
  CUDA_CHECK(cudaMemcpyAsync(b.h_output_pinned.ptr, d_output,
                             pixels * 4 * sizeof(float),
                             cudaMemcpyDeviceToHost, b.stream));
  CUDA_CHECK(cudaStreamSynchronize(b.stream));
  std::memcpy(out_rgba, b.h_output_pinned.ptr, pixels * 4 * sizeof(float));
  return 1;
}

extern "C" void cuda_cleanup(void) {
  std::lock_guard<SpinLock> lock(g_backend.mutex);
  if (g_backend.device_id >= 0) {
    (void)cudaSetDevice(g_backend.device_id);
  }
  free_buffers(g_backend);
  if (g_backend.stream) {
    cudaStreamDestroy(g_backend.stream);
    g_backend.stream = nullptr;
  }
  g_backend.initialized = false;
  g_backend.device_id = -1;
}
