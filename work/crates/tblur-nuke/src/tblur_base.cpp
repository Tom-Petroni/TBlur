static const char* const CLASS = "TBlur";
static const char* const HELP =
    "TBlur (CUDA only): production-ready edge-aware blur.";

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#if defined(_WIN32)
#include <excpt.h>
#define TBLUR_CUDA_SEH_FILTER \
  ((GetExceptionCode() == EXCEPTION_ACCESS_VIOLATION) ? EXCEPTION_EXECUTE_HANDLER : EXCEPTION_CONTINUE_SEARCH)
#endif

#include "DDImage/Format.h"
#include "DDImage/ImagePlane.h"
#include "DDImage/Iop.h"
#include "DDImage/Knob.h"
#include "DDImage/Knobs.h"
#include "DDImage/Row.h"
#include "DDImage/Thread.h"

using namespace DD::Image;

#ifdef TBLUR_CUDA
#include "tblur_cuda.h"
#endif

namespace {

constexpr int kMaxFilterRadius = 768;

inline void append_runtime_log(const char* message) {
  std::fprintf(stderr, "%s\n", message ? message : "TBlur log");
  const char* home = std::getenv("USERPROFILE");
  if (home && *home) {
    std::string log_path(home);
    log_path += "\\.nuke\\TBlur\\backend_runtime.log";
    if (FILE* fp = std::fopen(log_path.c_str(), "a")) {
      std::fprintf(fp, "%s\n", message ? message : "TBlur log");
      std::fclose(fp);
    }
  }
}

#ifdef TBLUR_CUDA
inline int safe_cuda_prepare(int width, int height) {
#if defined(_WIN32)
  __try {
    return cuda_prepare(width, height);
  } __except (TBLUR_CUDA_SEH_FILTER) {
    append_runtime_log("TBlur CUDA exception in cuda_prepare()");
    return 0;
  }
#else
  return cuda_prepare(width, height);
#endif
}

inline int safe_cuda_is_available() {
#if defined(_WIN32)
  __try {
    return cuda_is_available();
  } __except (TBLUR_CUDA_SEH_FILTER) {
    append_runtime_log("TBlur CUDA exception in cuda_is_available()");
    return 0;
  }
#else
  return cuda_is_available();
#endif
}

inline int cached_cuda_is_available() {
  // Cache only successful detection. If detection fails once (driver race,
  // startup timing, transient backend init failure), keep retrying instead of
  // latching a permanent disabled state for the whole Nuke session.
  static std::atomic<int> has_success{0};
  static std::atomic<int> logged_unavailable{0};

  if (has_success.load(std::memory_order_acquire) == 1) {
    return 1;
  }

  const int detected = (safe_cuda_is_available() == 1) ? 1 : 0;
  if (detected == 1) {
    has_success.store(1, std::memory_order_release);
    return 1;
  }

  int expected = 0;
  if (logged_unavailable.compare_exchange_strong(
          expected, 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
    append_runtime_log("TBlur CUDA availability check returned unavailable (will retry).");
  }
  return 0;
}

inline void warm_cuda_runtime_once() {
  static std::once_flag warm_once;
  std::call_once(warm_once, []() {
    if (cached_cuda_is_available() != 1) {
      return;
    }
    (void)safe_cuda_prepare(16, 16);
  });
}

inline int safe_cuda_process(const TBlurDispatch* dispatch, float* out_rgba) {
#if defined(_WIN32)
  __try {
    return cuda_process(dispatch, out_rgba);
  } __except (TBLUR_CUDA_SEH_FILTER) {
    append_runtime_log("TBlur CUDA exception in cuda_process()");
    return 0;
  }
#else
  return cuda_process(dispatch, out_rgba);
#endif
}

inline int safe_cuda_get_last_error_raw(char* out_msg, int out_size) {
#if defined(_WIN32)
  __try {
    return cuda_get_last_error(out_msg, out_size);
  } __except (TBLUR_CUDA_SEH_FILTER) {
    if (out_msg && out_size > 0) {
      out_msg[0] = '\0';
    }
    return 0;
  }
#else
  return cuda_get_last_error(out_msg, out_size);
#endif
}

inline std::string fetch_cuda_last_error() {
  char buf[2048];
  buf[0] = '\0';
  (void)safe_cuda_get_last_error_raw(buf, static_cast<int>(sizeof(buf)));
  return std::string(buf);
}
#endif

inline bool is_finitef(float v) {
  return std::isfinite(static_cast<double>(v));
}

inline float finite_or(float v, float fallback) {
  return is_finitef(v) ? v : fallback;
}

inline float clamp01(float v) {
  if (!is_finitef(v)) {
    return 0.0f;
  }
  return std::max(0.0f, std::min(1.0f, v));
}

inline int clamp_int(int v, int lo, int hi) {
  return std::max(lo, std::min(hi, v));
}

inline double clamp_double(double v, double lo, double hi) {
  return std::max(lo, std::min(hi, v));
}

inline size_t pixel_offset(int x, int y, int width) {
  return static_cast<size_t>((y * width + x) * 4);
}

inline void sanitize_rgba_buffer_in_place(std::vector<float>& rgba) {
  if (rgba.empty()) {
    return;
  }
  const size_t pixel_count = rgba.size() / 4;
  for (size_t i = 0; i < pixel_count; ++i) {
    const size_t p = i * 4;
    rgba[p + 0] = finite_or(rgba[p + 0], 0.0f);
    rgba[p + 1] = finite_or(rgba[p + 1], 0.0f);
    rgba[p + 2] = finite_or(rgba[p + 2], 0.0f);
    rgba[p + 3] = finite_or(rgba[p + 3], 1.0f);
  }
}

inline float map_edge_threshold(double threshold_ui) {
  const float t = std::max(0.0f, static_cast<float>(threshold_ui));
  constexpr float kMinSigmaR = 0.02f;
  constexpr float kMaxSigmaR = 0.65f;
  if (t <= 1.0f) {
    const float curve = std::pow(t, 1.35f);
    return std::max(1e-5f, kMinSigmaR * std::pow(kMaxSigmaR / kMinSigmaR, curve));
  }
  const float extra = t - 1.0f;
  const float sigma_ext = kMaxSigmaR * std::exp2(1.25f * extra);
  return std::max(1e-5f, std::min(sigma_ext, 8.0f));
}

inline float map_blur_size_ui(double blur_ui) {
  const float u = std::max(0.0f, static_cast<float>(blur_ui));
  if (u <= 0.0f) {
    return 0.0f;
  }
  constexpr float kExpoRange = 15.0f;
  if (u <= 100.0f) {
    const float t = u / 100.0f;
    return std::exp2(t * kExpoRange) - 1.0f;
  }
  const float base_at_100 = std::exp2(kExpoRange) - 1.0f;
  const float extra = (u - 100.0f) / 100.0f;
  const float grown = base_at_100 * (1.0f + 10.0f * std::pow(extra, 1.22f));
  return std::min(grown, 1.0e9f);
}

inline float compute_organic_cleanup_strength(
    float blur_amount,
    float edge_threshold,
    float edge_smooth,
    int iterations) {
  const float blur_log2 = std::log2(1.0f + std::max(0.0f, blur_amount));
  const float blur_term = clamp01((blur_log2 - 7.5f) / 4.5f);
  const float edge_term = clamp01((0.35f - clamp01(edge_threshold)) / 0.35f);
  const float iter_term = clamp01((static_cast<float>(iterations) - 1.0f) / 9.0f);
  const float smooth_damp = 1.0f - 0.35f * clamp01(edge_smooth);
  return clamp01(0.28f * blur_term * edge_term * (0.55f + 0.45f * iter_term) * smooth_damp);
}

// Initialize a TBlurInputDesc to "no input". chan_a..chan_d default to -1.
inline void clear_input_desc(TBlurInputDesc* d) {
  d->packed_data = nullptr;
  d->planar_data = nullptr;
  d->plane_size_floats = 0;
  d->row_stride = 0;
  d->col_stride = 0;
  d->chan_a = -1;
  d->chan_b = -1;
  d->chan_c = -1;
  d->chan_d = -1;
}

// Fill desc from an RGBA ImagePlane.
//   - Fast path: plane is already RGBA-packed → desc.packed_data = plane.readable()
//   - Slow path: any other layout → desc.planar_data + strides + chan indices
//                (GPU runs deinterleave_rgba_kernel, no host scatter)
// Returns false only if the plane is empty/invalid.
inline bool fill_source_desc(
    const ImagePlane& plane, int width, int height, TBlurInputDesc* desc) {
  clear_input_desc(desc);
  const float* data = plane.readable();
  if (!data || width <= 0 || height <= 0) return false;

  const int row_stride = plane.rowStride();
  const int col_stride = plane.colStride();
  const int r_chan = plane.chanNo(Chan_Red);
  const int g_chan = plane.chanNo(Chan_Green);
  const int b_chan = plane.chanNo(Chan_Blue);
  const int a_chan = plane.chanNo(Chan_Alpha);

  const bool packed_rgba =
      plane.packed() && col_stride == 4 && row_stride == width * 4 &&
      r_chan == 0 && g_chan == 1 && b_chan == 2 && a_chan == 3;
  if (packed_rgba) {
    desc->packed_data = data;
    return true;
  }

  desc->planar_data = data;
  desc->plane_size_floats = row_stride * height;
  desc->row_stride = row_stride;
  desc->col_stride = col_stride;
  desc->chan_a = r_chan;
  desc->chan_b = g_chan;
  desc->chan_c = b_chan;
  desc->chan_d = a_chan;
  return true;
}

// Fill desc for a single-channel guide-luma or mask plane.
inline bool fill_scalar_desc(
    const ImagePlane& plane, int width, int height,
    int preferred_channel, TBlurInputDesc* desc) {
  clear_input_desc(desc);
  const float* data = plane.readable();
  if (!data || width <= 0 || height <= 0) return false;

  const int row_stride = plane.rowStride();
  const int col_stride = plane.colStride();
  const int comps = plane.nComps();
  const int chan_red = plane.chanNo(Chan_Red);
  const int chan_alpha = plane.chanNo(Chan_Alpha);
  int chan = preferred_channel;
  if (chan < 0 || chan >= comps) {
    chan = (chan_red >= 0 && chan_red < comps) ? chan_red : -1;
  }
  if (chan < 0 || chan >= comps) {
    chan = (chan_alpha >= 0 && chan_alpha < comps) ? chan_alpha : -1;
  }
  if (chan < 0 || chan >= comps) chan = 0;

  const bool direct_single =
      plane.packed() && comps == 1 && col_stride == 1 && row_stride == width;
  if (direct_single) {
    desc->packed_data = data;
    desc->chan_a = 0;
    return true;
  }

  desc->planar_data = data;
  desc->plane_size_floats = row_stride * height;
  desc->row_stride = row_stride;
  desc->col_stride = col_stride;
  desc->chan_a = chan;
  return true;
}

// Fill desc for a 3-component guide-RGB plane.
inline bool fill_rgb_desc(
    const ImagePlane& plane, int width, int height, TBlurInputDesc* desc) {
  clear_input_desc(desc);
  const float* data = plane.readable();
  if (!data || width <= 0 || height <= 0) return false;

  const int row_stride = plane.rowStride();
  const int col_stride = plane.colStride();
  const int comps = plane.nComps();
  const int chan_r = plane.chanNo(Chan_Red);
  const int chan_g = plane.chanNo(Chan_Green);
  const int chan_b = plane.chanNo(Chan_Blue);

  const bool direct_rgb =
      plane.packed() && comps == 3 && col_stride == 3 &&
      row_stride == width * 3 &&
      chan_r == 0 && chan_g == 1 && chan_b == 2;
  if (direct_rgb) {
    desc->packed_data = data;
    desc->chan_a = 0;
    desc->chan_b = 1;
    desc->chan_c = 2;
    return true;
  }

  desc->planar_data = data;
  desc->plane_size_floats = row_stride * height;
  desc->row_stride = row_stride;
  desc->col_stride = col_stride;
  desc->chan_a = chan_r;
  desc->chan_b = chan_g;
  desc->chan_c = chan_b;
  return true;
}

// Fill desc for a mask plane that may have both Red and Alpha channels.
// out_components / out_chan_red / out_chan_alpha report what the kernel should
// pick from. When the mask is the "direct" 1ch case (alpha or red, no invert),
// the caller can pass it unchanged; otherwise mask_extract_kernel will pick.
inline bool fill_mask_desc(
    const ImagePlane& plane, int width, int height, TBlurInputDesc* desc,
    int* out_components, int* out_chan_red, int* out_chan_alpha) {
  clear_input_desc(desc);
  const float* data = plane.readable();
  if (!data || width <= 0 || height <= 0) return false;

  const int row_stride = plane.rowStride();
  const int col_stride = plane.colStride();
  const int comps = plane.nComps();
  const int chan_red = plane.chanNo(Chan_Red);
  const int chan_alpha = plane.chanNo(Chan_Alpha);

  *out_components = (comps >= 2) ? 2 : 1;
  *out_chan_red = (chan_red >= 0 && chan_red < comps) ? chan_red : -1;
  *out_chan_alpha = (chan_alpha >= 0 && chan_alpha < comps) ? chan_alpha : -1;

  const bool direct_single =
      plane.packed() && comps == 1 && col_stride == 1 && row_stride == width;
  if (direct_single) {
    desc->packed_data = data;
    desc->chan_a = 0;
    desc->chan_b = -1;
    return true;
  }

  desc->planar_data = data;
  desc->plane_size_floats = row_stride * height;
  desc->row_stride = row_stride;
  desc->col_stride = col_stride;
  desc->chan_a = *out_chan_red;
  desc->chan_b = *out_chan_alpha;
  return true;
}

inline void assign_black_opaque_rgba(
    int width, int height, std::vector<float>& out_rgba) {
  const size_t pixel_count =
      static_cast<size_t>(std::max(0, width)) *
      static_cast<size_t>(std::max(0, height));
  out_rgba.assign(pixel_count * 4, 0.0f);
  for (size_t i = 0; i < pixel_count; ++i) {
    out_rgba[i * 4 + 3] = 1.0f;
  }
}

inline void copy_direct_rgba_to_vector(
    const float* source_ptr, int width, int height, std::vector<float>& out_rgba) {
  const size_t pixel_count =
      static_cast<size_t>(std::max(0, width)) *
      static_cast<size_t>(std::max(0, height));
  out_rgba.assign(pixel_count * 4, 0.0f);
  if (!source_ptr || pixel_count == 0) {
    return;
  }
  std::memcpy(
      out_rgba.data(),
      source_ptr,
      pixel_count * 4 * sizeof(float));
}

inline void copy_source_desc_to_rgba(
    const TBlurInputDesc& desc, int width, int height, std::vector<float>& out_rgba) {
  const size_t pixel_count =
      static_cast<size_t>(std::max(0, width)) *
      static_cast<size_t>(std::max(0, height));
  out_rgba.assign(pixel_count * 4, 0.0f);
  if (pixel_count == 0) {
    return;
  }

  if (desc.packed_data) {
    copy_direct_rgba_to_vector(desc.packed_data, width, height, out_rgba);
    return;
  }

  if (!desc.planar_data || desc.row_stride <= 0 || desc.col_stride <= 0) {
    assign_black_opaque_rgba(width, height, out_rgba);
    return;
  }

  const int plane_size = desc.plane_size_floats;
  for (int y = 0; y < height; ++y) {
    const int row_base = y * desc.row_stride;
    for (int x = 0; x < width; ++x) {
      const int src_base = row_base + x * desc.col_stride;
      const size_t dst = (static_cast<size_t>(y) * static_cast<size_t>(width) +
                          static_cast<size_t>(x)) * 4;

      auto fetch = [&](int chan, float fallback) -> float {
        if (chan < 0) {
          return fallback;
        }
        const int idx = src_base + chan;
        if (idx < 0 || idx >= plane_size) {
          return fallback;
        }
        return finite_or(desc.planar_data[idx], fallback);
      };

      out_rgba[dst + 0] = fetch(desc.chan_a, 0.0f);
      out_rgba[dst + 1] = fetch(desc.chan_b, 0.0f);
      out_rgba[dst + 2] = fetch(desc.chan_c, 0.0f);
      out_rgba[dst + 3] = fetch(desc.chan_d, 1.0f);
    }
  }
}

inline bool extract_rgba_from_plane(
    const ImagePlane& plane,
    int width,
    int height,
    std::vector<float>& out_rgba,
    bool* out_is_direct,
    const float** out_direct_ptr) {
  if (out_is_direct) {
    *out_is_direct = false;
  }
  if (out_direct_ptr) {
    *out_direct_ptr = nullptr;
  }

  const float* data = plane.readable();
  const int row_stride = plane.rowStride();
  const int col_stride = plane.colStride();
  const int r_chan = plane.chanNo(Chan_Red);
  const int g_chan = plane.chanNo(Chan_Green);
  const int b_chan = plane.chanNo(Chan_Blue);
  const int a_chan = plane.chanNo(Chan_Alpha);

  const bool packed_rgba =
      data && plane.packed() && col_stride == 4 && row_stride == width * 4 &&
      r_chan == 0 && g_chan == 1 && b_chan == 2 && a_chan == 3;
  if (packed_rgba) {
    if (out_is_direct) {
      *out_is_direct = true;
    }
    if (out_direct_ptr) {
      *out_direct_ptr = data;
    }
    out_rgba.clear();
    return true;
  }

  const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
  out_rgba.assign(pixel_count * 4, 0.0f);
  if (!data || width <= 0 || height <= 0) {
    for (size_t i = 0; i < pixel_count; ++i) {
      out_rgba[i * 4 + 3] = 1.0f;
    }
    return false;
  }

  for (int y = 0; y < height; ++y) {
    const float* row_ptr = data + static_cast<size_t>(y) * static_cast<size_t>(row_stride);
    for (int x = 0; x < width; ++x) {
      const float* px = row_ptr + static_cast<size_t>(x) * static_cast<size_t>(col_stride);
      const size_t p = pixel_offset(x, y, width);
      out_rgba[p + 0] = finite_or((r_chan >= 0) ? px[r_chan] : 0.0f, 0.0f);
      out_rgba[p + 1] = finite_or((g_chan >= 0) ? px[g_chan] : 0.0f, 0.0f);
      out_rgba[p + 2] = finite_or((b_chan >= 0) ? px[b_chan] : 0.0f, 0.0f);
      out_rgba[p + 3] = finite_or((a_chan >= 0) ? px[a_chan] : 1.0f, 1.0f);
    }
  }

  return true;
}

inline bool extract_single_channel_from_plane(
    const ImagePlane& plane,
    int width,
    int height,
    int preferred_channel,
    std::vector<float>& out_values,
    bool* out_is_direct,
    const float** out_direct_ptr) {
  if (out_is_direct) {
    *out_is_direct = false;
  }
  if (out_direct_ptr) {
    *out_direct_ptr = nullptr;
  }

  const float* data = plane.readable();
  const int row_stride = plane.rowStride();
  const int col_stride = plane.colStride();
  const int comps = plane.nComps();
  const int chan_red = plane.chanNo(Chan_Red);
  const int chan_alpha = plane.chanNo(Chan_Alpha);
  int chan = preferred_channel;
  if (chan < 0 || chan >= comps) {
    chan = (chan_red >= 0 && chan_red < comps) ? chan_red : -1;
  }
  if (chan < 0 || chan >= comps) {
    chan = (chan_alpha >= 0 && chan_alpha < comps) ? chan_alpha : -1;
  }
  if (chan < 0 || chan >= comps) {
    chan = 0;
  }

  const bool direct_one_channel =
      data && plane.packed() && comps == 1 && col_stride == 1 && row_stride == width;
  if (direct_one_channel) {
    if (out_is_direct) {
      *out_is_direct = true;
    }
    if (out_direct_ptr) {
      *out_direct_ptr = data;
    }
    out_values.clear();
    return true;
  }

  const size_t count = static_cast<size_t>(width) * static_cast<size_t>(height);
  out_values.assign(count, 0.0f);
  if (!data || width <= 0 || height <= 0) {
    return false;
  }

  for (int y = 0; y < height; ++y) {
    const float* row_ptr = data + static_cast<size_t>(y) * static_cast<size_t>(row_stride);
    for (int x = 0; x < width; ++x) {
      const float* px = row_ptr + static_cast<size_t>(x) * static_cast<size_t>(col_stride);
      out_values[static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)] =
          finite_or(px[chan], 0.0f);
    }
  }
  return true;
}

inline bool extract_rgb_from_plane(
    const ImagePlane& plane,
    int width,
    int height,
    std::vector<float>& out_values,
    bool* out_is_direct,
    const float** out_direct_ptr) {
  if (out_is_direct) {
    *out_is_direct = false;
  }
  if (out_direct_ptr) {
    *out_direct_ptr = nullptr;
  }

  const float* data = plane.readable();
  const int row_stride = plane.rowStride();
  const int col_stride = plane.colStride();
  const int comps = plane.nComps();
  const int chan_r = plane.chanNo(Chan_Red);
  const int chan_g = plane.chanNo(Chan_Green);
  const int chan_b = plane.chanNo(Chan_Blue);
  const bool has_rgb =
      (chan_r >= 0 && chan_r < comps) &&
      (chan_g >= 0 && chan_g < comps) &&
      (chan_b >= 0 && chan_b < comps);

  const bool direct_rgb =
      data &&
      plane.packed() &&
      comps == 3 &&
      col_stride == comps &&
      row_stride == width * comps &&
      chan_r == 0 && chan_g == 1 && chan_b == 2;
  if (direct_rgb) {
    if (out_is_direct) {
      *out_is_direct = true;
    }
    if (out_direct_ptr) {
      *out_direct_ptr = data;
    }
    out_values.clear();
    return true;
  }

  const size_t count = static_cast<size_t>(width) * static_cast<size_t>(height);
  out_values.assign(count * 3, 0.0f);
  if (!data || width <= 0 || height <= 0) {
    return false;
  }

  for (int y = 0; y < height; ++y) {
    const float* row_ptr = data + static_cast<size_t>(y) * static_cast<size_t>(row_stride);
    for (int x = 0; x < width; ++x) {
      const float* px = row_ptr + static_cast<size_t>(x) * static_cast<size_t>(col_stride);
      const size_t p =
          (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 3;
      if (has_rgb) {
        out_values[p + 0] = finite_or(px[chan_r], 0.0f);
        out_values[p + 1] = finite_or(px[chan_g], 0.0f);
        out_values[p + 2] = finite_or(px[chan_b], 0.0f);
      } else {
        const float v = finite_or(px[0], 0.0f);
        out_values[p + 0] = v;
        out_values[p + 1] = v;
        out_values[p + 2] = v;
      }
    }
  }
  return true;
}

// Host-side scalar extract used as a safe fallback when an AOV CUDA dispatch
// fails. Keeping this path avoids dropping channels on backend failures.
inline void extract_channel_to_vector(
    const ImagePlane& plane, int width, int height, int preferred_channel,
    std::vector<float>& out_values) {
  const size_t count = static_cast<size_t>(width) * static_cast<size_t>(height);
  out_values.assign(count, 0.0f);
  const float* data = plane.readable();
  if (!data || width <= 0 || height <= 0) return;

  const int row_stride = plane.rowStride();
  const int col_stride = plane.colStride();
  const int comps = plane.nComps();
  int chan = preferred_channel;
  if (chan < 0 || chan >= comps) chan = 0;

  for (int y = 0; y < height; ++y) {
    const float* row_ptr =
        data + static_cast<size_t>(y) * static_cast<size_t>(row_stride);
    for (int x = 0; x < width; ++x) {
      const float* px =
          row_ptr + static_cast<size_t>(x) * static_cast<size_t>(col_stride);
      out_values[static_cast<size_t>(y) * static_cast<size_t>(width) +
                 static_cast<size_t>(x)] = finite_or(px[chan], 0.0f);
    }
  }
}

}  // namespace

enum GuideMode {
  GUIDE_LUMA = 0,
  GUIDE_RGB = 1,
};

enum QualityMode {
  QUALITY_FINAL = 0,
  QUALITY_DRAFT = 1,
};

static const char* const kGuideModes[] = {
    "Luma",
    "RGB",
    nullptr,
};

static const char* const kQualityModes[] = {
    "Final",
    "Draft",
    nullptr,
};

class TBlurBase : public Iop {
  double blur_size_ = 50.0;
  double edge_threshold_ = 0.2;
  double edge_smooth_ = 0.0;
  double guide_influence_ = 0.0;
  int guide_mode_ = GUIDE_LUMA;
  int quality_mode_ = QUALITY_FINAL;
  double edge_boost_ = 1.0;
  double anisotropy_ = 0.0;
  bool show_guide_edge_ = false;
  bool invert_mask_ = false;
  int iterations_ = 8;
  double mix_ = 1.0;
  bool keep_alpha_ = false;
  ChannelSet channels_ = Mask_RGBA;

  Lock cache_lock_;
  bool cache_valid_ = false;
  bool cache_hash_valid_ = false;
  Hash cache_hash_;
  double cache_frame_ = std::numeric_limits<double>::quiet_NaN();
  int cache_view_ = 0;
  int cache_x_ = 0;
  int cache_y_ = 0;
  int cache_w_ = 0;
  int cache_h_ = 0;
  std::shared_ptr<const std::vector<float>> cached_rgba_;
  std::vector<Channel> cached_extra_channels_;
  std::shared_ptr<const std::vector<float>> cached_extra_values_;

 public:
  explicit TBlurBase(Node* node) : Iop(node) {
    inputs(3);
#ifdef TBLUR_CUDA
    warm_cuda_runtime_once();
#endif
  }

  void knobs(Knob_Callback f) override {
    Input_ChannelSet_knob(f, &channels_, 0, "channels", "channels");
    Tooltip(f, "Channels processed by TBlur.");

    BeginGroup(f, "tblur_blur_group", "Blur");

    Knob* blur_size_knob = Double_knob(f, &blur_size_, "blur_size", "Size");
    SetRange(f, 0.0, 500.0);
    Tooltip(f, "Main blur amount.");
    if (blur_size_knob) {
      blur_size_knob->set_flag(Knob::KNOB_CHANGED_ALWAYS);
    }
    Divider(f);

    Knob* edge_threshold_knob =
        Double_knob(f, &edge_threshold_, "edge_threshold", "Threshold");
    SetRange(f, 0.0, 1.0);
    Tooltip(f, "Higher values allow more blur through edges.");
    if (edge_threshold_knob) {
      edge_threshold_knob->set_flag(Knob::KNOB_CHANGED_ALWAYS);
    }

    Knob* edge_smooth_knob =
        Double_knob(f, &edge_smooth_, "edge_smooth", "Smoothness");
    SetRange(f, 0.0, 1.0);
    Tooltip(f, "Softens edge transitions.");
    if (edge_smooth_knob) {
      edge_smooth_knob->set_flag(Knob::KNOB_CHANGED_ALWAYS);
    }
    Divider(f);

    Knob* anisotropy_knob =
        Double_knob(f, &anisotropy_, "anisotropy", "Anisotropy");
    SetRange(f, -1.0, 1.0);
    Tooltip(f, "Negative favors vertical blur, positive favors horizontal blur.");
    if (anisotropy_knob) {
      anisotropy_knob->set_flag(Knob::KNOB_CHANGED_ALWAYS);
    }
    Divider(f);

    EndGroup(f);

    BeginGroup(f, "tblur_guide_group", "Guide");

    Knob* guide_influence_knob =
        Double_knob(f, &guide_influence_, "guide_influence", "Influence");
    SetRange(f, 0.0, 1.0);
    Tooltip(f, "How much the Guide input drives edge detection.");
    if (guide_influence_knob) {
      guide_influence_knob->set_flag(Knob::KNOB_CHANGED_ALWAYS);
    }

    Enumeration_knob(f, &guide_mode_, kGuideModes, "guide_mode", "Mode");
    Tooltip(f, "Guide interpretation mode.");

    Bool_knob(f, &show_guide_edge_, "show_guide_edge", "Show Edge Map");
    Tooltip(f, "Display guide edge map.");
    Divider(f);

    EndGroup(f);

    BeginGroup(f, "tblur_advanced_group", "Advanced");

    Knob* iterations_knob = Int_knob(f, &iterations_, "iterations", "Iterations");
    SetRange(f, 1.0, 16.0);
    Tooltip(f, "More iterations can improve quality at higher cost.");
    if (iterations_knob) {
      iterations_knob->set_flag(Knob::KNOB_CHANGED_ALWAYS);
    }
    Divider(f);

    EndGroup(f);

    Knob* mix_knob = Double_knob(f, &mix_, "mix", "Mix");
    SetRange(f, 0.0, 1.0);
    Tooltip(f, "Mix filtered result over source.");
    if (mix_knob) {
      mix_knob->set_flag(Knob::KNOB_CHANGED_ALWAYS);
    }

    Bool_knob(f, &invert_mask_, "invert_mask", "invert");
    Tooltip(f, "Invert Mask input.");

    Newline(f);
    Bool_knob(f, &keep_alpha_, "keep_alpha", "Keep Source Alpha");
    Tooltip(f, "Keep source alpha untouched.");
  }

  int knob_changed(Knob* k) override {
    const int base = Iop::knob_changed(k);
    if (!k) {
      return base;
    }

    const bool affects_render =
        k->is("channels") ||
        k->is("blur_size") ||
        k->is("edge_threshold") ||
        k->is("edge_smooth") ||
        k->is("edge_boost") ||
        k->is("anisotropy") ||
        k->is("guide_influence") ||
        k->is("guide_mode") ||
        k->is("show_guide_edge") ||
        k->is("iterations") ||
        k->is("mix") ||
        k->is("quality") ||
        k->is("keep_alpha") ||
        k->is("invert_mask");
    if (affects_render) {
      invalidateSameHash();
      asapUpdate();
      return 1;
    }

    return base;
  }

  void _validate(bool) override {
    if (!input(0)) {
      info_.channels(Mask_None);
      set_out_channels(Mask_None);
      return;
    }

    copy_info();
    info_.channels(input0().channels());
    set_out_channels(Mask_All);
    info_.black_outside(false);

    blur_size_ = clamp_double(blur_size_, 0.0, 5000.0);
    edge_threshold_ = clamp_double(edge_threshold_, 0.0, 8.0);
    edge_smooth_ = clamp_double(edge_smooth_, 0.0, 1.0);
    edge_boost_ = clamp_double(edge_boost_, 0.0, 2.0);
    anisotropy_ = clamp_double(anisotropy_, -1.0, 1.0);
    guide_influence_ = clamp_double(guide_influence_, 0.0, 1.0);
    guide_mode_ = clamp_int(guide_mode_, GUIDE_LUMA, GUIDE_RGB);
    quality_mode_ = clamp_int(quality_mode_, QUALITY_FINAL, QUALITY_DRAFT);
    iterations_ = clamp_int(iterations_, 1, 16);
    mix_ = clamp_double(mix_, 0.0, 1.0);
    if (!channels_) {
      channels_ = Mask_RGBA;
    }

#ifdef TBLUR_CUDA
    warm_cuda_runtime_once();
#endif
  }

  void _request(int x, int y, int r, int t, ChannelMask channels, int count) override {
    (void)x;
    (void)y;
    (void)r;
    (void)t;
    const Format fmt = format();
    ChannelSet source_request_channels = channels;
    source_request_channels += Mask_RGBA;
    if (!channels_.all()) {
      source_request_channels += channels_;
    }
    input0().request(
        fmt.x(),
        fmt.y(),
        fmt.r(),
        fmt.t(),
        source_request_channels,
        count);
    const float guide_mix = static_cast<float>(clamp_double(guide_influence_, 0.0, 1.0));
    if ((guide_mix > 1e-6f || show_guide_edge_) && has_connected_input(1)) {
      if (Iop* guide = input(1)) {
        const ChannelMask guide_channels =
            (guide_mode_ == GUIDE_RGB) ? Mask_RGB : Mask_Red;
        guide->request(fmt.x(), fmt.y(), fmt.r(), fmt.t(), guide_channels, count);
      }
    }
    if (has_connected_input(2)) {
      if (Iop* mask = input(2)) {
        mask->request(fmt.x(), fmt.y(), fmt.r(), fmt.t(), Mask_Red | Mask_Alpha, count);
      }
    }
  }

  void engine(int y, int x, int r, ChannelMask channels, Row& row) override {
    ensure_cache();

    const Hash req_hash = hash();
    const double req_frame = outputContext().frame();
    const int req_view = outputContext().view();

    std::shared_ptr<const std::vector<float>> cache_snapshot;
    std::shared_ptr<const std::vector<float>> cache_extra_snapshot;
    std::vector<Channel> extra_channels_snapshot;
    int cx = 0;
    int cy = 0;
    int cw = 0;
    int ch = 0;

    {
      Guard guard(cache_lock_);
      if (cache_valid_ &&
          cache_hash_valid_ &&
          cache_hash_ == req_hash &&
          cache_frame_ == req_frame &&
          cache_view_ == req_view &&
          cached_rgba_) {
        cache_snapshot = cached_rgba_;
        cache_extra_snapshot = cached_extra_values_;
        extra_channels_snapshot = cached_extra_channels_;
        cx = cache_x_;
        cy = cache_y_;
        cw = cache_w_;
        ch = cache_h_;
      }
    }

    const bool has_cache =
        cache_snapshot &&
        !cache_snapshot->empty() &&
        (cache_snapshot->size() >= static_cast<size_t>(std::max(0, cw * ch * 4)));
    const bool inside_y = has_cache && (y >= cy) && (y < cy + ch);
    const int ly = y - cy;

    Row source_row(x, r);
    source_row.get(input0(), y, x, r, Mask_RGBA);
    row.get(input0(), y, x, r, channels);
    float* out_r = (channels & Chan_Red) ? row.writable(Chan_Red) : nullptr;
    float* out_g = (channels & Chan_Green) ? row.writable(Chan_Green) : nullptr;
    float* out_b = (channels & Chan_Blue) ? row.writable(Chan_Blue) : nullptr;
    float* out_a = (channels & Chan_Alpha) ? row.writable(Chan_Alpha) : nullptr;
    const float* src_r = source_row[Chan_Red];
    const float* src_g = source_row[Chan_Green];
    const float* src_b = source_row[Chan_Blue];
    const float* src_a = source_row[Chan_Alpha];

    const float* cache_ptr = has_cache ? cache_snapshot->data() : nullptr;
    const bool process_r = (channels_ & Chan_Red);
    const bool process_g = (channels_ & Chan_Green);
    const bool process_b = (channels_ & Chan_Blue);
    const bool process_a = (channels_ & Chan_Alpha);
    for (int px = x; px < r; ++px) {
      float vr = finite_or(src_r ? src_r[px] : 0.0f, 0.0f);
      float vg = finite_or(src_g ? src_g[px] : 0.0f, 0.0f);
      float vb = finite_or(src_b ? src_b[px] : 0.0f, 0.0f);
      float va = finite_or(src_a ? src_a[px] : 1.0f, 1.0f);

      if (cache_ptr && inside_y && px >= cx && px < cx + cw) {
        const size_t idx = pixel_offset(px - cx, ly, cw);
        if (process_r) vr = finite_or(cache_ptr[idx + 0], vr);
        if (process_g) vg = finite_or(cache_ptr[idx + 1], vg);
        if (process_b) vb = finite_or(cache_ptr[idx + 2], vb);
        if (process_a) va = finite_or(cache_ptr[idx + 3], va);
      }

      if (out_r) out_r[px] = vr;
      if (out_g) out_g[px] = vg;
      if (out_b) out_b[px] = vb;
      if (out_a) out_a[px] = va;
    }

    if (cache_extra_snapshot && !extra_channels_snapshot.empty() && inside_y) {
      const float* extra_ptr = cache_extra_snapshot->data();
      const size_t pixel_count = static_cast<size_t>(cw) * static_cast<size_t>(ch);
      for (Channel z : channels) {
        if (z == Chan_Red || z == Chan_Green || z == Chan_Blue || z == Chan_Alpha) {
          continue;
        }
        int channel_index = -1;
        for (size_t i = 0; i < extra_channels_snapshot.size(); ++i) {
          if (extra_channels_snapshot[i] == z) {
            channel_index = static_cast<int>(i);
            break;
          }
        }
        if (channel_index < 0) {
          continue;
        }
        float* out = row.writable(z);
        if (!out) {
          continue;
        }
        const size_t base =
            static_cast<size_t>(channel_index) * pixel_count +
            static_cast<size_t>(ly) * static_cast<size_t>(cw);
        for (int px = x; px < r; ++px) {
          if (px < cx || px >= (cx + cw)) {
            continue;
          }
          out[px] = finite_or(extra_ptr[base + static_cast<size_t>(px - cx)], out[px]);
        }
      }
    }
  }

  void append(Hash& hash) override {
    Iop::append(hash);
    hash.append(0x54424C5232ull); // TBLR2
    hash.append(static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(this)));
    hash.append(node_name());
    hash.append(outputContext().frame());
    hash.append(outputContext().view());

    const Format fmt = format();
    hash.append(fmt.x());
    hash.append(fmt.y());
    hash.append(fmt.r());
    hash.append(fmt.t());

    hash.append(static_cast<float>(blur_size_));
    hash.append(static_cast<float>(edge_threshold_));
    hash.append(static_cast<float>(edge_smooth_));
    hash.append(static_cast<float>(edge_boost_));
    hash.append(static_cast<float>(anisotropy_));
    hash.append(static_cast<float>(guide_influence_));
    hash.append(guide_mode_);
    hash.append(show_guide_edge_ ? 1 : 0);
    hash.append(iterations_);
    hash.append(static_cast<float>(mix_));
    hash.append(quality_mode_);
    hash.append(keep_alpha_ ? 1 : 0);
    hash.append(invert_mask_ ? 1 : 0);
    channels_.append(hash);
    hash.append(has_connected_input(1) ? 1 : 0);
    hash.append(has_connected_input(2) ? 1 : 0);

#if defined(kDDImageVersionMajorNum) && (kDDImageVersionMajorNum >= 14)
    enableVaryingOutputHash();
#endif
  }

  const char* Class() const override { return CLASS; }
  const char* node_help() const override { return HELP; }
  int minimum_inputs() const override { return 3; }
  int maximum_inputs() const override { return 3; }
  int optional_input() const override { return 2; }

  const char* input_label(int input, char*) const override {
    if (input == 0) {
      return "Img";
    }
    if (input == 1) {
      return "Guide";
    }
    if (input == 2) {
      return "mask";
    }
    return nullptr;
  }

 private:
  bool has_connected_input(int index) const {
    const Iop* in = input(index);
    if (!in) {
      return false;
    }
#if defined(kDDImageVersionMajorNum) && (kDDImageVersionMajorNum >= 14)
    return !in->isBlackIop();
#else
    Iop* default_iop = Iop::default_input(outputContext());
    return default_iop ? (in != default_iop) : true;
#endif
  }

  int compute_filter_radius() const {
    const float blur_amount = map_blur_size_ui(blur_size_);
    const int iterations = clamp_int(iterations_, 1, 16);
    if (blur_amount <= 0.0f || iterations <= 0) {
      return 0;
    }
    const float sigma_s_base = std::max(0.5f, std::sqrt(std::max(0.0f, blur_amount)));
    const float sigma_s = std::max(
        0.5f,
        sigma_s_base / std::sqrt(static_cast<float>(iterations)));
    return clamp_int(
        std::min(static_cast<int>(std::ceil(2.5f * sigma_s)), kMaxFilterRadius),
        1,
        kMaxFilterRadius);
  }

  void ensure_cache() {
    Guard guard(cache_lock_);

    const Format fmt = format();
    const int fmt_x = fmt.x();
    const int fmt_y = fmt.y();
    const int fmt_r = fmt.r();
    const int fmt_t = fmt.t();

    // Full-frame cache avoids ROI/tile seams during zoom/pan interactions.
    const int req_x = fmt_x;
    const int req_y = fmt_y;
    const int req_r = fmt_r;
    const int req_t = fmt_t;
    const int req_w = std::max(0, req_r - req_x);
    const int req_h = std::max(0, req_t - req_y);

    const Hash req_hash = hash();
    const double req_frame = outputContext().frame();
    const int req_view = outputContext().view();
    auto clear_cache_state = [&]() {
      cache_valid_ = false;
      cache_hash_valid_ = false;
      cached_rgba_.reset();
      cached_extra_channels_.clear();
      cached_extra_values_.reset();
    };

    if (aborted() || cancelled()) {
      clear_cache_state();
      return;
    }

    if (cache_valid_ &&
        cache_hash_valid_ &&
        cache_hash_ == req_hash &&
        cache_frame_ == req_frame &&
        cache_view_ == req_view &&
        cache_x_ == req_x &&
        cache_y_ == req_y &&
        cache_w_ == req_w &&
        cache_h_ == req_h &&
        ((req_w <= 0 || req_h <= 0) || cached_rgba_)) {
      return;
    }

    clear_cache_state();
    cache_x_ = req_x;
    cache_y_ = req_y;
    cache_w_ = req_w;
    cache_h_ = req_h;
    cache_frame_ = req_frame;
    cache_view_ = req_view;
    cached_rgba_.reset();
    cached_extra_channels_.clear();
    cached_extra_values_.reset();

    if (req_w <= 0 || req_h <= 0) {
      cached_rgba_ = std::make_shared<std::vector<float>>();
      cached_extra_channels_.clear();
      cached_extra_values_.reset();
      cache_hash_ = req_hash;
      cache_hash_valid_ = true;
      cache_valid_ = true;
      return;
    }

    const Box cache_box(req_x, req_y, req_r, req_t);
    ImagePlane source_plane(cache_box, true, Mask_RGBA);
    input0().fetchPlane(source_plane);

    std::vector<float> source_owned;
    TBlurInputDesc source_desc;
    const bool source_desc_ok = fill_source_desc(source_plane, req_w, req_h, &source_desc);
    const float* source_ptr = source_desc.packed_data;
    const bool source_desc_valid =
        source_desc_ok &&
        (source_desc.packed_data || source_desc.planar_data);
    if (!source_desc_valid) {
      assign_black_opaque_rgba(req_w, req_h, source_owned);
      source_ptr = source_owned.data();
      clear_input_desc(&source_desc);
      source_desc.packed_data = source_ptr;
    }
    const bool source_ready = (source_ptr != nullptr) || source_desc_valid;
    if (aborted() || cancelled()) {
      clear_cache_state();
      return;
    }

    const float guide_mix_ui = static_cast<float>(clamp_double(guide_influence_, 0.0, 1.0));
    const bool show_guide_edge = show_guide_edge_;
    const bool has_guide_input = has_connected_input(1);
    const bool wants_guide =
        has_guide_input && ((guide_mix_ui > 1e-6f) || show_guide_edge);
    TBlurInputDesc guide_desc;
    clear_input_desc(&guide_desc);
    int guide_components = 0;
    bool use_guide = false;
    if (wants_guide) {
      if (Iop* guide_iop = input(1)) {
        const bool guide_rgb_mode = (guide_mode_ == GUIDE_RGB);
        ImagePlane guide_plane(
            cache_box,
            true,
            guide_rgb_mode ? Mask_RGB : Mask_Red);
        guide_iop->fetchPlane(guide_plane);

        if (guide_rgb_mode) {
          if (fill_rgb_desc(guide_plane, req_w, req_h, &guide_desc)) {
            guide_components = 3;
            use_guide = true;
          }
        } else {
          if (fill_scalar_desc(
                  guide_plane, req_w, req_h, guide_plane.chanNo(Chan_Red), &guide_desc)) {
            guide_components = 1;
            use_guide = true;
          }
        }
      }
    }
    if (aborted() || cancelled()) {
      clear_cache_state();
      return;
    }

    TBlurInputDesc mask_desc;
    clear_input_desc(&mask_desc);
    int mask_components = 0;
    bool use_mask = false;
    if (has_connected_input(2)) {
      if (Iop* mask_iop = input(2)) {
        ImagePlane mask_plane(cache_box, true, Mask_Red | Mask_Alpha);
        mask_iop->fetchPlane(mask_plane);
        int mask_red = -1;
        int mask_alpha = -1;
        if (fill_mask_desc(
                mask_plane,
                req_w,
                req_h,
                &mask_desc,
                &mask_components,
                &mask_red,
                &mask_alpha)) {
          use_mask = true;
        }
      }
    }
    if (aborted() || cancelled()) {
      clear_cache_state();
      return;
    }

    std::vector<float> out_rgba;
    out_rgba.reserve(static_cast<size_t>(req_w) * static_cast<size_t>(req_h) * 4);
    std::vector<Channel> extra_channels_result;
    std::vector<float> extra_values_result;

    const float mix = static_cast<float>(clamp_double(mix_, 0.0, 1.0));
    const int iterations = clamp_int(iterations_, 1, 16);
    const float blur_amount = map_blur_size_ui(blur_size_);
    const float edge_boost = static_cast<float>(clamp_double(edge_boost_, 0.0, 2.0));
    const float anisotropy = static_cast<float>(clamp_double(anisotropy_, -1.0, 1.0));

    bool ran_cuda = false;
    bool cuda_ok = false;
    const bool blur_active =
        show_guide_edge ||
        ((mix > 0.0f) && (blur_amount > 0.0f) && (iterations > 0));

    if (!blur_active) {
      if (source_desc.packed_data) {
        copy_direct_rgba_to_vector(source_desc.packed_data, req_w, req_h, out_rgba);
      } else if (source_desc.planar_data) {
        copy_source_desc_to_rgba(source_desc, req_w, req_h, out_rgba);
      } else {
        out_rgba = source_owned;
      }
      sanitize_rgba_buffer_in_place(out_rgba);
      cached_rgba_ = std::make_shared<std::vector<float>>(std::move(out_rgba));
      cache_hash_ = req_hash;
      cache_hash_valid_ = true;
      cache_valid_ = true;
      return;
    }

#ifdef TBLUR_CUDA
    const bool cuda_available = (cached_cuda_is_available() == 1);
    if (!cuda_available) {
      append_runtime_log("TBlur CUDA unavailable -> passthrough (no blur applied).");
      error("TBlur: CUDA backend unavailable; node is in passthrough.");
    }
    if (cuda_available) {
      if (safe_cuda_prepare(req_w, req_h) == 1) {
        ran_cuda = true;

        int effective_iterations = iterations;
        if (quality_mode_ == QUALITY_DRAFT) {
          effective_iterations = std::max(1, iterations - 2);
        }

        const float sigma_s_base =
            std::max(0.5f, std::sqrt(std::max(0.0f, blur_amount)));
        const float axis_scale = std::exp2(anisotropy * 1.15f);
        const float sigma_x_base = std::max(0.5f, sigma_s_base * axis_scale);
        const float sigma_y_base = std::max(0.5f, sigma_s_base / axis_scale);
        const float sigma_x = std::max(
            0.5f,
            sigma_x_base / std::sqrt(static_cast<float>(effective_iterations)));
        const float sigma_y = std::max(
            0.5f,
            sigma_y_base / std::sqrt(static_cast<float>(effective_iterations)));
        const int radius_x = clamp_int(
            std::min(static_cast<int>(std::ceil(2.5f * sigma_x)), kMaxFilterRadius),
            1,
            kMaxFilterRadius);
        const int radius_y = clamp_int(
            std::min(static_cast<int>(std::ceil(2.5f * sigma_y)), kMaxFilterRadius),
            1,
            kMaxFilterRadius);

        const float inv2_sig_t_x = 1.0f / (2.0f * sigma_x * sigma_x);
        const float inv2_sig_t_y = 1.0f / (2.0f * sigma_y * sigma_y);
        const float sigma_r = map_edge_threshold(edge_threshold_);

        const float edge_ui = clamp01(static_cast<float>(edge_threshold_));
        const float edge_hold = std::pow(1.0f - edge_ui, 1.10f);
        const float blur_log2 = std::log2(1.0f + std::max(0.0f, blur_amount));
        const float blur_edge_t = clamp01((blur_log2 - 1.5f) / 10.0f);

        float k = (sigma_s_base / sigma_r) * (0.22f + 0.95f * edge_hold);
        k *= 0.58f;
        k /= (1.0f + static_cast<float>(edge_smooth_) * 8.0f);

        float edge_weight = (0.02f + 0.85f * edge_hold);
        edge_weight *= (1.0f + 0.28f * blur_edge_t);

        const float edge_boost_delta = edge_boost - 1.0f;
        const float edge_boost_scale = std::exp2(edge_boost_delta * 0.85f);
        k *= edge_boost_scale;
        edge_weight *= edge_boost_scale;

        float artifact_risk = clamp01(
            clamp01((blur_log2 - 7.0f) / 4.0f) *
            clamp01((0.35f - edge_ui) / 0.35f) *
            (1.0f - 0.60f * clamp01(static_cast<float>(edge_smooth_))));

        int sample_step = 1;
        const int radius_max = std::max(radius_x, radius_y);
        if (radius_max >= 48 || effective_iterations >= 12 || blur_amount >= 1200.0f) {
          sample_step = 2;
        }
        if (radius_max >= 120 || effective_iterations >= 16 || blur_amount >= 6000.0f) {
          sample_step = 3;
        }
        if (radius_max >= 220 || blur_amount >= 18000.0f) {
          sample_step = 4;
        }
        if (quality_mode_ == QUALITY_DRAFT) {
          sample_step = std::min(8, sample_step + 1);
        } else {
          sample_step = std::max(1, sample_step - 1);
        }
        if (artifact_risk > 0.12f) {
          const int max_step = (artifact_risk > 0.32f) ? 1 : 2;
          sample_step = std::min(sample_step, max_step);
        }

        float hard_stop_mix = clamp01(
            0.12f * std::pow(edge_hold, 2.2f) * (1.0f + 0.25f * blur_edge_t));
        hard_stop_mix = clamp01(
            hard_stop_mix * (0.85f + 0.30f * edge_boost));
        float edge_gate = 0.22f + 0.68f * std::pow(edge_ui, 1.10f);
        edge_gate = clamp01(edge_gate * (1.0f - 0.08f * edge_boost_delta));
        float edge_norm_scale = 0.35f + 0.75f * edge_hold;
        edge_norm_scale = std::max(0.05f, edge_norm_scale / edge_boost_scale);

        float edge_despeckle_mix = clamp01(
            0.06f +
            0.58f * clamp01(static_cast<float>(edge_smooth_)) +
            0.24f * artifact_risk);
        if (quality_mode_ == QUALITY_DRAFT) {
          edge_despeckle_mix *= 0.70f;
        } else {
          edge_despeckle_mix = clamp01(edge_despeckle_mix * 1.08f);
        }

        float cleanup_strength = compute_organic_cleanup_strength(
            blur_amount,
            static_cast<float>(edge_threshold_),
            static_cast<float>(edge_smooth_),
            effective_iterations);
        if (quality_mode_ == QUALITY_DRAFT) {
          cleanup_strength *= 0.55f;
        } else {
          cleanup_strength = clamp01(cleanup_strength * 1.10f);
        }

        if (aborted() || cancelled()) {
          clear_cache_state();
          return;
        }
        TBlurDispatch dispatch = {};
        dispatch.width = req_w;
        dispatch.height = req_h;
        dispatch.source = source_desc;
        clear_input_desc(&dispatch.guide);
        if (use_guide) {
          dispatch.guide = guide_desc;
        }
        dispatch.guide_present = use_guide ? 1 : 0;
        dispatch.guide_components = use_guide ? guide_components : 0;
        dispatch.guide_mode_rgb = (guide_mode_ == GUIDE_RGB) ? 1 : 0;
        dispatch.guide_mix = use_guide ? guide_mix_ui : 0.0f;
        clear_input_desc(&dispatch.mask);
        if (use_mask) {
          dispatch.mask = mask_desc;
        }
        dispatch.mask_present = use_mask ? 1 : 0;
        dispatch.mask_components = use_mask ? mask_components : 0;
        dispatch.invert_mask = invert_mask_ ? 1 : 0;
        dispatch.iterations = effective_iterations;
        dispatch.radius_x = radius_x;
        dispatch.radius_y = radius_y;
        dispatch.sample_step_x = sample_step;
        dispatch.sample_step_y = sample_step;
        dispatch.inv2_sig_t_x = inv2_sig_t_x;
        dispatch.inv2_sig_t_y = inv2_sig_t_y;
        dispatch.k = k;
        dispatch.edge_weight = edge_weight;
        dispatch.edge_despeckle_mix = edge_despeckle_mix;
        dispatch.hard_stop_mix = hard_stop_mix;
        dispatch.edge_gate = edge_gate;
        dispatch.edge_norm_scale = edge_norm_scale;
        dispatch.show_guide_edge = show_guide_edge ? 1 : 0;
        dispatch.keep_alpha = keep_alpha_ ? 1 : 0;
        dispatch.mix = show_guide_edge ? 1.0f : mix;
        dispatch.organic_cleanup_strength = cleanup_strength;

        out_rgba.assign(static_cast<size_t>(req_w) * static_cast<size_t>(req_h) * 4, 0.0f);
        cuda_ok = (source_ready && safe_cuda_process(&dispatch, out_rgba.data()) == 1);

        if (cuda_ok && !show_guide_edge) {
          ChannelSet extra_channels = channels_;
          if (channels_.all()) {
            // "all" means: process every channel actually requested in this render.
            // This keeps iteration finite and matches Nuke's lazy channel evaluation.
            extra_channels = requested_channels();
          }
          extra_channels &= input0().channels();
          extra_channels -= Mask_RGBA;
          if (extra_channels) {
            ImagePlane extra_plane(cache_box, true, extra_channels);
            input0().fetchPlane(extra_plane);

            const size_t pixel_count =
                static_cast<size_t>(req_w) * static_cast<size_t>(req_h);
            extra_channels_result.clear();
            for (Channel z : extra_channels) {
              extra_channels_result.push_back(z);
            }
            extra_values_result.assign(
                extra_channels_result.size() * pixel_count,
                0.0f);

            std::vector<float> channel_owned;
            std::vector<float> temp_source;
            std::vector<float> temp_out;
            temp_source.assign(pixel_count * 4, 0.0f);
            temp_out.assign(pixel_count * 4, 0.0f);

            for (size_t ci = 0; ci < extra_channels_result.size(); ++ci) {
              const Channel z = extra_channels_result[ci];
              const int pref_chan = extra_plane.chanNo(z);
              bool channel_is_direct = false;
              const float* channel_direct_ptr = nullptr;
              channel_owned.clear();
              (void)extract_single_channel_from_plane(
                  extra_plane,
                  req_w,
                  req_h,
                  pref_chan,
                  channel_owned,
                  &channel_is_direct,
                  &channel_direct_ptr);
              const float* channel_ptr = nullptr;
              if (channel_is_direct && channel_direct_ptr) {
                channel_ptr = channel_direct_ptr;
              } else if (!channel_owned.empty()) {
                channel_ptr = channel_owned.data();
              }
              if (!channel_ptr) {
                continue;
              }

              for (size_t p = 0; p < pixel_count; ++p) {
                const float v = finite_or(channel_ptr[p], 0.0f);
                temp_source[p * 4 + 0] = v;
                temp_source[p * 4 + 1] = v;
                temp_source[p * 4 + 2] = v;
                temp_source[p * 4 + 3] = 1.0f;
              }

              TBlurDispatch channel_dispatch = dispatch;
              clear_input_desc(&channel_dispatch.source);
              channel_dispatch.source.packed_data = temp_source.data();
              channel_dispatch.show_guide_edge = 0;
              channel_dispatch.keep_alpha = 0;
              channel_dispatch.mix = mix;
              const int channel_ok = safe_cuda_process(&channel_dispatch, temp_out.data());
              if (channel_ok != 1) {
                for (size_t p = 0; p < pixel_count; ++p) {
                  extra_values_result[ci * pixel_count + p] = finite_or(channel_ptr[p], 0.0f);
                }
              } else {
                for (size_t p = 0; p < pixel_count; ++p) {
                  extra_values_result[ci * pixel_count + p] = finite_or(temp_out[p * 4 + 0], 0.0f);
                }
              }
            }
          }
        }

        if (aborted() || cancelled()) {
          clear_cache_state();
          return;
        }
      }
    }
#else
    (void)mix;
    (void)blur_amount;
    (void)iterations;
#endif

    if (!ran_cuda || !cuda_ok) {
      if (source_desc.packed_data) {
        copy_direct_rgba_to_vector(source_desc.packed_data, req_w, req_h, out_rgba);
      } else if (source_desc.planar_data) {
        copy_source_desc_to_rgba(source_desc, req_w, req_h, out_rgba);
      } else {
        out_rgba = source_owned;
      }
      if (out_rgba.empty()) {
        out_rgba.assign(static_cast<size_t>(req_w) * static_cast<size_t>(req_h) * 4, 0.0f);
      }
      if (ran_cuda && !cuda_ok) {
        std::string cuda_error = fetch_cuda_last_error();
        if (cuda_error.empty()) {
          cuda_error = "unknown CUDA backend failure";
        }
        append_runtime_log(("TBlur: CUDA dispatch failed, passthrough applied: " + cuda_error).c_str());
        error("TBlur: CUDA failed: %s", cuda_error.c_str());
      }
    }

    if (aborted() || cancelled()) {
      clear_cache_state();
      return;
    }
    sanitize_rgba_buffer_in_place(out_rgba);
    cached_rgba_ = std::make_shared<std::vector<float>>(std::move(out_rgba));
    cached_extra_channels_ = std::move(extra_channels_result);
    if (!extra_values_result.empty()) {
      cached_extra_values_ =
          std::make_shared<std::vector<float>>(std::move(extra_values_result));
    } else {
      cached_extra_values_.reset();
    }
    cache_hash_ = req_hash;
    cache_hash_valid_ = true;
    cache_valid_ = true;
  }

  static const Iop::Description d;
};

static Iop* build(Node* node) { return new TBlurBase(node); }
const Iop::Description TBlurBase::d(CLASS, "Filter/TBlur", build);

extern "C" void tblur_base_keepalive() {}
extern "C" void FnPlugin_GetAPI(int) {}
