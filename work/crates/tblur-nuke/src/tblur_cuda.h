#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Input descriptors
//
// Each input (source / guide / mask) is described by either:
//   - a packed buffer: contiguous, RGBA-packed for source, single-channel for
//     guide-luma / mask, RGB-packed for guide-RGB. `planar_data` must be null.
//   - a planar buffer: a raw `ImagePlane::readable()` pointer with arbitrary
//     row/col strides and channel indices. The GPU runs a deinterleave kernel
//     to produce a packed buffer in device memory. `packed_data` must be null.
//
// Channel indices are read in the order chan_a..chan_d. For source, that maps
// to R,G,B,A. For guide-RGB it's R,G,B (chan_d ignored). For guide-luma and
// mask, only chan_a (and chan_b for masks with both red+alpha) is used.
// ---------------------------------------------------------------------------
typedef struct TBlurInputDesc {
  const float* packed_data;        // packed: tightly-packed n-component data; null if planar
  const float* planar_data;        // planar: ImagePlane::readable(); null if packed
  int plane_size_floats;           // length of planar_data (typically row_stride * height)
  int row_stride;                  // in floats
  int col_stride;                  // in floats
  int chan_a, chan_b, chan_c, chan_d; // -1 if unused
} TBlurInputDesc;

// ---------------------------------------------------------------------------
// Dispatch params
// ---------------------------------------------------------------------------
typedef struct TBlurDispatch {
  int width;
  int height;

  // Source: required. R=chan_a, G=chan_b, B=chan_c, A=chan_d.
  TBlurInputDesc source;

  // Guide: optional. Set guide_present=0 to skip.
  // guide_components: 1 (luma) or 3 (rgb).
  // guide_mode_rgb: when 1 and components==3, GPU computes luma from RGB.
  TBlurInputDesc guide;
  int guide_present;
  int guide_components;
  int guide_mode_rgb;
  float guide_mix;

  // Mask: optional. Set mask_present=0 to skip.
  // mask_components: 1 or 2. chan_a=red index, chan_b=alpha index (-1 if absent).
  TBlurInputDesc mask;
  int mask_present;
  int mask_components;
  int invert_mask;

  // Filter params
  int iterations;
  int radius_x, radius_y;
  int sample_step_x, sample_step_y;
  float inv2_sig_t_x, inv2_sig_t_y;
  float k;
  float edge_weight;
  float edge_despeckle_mix;
  float hard_stop_mix;
  float edge_gate;
  float edge_norm_scale;
  int show_guide_edge;
  int keep_alpha;
  float mix;
  float organic_cleanup_strength;
} TBlurDispatch;

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------
int cuda_is_available(void);
int cuda_get_device_name(char* out_name, int out_size);
int cuda_prepare(int width, int height);
int cuda_process(const TBlurDispatch* dispatch, float* out_rgba);

// Returns the last CUDA error message into out_msg (empty string if none).
// Thread-safe.
int cuda_get_last_error(char* out_msg, int out_size);

void cuda_cleanup(void);

#ifdef __cplusplus
}
#endif
