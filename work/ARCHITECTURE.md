# TBlur Nuke Plugin - Architecture

`TBlur` is a **GPU-only** Nuke filter node. Every filtering stage runs on
the device; the C++ side only handles knobs, caching, and dispatching.

- C++ for Nuke node integration, knob layout, and full-frame cache
- CUDA C++ for the entire image pipeline
- Rust for minimal plugin linkage shim

Core files:

- `crates/tblur-nuke/src/tblur_base.cpp`     (node, knobs, cache, GPU dispatch)
- `crates/tblur-nuke/src/tblur_cuda.cu`      (kernels, persistent device buffers, pinned host staging)
- `crates/tblur-nuke/src/tblur_cuda.h`       (backend C ABI)
- `crates/tblur-nuke/src/lib.rs`             (Rust linkage shim)
- `crates/tblur-nuke/build.rs`               (cc + nvcc orchestration)
- `xtask/src/nuke/compile.rs`                (Nuke version dispatch)

## Runtime flow

1. `_request()` asks input 0 (source), and conditionally inputs 1 (guide)
   and 2 (mask), for the full frame.
2. `engine()` triggers `ensure_cache()`, which builds (or returns from
   cache) a full-frame RGBA buffer keyed by `Iop::hash()`.
3. Inside `ensure_cache()`:
   - **show_guide_** → lightweight CPU debug viz (luma / guide RGB only).
   - **Otherwise** → `try_cuda_backend()`:
     1. `cuda_backend_prepare(w,h)` — lazy `cudaMalloc` of all persistent
        device buffers + `cudaHostAlloc` of pinned staging.
     2. Per-input hash check: skip upload if the input's hash matches
        what was last uploaded.
     3. Upload — fast path if the source/guide plane is already
        RGBA-packed (single pinned `cudaMemcpyAsync`); otherwise pass the
        raw `ImagePlane` buffer + per-channel offsets to a CUDA
        `deinterleave_*_kernel` that scatters into the typed device
        buffer in one launch (no host-side scatter).
     4. `cuda_backend_process()` — fused chain of kernels on the backend
        stream: source-luma / guide-luma-mix → edge map (Lab ΔE76 +
        multi-scale guide gradients + dark-speckle suppression) →
        optional 3-pass despeckle → N×(H,V) domain-transform passes
        ping-ponging between `d_filter_a/b` → optional blend with source
        through `mask × mix` → optional organic 3×3 cleanup.
     5. D2H readback through pinned staging, then plain memcpy into the
        cache buffer.
   - **On CUDA failure** → `Op::error()` with the underlying CUDA error
     string, then write source passthrough into the cache. The downstream
     graph stays valid.
4. `engine()` copies the requested rows from the cache.

## Why these specific choices

- **No CPU fallback path** — predictable behavior (and predictable
  performance). When the GPU isn't available the user gets a clear error
  rather than a 4K render that's "just slow."
- **Channel-major upload kernels** — Nuke ImagePlane buffers very rarely
  match the GPU's expected RGBA-packed layout (different channels often
  come from different upstream nodes). Doing the deinterleave on-device
  eliminates the dominant per-frame CPU spike during slider scrubbing.
- **Pinned host memory** — both H2D and D2H avoid pageable bounce
  buffers, roughly doubling effective PCIe bandwidth in our measurements.
- **`std::mutex` not spinlock** — Nuke runs many worker threads; a
  spinlock blocked the UI thread on contention during scrubbing.
- **Knob quantization in `build_render_tuning()`** — interactive previews
  snap blur/edge/mix sliders to coarse steps and cap iterations so each
  scrub frame is fast enough to be interactive.

For full per-knob behavior:

- `docs/TBLUR_DOCUMENTATION_FR.md`
- `docs/TBLUR_NODE_GUIDE.md`
