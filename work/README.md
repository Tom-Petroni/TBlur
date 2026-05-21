# TBlur Nuke Plugin

TBlur is a Nuke filter node for edge-preserving blur — a CUDA implementation
of Gastal & Oliveira's domain-transform recursive filter, tuned for
Spider-Verse-style NPR pipelines. The node is **GPU-only**: every stage
(source/guide/mask deinterleave, edge map, despeckle, separable filter,
blend, organic cleanup) runs on the device.

## Requirements

- NVIDIA GPU with compute capability 7.5+ (Turing, Ampere, Ada, Hopper)
- Recent NVIDIA driver
- If CUDA is unavailable at runtime, the node sets a Nuke-level error and
  passes the source through unfiltered (no silent CPU fallback).

## Main Docs

- Full technical documentation (FR):
  - `docs/TBLUR_DOCUMENTATION_FR.md`
- Node usage guide:
  - `docs/TBLUR_NODE_GUIDE.md`
- High level architecture notes:
  - `ARCHITECTURE.md`

## Quick Build (Nuke 16, Windows)

Run from repo root, inside a `vcvars64`-initialized shell so `nvcc` can find
`cl.exe`:

```bash
cargo xtask --compile --nuke-versions 16.0 --target-platform windows --output-to-package --limit-threads --cuda-backend
```

For Nuke 17, swap `--nuke-versions 16.0` for `--nuke-versions 17.0`.

Generated binary path:

`tblur_plugins/tblur_plugin/bin/16.0/windows/x86_64/TBlur.dll`

## Quick Deploy To `.nuke`

Copy the built plugin to:

`%USERPROFILE%\.nuke\TBlur\bin\16.0\windows\x86_64\TBlur.dll`

Then restart Nuke fully.

## What changed in the full-GPU rewrite

- Removed the CPU domain-transform / edge / despeckle / organic-cleanup
  paths entirely (~600 lines).
- Added GPU `deinterleave_*_kernel` so non-RGBA-packed source/guide/mask
  inputs are scattered on-device instead of via a host `parallel_for_rows`
  pass — eliminates the dominant slider-lag spike at 4K.
- Pinned host staging (`cudaHostAlloc`) on both upload and readback paths.
- `std::mutex` on the backend instead of a busy spinlock.
- Per-input upload hash tracking, so changing only `mix` or `iterations`
  no longer re-uploads the source / guide / mask.
- Clean error reporting: on CUDA failure, the node calls `Op::error()` with
  the underlying CUDA error string and passes source through, instead of
  silently CPU-rendering.

## Repo Map

- `crates/tblur-nuke/src/tblur_base.cpp`: Nuke node, knobs, cache, GPU dispatch
- `crates/tblur-nuke/src/lib.rs`: minimal Rust shim for plugin linkage
- `crates/tblur-nuke/src/tblur_cuda.cu`: CUDA backend (kernels, persistent buffers, pinned staging, channel-major upload)
- `crates/tblur-nuke/src/tblur_cuda.h`: backend C ABI
- `crates/tblur-nuke/build.rs`: C++ + nvcc compile and DDImage linkage
- `xtask/`: source fetch/build/package automation
- `tblur_plugins/tblur_plugin/`: Python package loaded by Nuke (`init.py`, `menu.py`, plugin path resolver)

