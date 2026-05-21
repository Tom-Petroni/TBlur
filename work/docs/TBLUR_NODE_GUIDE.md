# TBlur Node Guide

## Purpose

`TBlur` smooths noisy textures while preserving visible edges.

It is available as:

- Node class: `TBlur`
- Node menu path: `Filter/TBlur`

## Inputs

- Input 0 `Source`: image to filter.
- Input 1 `Guide`: optional guidance image.
- Input 2 `mask`: optional mask controlling `mix`.

If `Guide` is not connected, the node uses `Source` as guide.
If `mask` is not connected, mask = `1` everywhere.

## Main Knobs

- `Local GPU`: detected local GPU info line.
- `Use GPU if available`: use CUDA path when available.
- `Vectorize on CPU`: enable threaded CPU filtering.
- `Safety Rails`: conservative settings to reduce artifacts at extreme values.
- `Presets`: look presets plus `Custom`.
- `Blur Type`: `Sharp Edges` or `Soft Edges`.
- `Filter`: `Gauss` or `Box`.
- `Blur Size`: XY blur control (`WH` knob). Slider is `0..100`, manual values above `100` are allowed.
- `Edge Threshold`: edge hold threshold (`0..1`).
- `Edge Smooth`: softens edge hold (`0..1`).
- `Guide Influence`: blend between source guide and guide input (`0..1`).
- `Iterations`: number of domain transform iterations (`1..16`).
- `guide mode`: `Luma` or `RGB`.
- `show guide`: displays guide preview instead of filtered output.
- `mix`: blend filtered result with source (`0..1`).
- `Keep Alpha`: keeps source alpha when enabled, or filters alpha with edge-aware blur when disabled.
- `invert`: inverts the optional mask.

## Practical Notes

- `mix = 0` bypasses filtering.
- `Blur Size = 0` bypasses filtering.
- `show guide = on` skips filtering and shows guide only.
- With high blur values and large frames, `Use GPU if available` can be significantly faster when CUDA is available.

## Mask Behavior

- Uses alpha channel first if available.
- Falls back to red channel if alpha is missing.
- `invert` flips mask (`m` -> `1 - m`).

