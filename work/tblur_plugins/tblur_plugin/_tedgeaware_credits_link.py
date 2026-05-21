"""Compatibility shim for legacy credits knob imports."""

from __future__ import annotations

try:
    from tblur_plugin._tblur_credits_link import TBlurCreditsLinkKnob
except Exception:
    from _tblur_credits_link import TBlurCreditsLinkKnob


class TEdgeAwareCreditsLinkKnob(TBlurCreditsLinkKnob):
    """Legacy alias kept for old node instances."""


__all__ = ["TEdgeAwareCreditsLinkKnob", "TBlurCreditsLinkKnob"]
