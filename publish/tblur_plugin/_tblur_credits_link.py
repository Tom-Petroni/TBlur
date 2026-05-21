"""Inline credits link widget for TBlur knobs."""

from __future__ import annotations

import webbrowser

try:
    from tblur_plugin._tblur_consts import product_credits_html
except Exception:
    from _tblur_consts import product_credits_html

try:
    from PySide2 import QtCore, QtWidgets  # ty: ignore[import-not-found]
except Exception:
    try:
        from PySide6 import QtCore, QtWidgets  # ty: ignore[import-not-found]
    except Exception:
        QtCore = None
        QtWidgets = None

_CREDITS_HTML = product_credits_html()


if QtWidgets is not None:

    class _CreditsLinkWidget(QtWidgets.QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setSizePolicy(
                QtWidgets.QSizePolicy(
                    QtWidgets.QSizePolicy.Preferred,
                    QtWidgets.QSizePolicy.Fixed,
                ),
            )

            layout = QtWidgets.QHBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            self._label = QtWidgets.QLabel(_CREDITS_HTML, self)
            self._label.setTextFormat(QtCore.Qt.RichText)
            self._label.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)
            self._label.setOpenExternalLinks(False)
            self._label.setStyleSheet("QLabel { background: transparent; border: none; }")
            self._label.linkActivated.connect(self._open_link)
            self._label.setSizePolicy(
                QtWidgets.QSizePolicy(
                    QtWidgets.QSizePolicy.Maximum,
                    QtWidgets.QSizePolicy.Fixed,
                ),
            )

            layout.addWidget(self._label)
            layout.addStretch(1)

            fixed_h = max(18, int(self._label.sizeHint().height()) + 2)
            self.setMinimumHeight(fixed_h)
            self.setMaximumHeight(fixed_h)

        def _open_link(self, link: str) -> None:
            webbrowser.open(link)


class TBlurCreditsLinkKnob:
    """Bridge object used by Nuke PythonKnob/PyCustom to create credit link UI."""

    def makeUI(self):
        if QtWidgets is None:
            return None
        return _CreditsLinkWidget()

