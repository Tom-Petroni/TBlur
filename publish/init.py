"""TBlur init.py to load plugin in Nuke."""

import os

import nuke  # ty:ignore[unresolved-import]

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
nuke.pluginAddPath(  # ty:ignore[unresolved-attribute]
    os.path.join(_THIS_DIR, "tblur_plugin").replace(os.sep, "/"),
)
