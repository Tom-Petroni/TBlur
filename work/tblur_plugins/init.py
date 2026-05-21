"""TBlur init.py to load plugin in Nuke."""

import os

import nuke  # ty:ignore[unresolved-import]

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
nuke.pluginAddPath(os.path.join(_THIS_DIR, "tedgeaware_plugin"))  # ty:ignore[unresolved-attribute]
