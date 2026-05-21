"""Main entry point for TBlur Nuke plugin."""

import logging

try:
    from tblur_plugin._tblur_node_setup import setup_node_ui
    from tblur_plugin._tblur_plugin_loader import add_plugin_path_safe
except Exception:
    from _tblur_node_setup import setup_node_ui
    from _tblur_plugin_loader import add_plugin_path_safe

FORMAT = "[%(asctime)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)

add_plugin_path_safe()
setup_node_ui()

