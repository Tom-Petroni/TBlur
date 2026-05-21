"""Functions that handle creation of the TBlur menu."""

from __future__ import annotations

import logging
import os

import nuke  # ty:ignore[unresolved-import]

try:
    from tblur_plugin._tblur_consts import (
        INSTALLATION_PATH,
        INSTALL_ENV_KEY,
        LOADED_ENV_KEY,
        MENU_ICON_FALLBACK,
        MENU_ICON_NAME,
        NODE_CLASS_NAME,
        NODE_MENU_NAME,
    )
    from tblur_plugin._tblur_plugin_loader import ensure_node_class_loaded
except Exception:
    from _tblur_consts import (
        INSTALLATION_PATH,
        INSTALL_ENV_KEY,
        LOADED_ENV_KEY,
        MENU_ICON_FALLBACK,
        MENU_ICON_NAME,
        NODE_CLASS_NAME,
        NODE_MENU_NAME,
    )
    from _tblur_plugin_loader import ensure_node_class_loaded

logger = logging.getLogger(__name__)


def _pick_menu_icon() -> str:
    executable = nuke.env.get("ExecutablePath", "")
    if executable:
        icon_dir = os.path.join(os.path.dirname(executable), "icons")
        preferred = os.path.join(icon_dir, MENU_ICON_NAME)
        if os.path.isfile(preferred):
            return MENU_ICON_NAME
    return MENU_ICON_FALLBACK


def _create_node_from_menu() -> None:
    try:
        ensure_node_class_loaded()
        nuke.createNode(NODE_CLASS_NAME)
    except Exception as exc:
        nuke.tprint(f"[TBlur] Unable to create node '{NODE_CLASS_NAME}': {exc}")
        nuke.message(
            "TBlur could not be loaded correctly.\n\n"
            f"Detail: {exc}",
        )


def _create_menu() -> None:
    toolbar = nuke.menu("Nodes")
    icon_name = _pick_menu_icon()
    menu = toolbar.addMenu(NODE_MENU_NAME, icon=icon_name)
    callback = f"import {__name__} as _tblur_menu; _tblur_menu._create_node_from_menu()"
    menu.addCommand(
        NODE_CLASS_NAME,
        callback,
        icon=icon_name,
    )


def add_menu() -> None:
    if os.getenv(LOADED_ENV_KEY) != "1":
        return

    _add_menu_dependencies_to_plugin_path()
    _set_installation_directory()
    _create_menu()


def _add_menu_dependencies_to_plugin_path() -> None:
    resources = os.path.join(INSTALLATION_PATH, "resources").replace(os.sep, "/")
    if os.path.isdir(resources):
        nuke.pluginAppendPath(resources)


def _set_installation_directory() -> None:
    os.environ[INSTALL_ENV_KEY] = str(INSTALLATION_PATH)
