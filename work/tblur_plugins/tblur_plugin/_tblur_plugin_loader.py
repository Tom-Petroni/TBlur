"""Plugin loader and lookup script for TBlur."""

from __future__ import annotations

import logging
import os
import platform
from datetime import datetime

import nuke  # ty:ignore[unresolved-import]

try:
    from tblur_plugin._tblur_consts import INSTALLATION_PATH, LOADED_ENV_KEY, NODE_CLASS_NAME
except Exception:
    from _tblur_consts import INSTALLATION_PATH, LOADED_ENV_KEY, NODE_CLASS_NAME

logger = logging.getLogger(__name__)

NUKE_ARM_VERSION = 15


class PluginNotFoundError(Exception):
    """Raised when the plugin path or binary is not found."""


class PluginLoadError(Exception):
    """Raised when Nuke cannot load the plugin binary."""


class UnsupportedSystemError(Exception):
    """Raised when the operating system is not supported."""


def _get_nuke_version() -> str:
    return f"{nuke.NUKE_VERSION_MAJOR}.{nuke.NUKE_VERSION_MINOR}"


def _get_operating_system_name() -> str:
    operating_system = platform.system().lower()
    if "linux" in operating_system:
        return "linux"
    if "windows" in operating_system:
        return "windows"
    if "darwin" in operating_system:
        return "macos"
    raise UnsupportedSystemError(f"System '{operating_system}' is not supported.")


def _get_arch() -> str:
    architecture = (platform.machine() or platform.processor() or "").lower()
    is_macos = platform.system().lower() == "darwin"

    if architecture in ("x86_64", "amd64", "x64", "x86-64", "em64t"):
        return "x86_64"

    if not is_macos:
        raise UnsupportedSystemError(f"Architecture '{architecture}' is not supported.")

    if architecture in ("x86_64", "amd64", "x64", "x86-64", "em64t", "i386"):
        return "x86_64"

    if "arm" in architecture or "aarch" in architecture:
        if nuke.NUKE_VERSION_MAJOR >= NUKE_ARM_VERSION:
            return "aarch64"
        return "x86_64"

    raise UnsupportedSystemError(f"Architecture '{architecture}' is not supported.")


def _library_filename() -> str:
    override = os.environ.get("TBLUR_BINARY_FILENAME", "").strip()
    if override:
        return override

    os_name = _get_operating_system_name()
    if os_name == "windows":
        return f"{NODE_CLASS_NAME}.dll"
    if os_name == "linux":
        return f"lib{NODE_CLASS_NAME}.so"
    return f"lib{NODE_CLASS_NAME}.dylib"


def _candidate_binary_filenames(os_name: str) -> list[str]:
    override = os.environ.get("TBLUR_BINARY_FILENAME", "").strip()
    if override:
        return [override]

    if os_name == "windows":
        # Allows hot-swapping when canonical DLL is file-locked by a running Nuke.
        return [f"{NODE_CLASS_NAME}_hotfix.dll", f"{NODE_CLASS_NAME}.dll"]

    return [_library_filename()]


def _build_plugin_path(
    version_folder: str = "",
    os_name: str = "",
    arch_name: str = "",
) -> str:
    version_folder = version_folder or _get_nuke_version()
    os_name = os_name or _get_operating_system_name()
    arch_name = arch_name or _get_arch()
    return os.path.join(
        INSTALLATION_PATH,
        "bin",
        version_folder,
        os_name,
        arch_name,
    ).replace(os.sep, "/")


def _build_binary_path(plugin_path: str) -> str:
    return os.path.join(plugin_path, _library_filename()).replace(os.sep, "/")


def _resolve_binary_path(plugin_path: str, os_name: str = "") -> str:
    os_name = os_name or _get_operating_system_name()
    for filename in _candidate_binary_filenames(os_name):
        binary_path = os.path.join(plugin_path, filename).replace(os.sep, "/")
        if os.path.isfile(binary_path):
            return binary_path

    # Fallback for explicit override and clear error path.
    return _build_binary_path(plugin_path)


def _normalize_path(path: str) -> str:
    return os.path.normcase(os.path.normpath(path)).replace("\\", "/")


def _is_minor_version_folder(name: str) -> bool:
    parts = name.split(".")
    return len(parts) == 2 and all(part.isdigit() for part in parts)


def _binary_exists_for_version(version_folder: str, os_name: str, arch_name: str) -> bool:
    plugin_path = _build_plugin_path(version_folder, os_name, arch_name)
    if not os.path.isdir(plugin_path):
        return False
    return os.path.isfile(_resolve_binary_path(plugin_path, os_name))


def _resolve_version_folder() -> str:
    requested = _get_nuke_version()
    os_name = _get_operating_system_name()
    arch_name = _get_arch()
    plugin_bin_root = os.path.join(INSTALLATION_PATH, "bin")

    if _binary_exists_for_version(requested, os_name, arch_name):
        return requested

    if not os.path.isdir(plugin_bin_root):
        return requested

    try:
        available = [
            entry
            for entry in os.listdir(plugin_bin_root)
            if _is_minor_version_folder(entry)
            and os.path.isdir(os.path.join(plugin_bin_root, entry))
        ]
    except OSError:
        return requested

    try:
        requested_major, requested_minor = (
            int(part) for part in requested.split(".", 1)
        )
    except ValueError:
        return requested

    same_major = []
    for entry in available:
        major, minor = (int(part) for part in entry.split(".", 1))
        if major == requested_major:
            same_major.append((minor, entry))

    if not same_major:
        return requested

    lower_or_equal = sorted(
        (entry for minor, entry in same_major if minor <= requested_minor),
        key=lambda version: int(version.split(".", 1)[1]),
        reverse=True,
    )
    higher = sorted(
        (entry for minor, entry in same_major if minor > requested_minor),
        key=lambda version: int(version.split(".", 1)[1]),
    )

    for candidate in (lower_or_equal + higher):
        if _binary_exists_for_version(candidate, os_name, arch_name):
            logger.warning(
                "TBlur binary for '%s' on %s/%s not found, using '%s' fallback.",
                requested,
                os_name,
                arch_name,
                candidate,
            )
            return candidate

    return requested


def _is_plugin_path_registered(path: str) -> bool:
    normalized = _normalize_path(path)
    for existing in nuke.pluginPath():
        if _normalize_path(existing) == normalized:
            return True
    return False


def _ensure_plugin_path_registered(path: str) -> None:
    if _is_plugin_path_registered(path):
        return
    # pluginAddPath keeps this path high priority and avoids stale binaries.
    nuke.pluginAddPath(path)  # ty:ignore[unresolved-attribute]


def _is_node_class_available() -> bool:
    return hasattr(nuke, "nodes") and hasattr(nuke.nodes, NODE_CLASS_NAME)


def _append_loader_log(message: str) -> None:
    try:
        log_path = os.path.join(INSTALLATION_PATH, "loader_runtime.log")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}\n")
    except Exception:
        # Logging must never block plugin load.
        pass


def _load_binary(binary_path: str) -> None:
    _append_loader_log(f"Trying load: {binary_path}")
    try:
        nuke.load(binary_path)  # ty:ignore[unresolved-attribute]
    except Exception as exc:
        _append_loader_log(f"Load failed: {binary_path} ({exc})")
        raise PluginLoadError(
            f"Unable to load '{NODE_CLASS_NAME}' from '{binary_path}': {exc}",
        ) from exc

    if not _is_node_class_available():
        _append_loader_log(
            f"Load incomplete: class '{NODE_CLASS_NAME}' unavailable after loading '{binary_path}'.",
        )
        raise PluginLoadError(
            f"Binary '{binary_path}' loaded but node class '{NODE_CLASS_NAME}' is unavailable.",
        )
    _append_loader_log(f"Load success: class '{NODE_CLASS_NAME}' from '{binary_path}'")


def ensure_node_class_loaded() -> str:
    requested_version = _get_nuke_version()
    resolved_version = _resolve_version_folder()
    os_name = _get_operating_system_name()
    arch_name = _get_arch()
    plugin_path = _build_plugin_path(resolved_version, os_name, arch_name)
    if not os.path.isdir(plugin_path):
        raise PluginNotFoundError(
            (
                "TBlur is installed, but this Nuke version '{}' is not available "
                "in this package."
            ).format(nuke.NUKE_VERSION_STRING),
        )

    binary_path = _resolve_binary_path(plugin_path)
    if not os.path.isfile(binary_path):
        raise PluginNotFoundError(
            f"TBlur binary was not found at '{binary_path}'.",
        )

    _append_loader_log(
        (
            f"Resolved plugin path='{plugin_path}', binary='{binary_path}', "
            f"nuke='{nuke.NUKE_VERSION_STRING}', requested='{requested_version}', "
            f"resolved='{resolved_version}', os='{os_name}', arch='{arch_name}'"
        ),
    )
    _ensure_plugin_path_registered(plugin_path)
    _load_binary(binary_path)
    return plugin_path


def add_plugin_path() -> None:
    os.environ[LOADED_ENV_KEY] = "0"
    ensure_node_class_loaded()
    os.environ[LOADED_ENV_KEY] = "1"


def add_plugin_path_safe() -> None:
    try:
        add_plugin_path()
        logger.info("TBlur plugin loaded successfully.")
    except Exception:
        logger.exception("TBlur plugin loading failed.")
