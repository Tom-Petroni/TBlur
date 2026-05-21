"""Constants for TBlur Nuke plugin."""

import os

INSTALLATION_PATH = os.path.dirname(os.path.abspath(__file__))
NODE_CLASS_NAME = "TBlur"
NODE_MENU_NAME = "TBlur"
LOADED_ENV_KEY = "TBLUR_LOADED"
INSTALL_ENV_KEY = "TBLUR_INSTALLATION"
MENU_ICON_NAME = "Bilateral.png"
MENU_ICON_FALLBACK = "Blur.png"

PRODUCT_NAME = "TBlur"
PRODUCT_VERSION = "1.0"
PRODUCT_RELEASE_YEAR = "2026"
PRODUCT_VENDOR = "Thomas Petroni"
PRODUCT_VENDOR_URL = "https://www.linkedin.com/in/thomas-petroni/"


def product_credits_html() -> str:
    return (
        f"{PRODUCT_NAME} {PRODUCT_VERSION} - {PRODUCT_RELEASE_YEAR} - "
        f"<a href='{PRODUCT_VENDOR_URL}' "
        "style='text-decoration: underline; color: #9ec3ff;'>"
        f"{PRODUCT_VENDOR}"
        "</a>"
    )

