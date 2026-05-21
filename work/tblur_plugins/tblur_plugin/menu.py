"""Plugin creation script for user interface in Nuke."""

try:
    from tblur_plugin._tblur_menu_creator import add_menu
except Exception:
    from _tblur_menu_creator import add_menu

add_menu()

