"""Look of the window: the Sun Valley theme (light or dark) and the colors ttk cannot reach."""

import tkinter as tk
from tkinter import ttk

import darkdetect
import sv_ttk

#: Colors for the plain tk log and the hint labels, matching the Sun Valley palettes.
COLORS = {
    "light": {
        "hint": "#5f5f5f",
        "base": "#ffffff",
        "text": "#1c1c1c",
        "select": "#005fb8",
        "select_text": "#ffffff",
    },
    "dark": {
        "hint": "#a6a6a6",
        "base": "#1c1c1c",
        "text": "#fafafa",
        "select": "#57c8ff",
        "select_text": "#1c1c1c",
    },
}


def system_theme() -> str:
    """``"dark"`` if Windows (or the OS) is set to dark, else ``"light"``."""
    return "dark" if darkdetect.isDark() else "light"


def current() -> str:
    return "dark" if sv_ttk.get_theme() == "dark" else "light"


def apply(theme: str) -> None:
    sv_ttk.set_theme(theme)


def restyle(log: tk.Text) -> None:
    """Re-apply the colors ttk themes do not set; call after every theme change."""
    colors = COLORS[current()]
    ttk.Style().configure("Hint.TLabel", foreground=colors["hint"])
    log.configure(
        background=colors["base"],
        foreground=colors["text"],
        insertbackground=colors["text"],
        selectbackground=colors["select"],
        selectforeground=colors["select_text"],
        relief="flat",
        borderwidth=0,
        padx=6,
        pady=4,
    )
