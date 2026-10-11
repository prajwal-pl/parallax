from __future__ import annotations

from rich.box import ROUNDED, Box
from rich.theme import Theme

# Semantic icons (clean Unicode developer glyphs, strictly NO emojis)
ICON_AGENT = "●"
ICON_USER = "›"
ICON_SUCCESS = "✓"
ICON_ERROR = "✕"
ICON_WARNING = "!"
ICON_BULLET = "·"
ICON_ARROW = "→"
ICON_CHEVRON = "»"
ICON_COMPACT = "◆"
ICON_RETRY = "↻"
ICON_LINE_V = "│"
ICON_LINE_BRANCH = "├─"
ICON_LINE_END = "└─"

# Box style for panels
PANEL_BOX: Box = ROUNDED

# Rich Theme mapping
PARALLAX_THEME = Theme(
    {
        # Brand & Roles
        "brand": "bold #818cf8",
        "brand.dim": "#6366f1",
        "user": "bold #38bdf8",
        # Status indicators
        "status.success": "#34d399",
        "status.error": "#f87171",
        "status.warning": "#fbbf24",
        "status.info": "#38bdf8",
        # Structural & Metadata
        "muted": "#64748b",
        "border": "#334155",
        "border.subtle": "#1e293b",
        "meta": "dim #94a3b8",
        "timestamp": "dim #64748b",
        # Tool execution
        "tool.name": "bold #e2e8f0",
        "tool.arg": "#94a3b8",
        "tool.bash": "#a7f3d0",
        "tool.editor": "#fed7aa",
        "tool.search": "#bae6fd",
        # Output highlights
        "diff.add": "#34d399",
        "diff.remove": "#f87171",
    }
)
