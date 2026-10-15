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
        "brand": "bold #e4e4e7",
        "brand.dim": "#a1a1aa",
        "user": "bold #f4f4f5",
        # Status indicators
        "status.success": "#22c55e",
        "status.error": "#ef4444",
        "status.warning": "#f59e0b",
        "status.info": "#a1a1aa",
        # Structural & Metadata
        "muted": "#71717a",
        "border": "#3f3f46",
        "border.subtle": "#27272a",
        "meta": "dim #a1a1aa",
        "timestamp": "dim #71717a",
        # Tool execution
        "tool.name": "bold #f4f4f5",
        "tool.arg": "#a1a1aa",
        "tool.bash": "#d4d4d8",
        "tool.editor": "#d4d4d8",
        "tool.search": "#d4d4d8",
        # Output highlights
        "diff.add": "#22c55e",
        "diff.remove": "#ef4444",
    }
)
