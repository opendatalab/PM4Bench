"""Compatibility facade; canonical implementation lives in vision.tasks."""
from ..vision.tasks._mgui_browser import (
    FONT_STACK,
    LANGUAGES,
    compare_gt,
    normalize_historical_text,
    render_mgui,
)

__all__ = ['FONT_STACK', 'LANGUAGES', 'compare_gt', 'normalize_historical_text', 'render_mgui']
