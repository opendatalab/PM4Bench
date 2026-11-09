"""Packaged visual resources and per-language font filenames."""
from pathlib import Path

RESOURCE_ROOT = Path(__file__).parent / 'resources'
FONTS = {
    'ar': 'NotoSansArabic-Regular.ttf', 'cs': 'NotoSans-Regular.ttf',
    'en': 'NotoSans-Regular.ttf', 'hu': 'NotoSans-Regular.ttf',
    'ko': 'NotoSansKR-Regular.ttf', 'ru': 'NotoSans-Regular.ttf',
    'sr': 'NotoSans-Regular.ttf', 'th': 'NotoSansThai-Regular.ttf',
    'vi': 'NotoSans-Regular.ttf', 'zh': 'NotoSansSC-Regular.ttf',
}
