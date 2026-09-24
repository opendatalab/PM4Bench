"""Explicit renderer options shared by the task backends."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RenderOptions:
    fonts_root: Path | None = None
    browser_executable: Path | None = None
    segoe_root: Path | None = None
    fit: bool = True
    styles: Path | None = None
    strict_glyphs: bool = False
    workers: int = 1
    comparison_mode: str = 'structure'
    bbox_tolerance: float = .2
    legacy_layout: bool = False
