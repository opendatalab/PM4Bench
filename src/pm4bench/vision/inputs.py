"""Compatibility imports. New code uses data.benchmark and vision.tasks."""
from ..data.benchmark import image_map, load_manifest, sha256, text_digest
from .resources import FONTS, RESOURCE_ROOT
from .tasks.miqa import miqa_plan
from .tasks.msocr import msocr_plan

__all__ = [
           'FONTS',
           'RESOURCE_ROOT',
           'image_map',
           'load_manifest',
           'miqa_plan',
           'msocr_plan',
           'sha256',
           'text_digest',
]
