"""Compatibility entry point. Prefer pm4bench render-vision --task mdur."""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

from ..data import LANGUAGES
from .render import render_vision
from .tasks.mdur import mdur_plan, safe_text, validate_style

__all__ = ['mdur_plan', 'render_mdur', 'safe_text', 'validate_style']


def render_mdur(dataset_root, output_root, languages=LANGUAGES, *, fonts_root=None,
                segoe_root=None, ids=None, limit=None, audit_only=False, fit=True,
                styles=None, browser_executable=None):
    """Retain the 2.2 signature/image layout through the shared orchestrator."""
    return render_vision(dataset_root, output_root, 'mdur', languages,
                         fonts_root=fonts_root, segoe_root=segoe_root, ids=ids,
                         limit=limit, audit_only=audit_only, fit=fit, styles=styles,
                         browser_executable=browser_executable, _legacy_layout=True)


def main():
    warnings.warn('Use pm4bench render-vision --task mdur; this entry point is retained '
                  'for compatibility.', FutureWarning, stacklevel=1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--fonts-root', type=Path)
    parser.add_argument('--segoe-root', type=Path)
    parser.add_argument('--language', action='append', choices=LANGUAGES)
    parser.add_argument('--id', action='append')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--no-fit', action='store_true')
    parser.add_argument('--styles', type=Path)
    parser.add_argument('--browser-executable', type=Path)
    args = parser.parse_args()
    print(json.dumps(render_mdur(
        args.dataset_root, args.output_root, tuple(args.language or LANGUAGES),
        fonts_root=args.fonts_root, segoe_root=args.segoe_root,
        ids=tuple(args.id) if args.id else None, limit=args.limit, audit_only=args.audit_only,
        fit=not args.no_fit, styles=args.styles, browser_executable=args.browser_executable), indent=2))


if __name__ == '__main__':
    main()
