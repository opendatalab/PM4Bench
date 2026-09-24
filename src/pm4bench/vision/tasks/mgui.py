"""MGUI manifest alignment and page-deduplicated rendering."""
from __future__ import annotations

import json
import re
from pathlib import Path

from ...data.benchmark import sha256, text_digest
from ...io import resolve_asset
from ._mgui_browser import compare_gt, normalize_historical_text, render_mgui


def mgui_plan(row: dict, root: Path) -> dict:
    template, language = row['template'], row['language']
    if not re.fullmatch(r'[A-Za-z0-9_-]+', template):
        raise ValueError(f"{row['id']}: invalid template name")
    config_name = f'metadata/mgui/templates/{template}.json'
    source_name = f'metadata/mgui/templates/{template}.html.j2'
    gt_name = f'metadata/mgui/gt/{template}_{language}.json'
    config = json.loads(resolve_asset(root, config_name).read_text(encoding='utf-8'))
    gt = json.loads(resolve_asset(root, gt_name).read_text(encoding='utf-8'))
    number = int(row['id'].removeprefix(f'{template}_{language}_'))
    if not 1 <= number <= len(config['questions']):
        raise ValueError(f"{row['id']}: invalid question index")
    question = config['questions'][number-1]
    reference = gt['questions'][number-1]
    expected = {'question': question['prompts'][language], 'target_qid': question['target_qid'],
                'difficulty': int(question['difficulty']), 'template_title': config['title'],
                'image_size': [int(config.get('width', 1280)), int(config.get('height', 800))],
                'bbox': reference['bbox'], 'click_xy': reference['click_xy']}
    if any(row[field] != value for field, value in expected.items()):
        raise ValueError(f"{row['id']}: manifest/template/GT mismatch")
    if (gt['template'] != template or gt['lang'] != language
            or reference['target_qid'] != row['target_qid']
            or reference['question'] != row['question']
            or normalize_historical_text(reference['target_text_sample'])
            != normalize_historical_text(row['target_text'])):
        raise ValueError(f"{row['id']}: reference question/target mismatch")
    metadata = {name: sha256(resolve_asset(root, name)) for name in
                (config_name, source_name, gt_name)}
    return {'id': row['id'], 'task': 'mgui', 'language': language, 'template': template,
            'image_id': f'{template}_{language}', 'question': row['question'],
            'target_qid': row['target_qid'], 'target_text': row['target_text'],
            'question_sha256': text_digest(row['question']), 'reference_image': row['image'],
            'reference_gt': gt_name, 'input_metadata_sha256': metadata}


def render_batch(plans, root, output, options):
    groups = {}
    for plan in plans:
        groups.setdefault(plan['language'], set()).add(plan['template'])
    comparisons = []
    for language, templates in groups.items():
        counts = render_mgui(root / 'metadata/mgui/templates', output, languages=(language,),
                             browser_executable=options.browser_executable,
                             template_names=tuple(sorted(templates)), task_layout=True)
        comparison = compare_gt(
            output / 'gt/mgui' / language, root / 'metadata/mgui/gt',
            tolerance=options.bbox_tolerance,
            compare_geometry=options.comparison_mode == 'strict',
            filenames={f'{template}_{language}.json' for template in templates},
        )
        comparisons.append({'language': language, **comparison})
    artifacts = []
    hashes = {}
    for plan in plans:
        relative = f"images/mgui/{plan['language']}/{plan['image_id']}.png"
        if relative not in hashes:
            hashes[relative] = sha256(output / relative)
        artifacts.append({'id': plan['id'], 'task': 'mgui', 'language': plan['language'],
                          'image': relative, 'sha256': hashes[relative],
                          'gt': f"gt/mgui/{plan['language']}/{plan['image_id']}.json"})
    return artifacts, {'comparisons': comparisons, 'browser': counts['browser'],
                       'passed': all(item['passed'] for item in comparisons),
                       'font_profile': 'historical-system-stack'}
