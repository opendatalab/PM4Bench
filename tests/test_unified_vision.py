import json

import pytest

from pm4bench.cli import build_parser
from pm4bench.data import TASKS, select_records
from pm4bench.data.benchmark import RESOURCE_ROOT
from pm4bench.io import iter_jsonl
from pm4bench.vision import render_vision
from pm4bench.vision.tasks import TASK_SPECS, mgui


@pytest.mark.parametrize('task', TASKS)
def test_all_tasks_share_audit_contract(dataset, tmp_path, task):
    output = tmp_path / task
    report = render_vision(dataset, output, task, ('en',), audit_only=True,
                           limit=1, allow_custom_manifest=True)
    assert report['status'] == 'complete'
    assert report['records'] == 1
    assert report['images'] == report['rendered_records'] == 0
    assert report['backend'] == TASK_SPECS[task].backend
    assert set(report['input_manifests_sha256']) == {f'data/{task}/en.jsonl'}
    plan, = iter_jsonl(output / 'plans.jsonl')
    assert plan['task'] == task and plan['language'] == 'en'
    assert (output / 'images.jsonl').read_text() == ''
    assert json.loads((output / 'report.json').read_text()) == report


def test_registry_and_manifest_lock_cover_all_tasks():
    assert set(TASK_SPECS) == set(TASKS)
    lock = json.loads((RESOURCE_ROOT / 'inputs.lock.json').read_text())
    assert len(lock['manifests']) == 40
    assert {name.split('/')[1] for name in lock['manifests']} == set(TASKS)
    for task in TASKS:
        args = build_parser().parse_args(['render-vision', '--task', task,
                                         '--dataset-root', 'data', '--output-root', 'out'])
        assert args.task == task


def test_shared_selection_is_explicit(dataset):
    rows, _ = select_records(dataset, 'mgui', ('en',), ids=('example_en_2',), custom=True)
    assert [row['id'] for row in rows] == ['example_en_2']
    for options in ({'ids': ('missing',)}, {'limit': 0}, {'ids': ()}):
        with pytest.raises(ValueError):
            select_records(dataset, 'mgui', ('en',), custom=True, **options)
    with pytest.raises(ValueError, match='distinct'):
        select_records(dataset, 'mgui', ('en', 'en'), custom=True)


def test_mgui_metadata_must_match_public_record(dataset, tmp_path):
    path = dataset / 'metadata/mgui/templates/example.json'
    config = json.loads(path.read_text())
    config['questions'][0]['prompts']['en'] = 'Changed instruction'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='mismatch'):
        render_vision(dataset, tmp_path / 'out', 'mgui', ('en',), audit_only=True,
                      allow_custom_manifest=True)
    assert not (tmp_path / 'out').exists()


def test_mgui_deduplicates_pages_and_emits_one_artifact_per_question(dataset, tmp_path, monkeypatch):
    calls = []

    def fake_browser(templates, output, **options):
        calls.append(options)
        image = output / 'images/mgui/en/example_en.png'
        image.parent.mkdir(parents=True)
        image.write_bytes(b'fake-image')
        return {'browser': 'fixture'}

    monkeypatch.setattr(mgui, 'render_mgui', fake_browser)
    monkeypatch.setattr(mgui, 'compare_gt', lambda *a, **kw: {'passed': True})
    output = tmp_path / 'rendered'
    report = render_vision(dataset, output, 'mgui', ('en',), allow_custom_manifest=True)
    assert len(calls) == 1
    assert calls[0]['template_names'] == ('example',) and calls[0]['task_layout']
    assert report['images'] == 1 and report['rendered_records'] == report['records'] == 2
    artifacts = list(iter_jsonl(output / 'images.jsonl'))
    assert len({a['image'] for a in artifacts}) == len({a['sha256'] for a in artifacts}) == 1
    assert {a['id'] for a in artifacts} == {'example_en_1', 'example_en_2'}


def test_failed_render_preserves_report(dataset, tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError('fixture render failure')

    monkeypatch.setattr(mgui, 'render_batch', fail)
    output = tmp_path / 'failed'
    with pytest.raises(RuntimeError, match='fixture'):
        render_vision(dataset, output, 'mgui', ('en',), allow_custom_manifest=True)
    report = json.loads((output / 'report.json').read_text())
    assert report['status'] == 'failed' and report['error_type'] == 'RuntimeError'
    assert (output / 'plans.jsonl').is_file()
    with pytest.raises(FileExistsError):
        render_vision(dataset, output, 'mgui', ('en',), allow_custom_manifest=True)


@pytest.mark.parametrize('task,options', [
    ('mgui', {'fonts_root': 'fonts'}), ('mdur', {'workers': 2}),
    ('miqa', {'fit': False}), ('msocr', {'comparison_mode': 'strict'}),
])
def test_inapplicable_options_fail_before_writing(dataset, tmp_path, task, options):
    from pathlib import Path

    if 'fonts_root' in options:
        options = {**options, 'fonts_root': Path(options['fonts_root'])}
    with pytest.raises(ValueError):
        render_vision(dataset, tmp_path / 'no-output', task, ('en',), audit_only=True,
                      allow_custom_manifest=True, **options)
    assert not (tmp_path / 'no-output').exists()


def test_comparison_subset_and_missing_selected_reference(dataset, tmp_path):
    from pm4bench.mgui.render import compare_gt

    gt = dataset / 'metadata/mgui/gt'
    (gt / 'unselected.json').write_text('{}')
    candidate = tmp_path / 'candidate'
    candidate.mkdir()
    (candidate / 'example_en.json').write_bytes((gt / 'example_en.json').read_bytes())
    assert compare_gt(candidate, gt, filenames={'example_en.json'})['passed']
    assert not compare_gt(candidate, gt, filenames={'missing.json'})['passed']


def test_legacy_mgui_import_is_a_facade():
    from pm4bench.mgui.render import render_mgui
    from pm4bench.vision.tasks._mgui_browser import render_mgui as canonical

    assert render_mgui is canonical
