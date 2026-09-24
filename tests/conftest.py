import json

import pytest

from pm4bench.io import write_jsonl
from pm4bench.vision.tasks import mdur


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    root = tmp_path / 'dataset'
    root.mkdir()
    image = root / 'image_1.png'
    image.touch()
    monkeypatch.setattr(mdur, 'routing_metadata', lambda: {
        'en': {'fixture': {'question_images': [1], 'option_images': [[], []],
                           'explanation_images': []}}})
    rows = {
        'mdur': [{'id': 'mdur_en_0', 'source_id': 'fixture', 'language': 'en',
                  'question': 'Why?', 'options': ['Yes', 'No'],
                  'rendered_text': 'Question: Why?\nOptions:\n(A)  Yes\n(B)  No',
                  'traditional_images': ['image_1.png'], 'vision_image': 'reference.png'}],
        'miqa': [{'id': '0_0', 'language': 'en', 'question': 'What is it?',
                  'rendered_text': 'Image 1:\nWhat is it?\n',
                  'traditional_images': ['image_1.png']}],
        'msocr': [{'id': 'en_001', 'language': 'en', 'image_size': [1280, 720],
                   'font_gradient': list(range(40, 0, -2)),
                   'lines': [{'font_size': n, 'text': f'Line {n}'} for n in range(40, 0, -2)],
                   'ground_truth': ' '.join(f'Line {n}' for n in range(40, 0, -2))}],
        'mgui': [],
    }
    template = root / 'metadata/mgui/templates'
    gt = root / 'metadata/mgui/gt'
    template.mkdir(parents=True)
    gt.mkdir()
    questions = [{'target_qid': 'button', 'difficulty': 1,
                  'prompts': {'en': f'Click Go {i}'}} for i in (1, 2)]
    config = {'title': 'Example', 'questions': questions, 'strings': {'go': {'en': 'Go'}}}
    reference = {'template': 'example', 'template_title': 'Example', 'lang': 'en',
                 'image': 'example_en.png', 'width': 1280, 'height': 800,
                 'all_qid_boxes': {'button': {'x': 8, 'y': 8, 'width': 30, 'height': 20,
                                               'text': 'Go'}}, 'questions': []}
    for i, question in enumerate(questions, 1):
        bbox = {'x': 8, 'y': 8, 'width': 30, 'height': 20}
        reference['questions'].append({
            'qid_in_template': i, 'difficulty': 1, 'target_qid': 'button',
            'target_text_sample': 'Go', 'question': question['prompts']['en'],
            'bbox': bbox, 'click_xy': [23, 18],
        })
        rows['mgui'].append({
            'id': f'example_en_{i}', 'language': 'en', 'template': 'example',
            'question': question['prompts']['en'], 'target_qid': 'button',
            'target_text': 'Go', 'difficulty': 1, 'template_title': 'Example',
            'image_size': [1280, 800], 'bbox': bbox, 'click_xy': [23, 18],
            'image': 'assets/mgui/example_en.png',
        })
    (template / 'example.json').write_text(json.dumps(config))
    (template / 'example.html.j2').write_text(
        '<!doctype html><html><body><button data-qid="button">{{ t.go }}</button></body></html>')
    (gt / 'example_en.json').write_text(json.dumps(reference))
    for task, records in rows.items():
        for row in records:
            row['task'] = task
        write_jsonl(root / f'data/{task}/en.jsonl', records)
    return root
