import copy
import re

import pytest

from pm4bench.vision.mdur_text import parse_vision_text, visible_text


def row(question='Is x < 20 true? <image 1>', options=None):
    options = options or ['Yes', 'No']
    clean = lambda text: re.sub(r'<.*?>', '', text)
    return {
        'id': 'mdur_en_fixture', 'language': 'en',
        'question': clean(question), 'options': [clean(s) for s in options],
        'rendered_text': 'Question: ' + question + '\nOptions:\n' + '\n'.join(
            f'({chr(65+i)})  {s}' for i, s in enumerate(options)
        ),
    }


def test_transcript_preserves_comparison_text_and_does_not_modify_dataset():
    record = row()
    before = copy.deepcopy(record)
    assert record['question'] == 'Is x '
    parsed = parse_vision_text(record)
    assert parsed['question'] == 'Is x < 20 true? <image 1>'
    assert visible_text(parsed['question']) == 'Is x < 20 true? '
    assert record == before


def test_preserve_option_symbols_and_multiline_text():
    options = ['<L(a);l(b)>', 'First line\nsecond line']
    assert parse_vision_text(row(options=options))['options'] == options


def test_historical_edge_whitespace_is_checked_without_rewriting_transcript():
    record = row()
    record['question'] += '  '
    assert parse_vision_text(record)['question'] == 'Is x < 20 true? <image 1>'


@pytest.mark.parametrize('field', ['question', 'options', 'rendered_text'])
def test_reject_inconsistent_content(field):
    record = row()
    if field == 'options':
        record[field][0] = 'Changed'
    else:
        record[field] += 'Changed'
    with pytest.raises(ValueError):
        parse_vision_text(record)


def test_translated_markers_do_not_erase_math_comparisons():
    assert visible_text('x < 20 <obrázek 1> <слика 2>') == 'x < 20  '


def test_reject_missing_option_label():
    record = row()
    record['rendered_text'] = record['rendered_text'].replace('(B)  ', '(C)  ')
    with pytest.raises(ValueError, match='option separator'):
        parse_vision_text(record)
