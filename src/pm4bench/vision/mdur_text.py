"""Recover vision text exclusively from the released OCR transcript.

The traditional question/options fields underwent historical angle-bracket
cleanup. They are validated, never silently used to truncate vision text.
"""
from __future__ import annotations

import re

QUESTION_LABELS = {
    "ar": "سؤال", "cs": "otázka", "en": "Question", "hu": "kérdés", "ko": "질문",
    "ru": "Вопрос", "sr": "питање", "th": "คำถาม", "vi": "câu hỏi", "zh": "题目",
}
OPTION_LABELS = {
    "ar": "خيارات", "cs": "Možnosti", "en": "Options", "hu": "Opciók", "ko": "옵션",
    "ru": "Параметры", "sr": "Опције", "th": "ตัวเลือก", "vi": "Tùy chọn", "zh": "选项",
}
MARKERS = re.compile(r'<(?:image|obrázek|obrázku|obraz|слика|kép|изображении|صورة) (\d+)>')


def parse_vision_text(row: dict) -> dict:
    language = row['language']
    prefix = QUESTION_LABELS[language] + ': '
    separator = '\n' + OPTION_LABELS[language] + ':\n'
    transcript = row['rendered_text']
    if not transcript.startswith(prefix) or transcript.count(separator) != 1:
        raise ValueError(f"{row['id']}: unexpected vision transcript labels")
    question, option_text = transcript[len(prefix):].split(separator)
    options = []
    rest = option_text
    for index in range(len(row['options'])):
        label = f'({chr(65 + index)})  '
        if not rest.startswith(label):
            raise ValueError(f"{row['id']}: unexpected option label {label!r}")
        rest = rest[len(label):]
        if index + 1 == len(row['options']):
            options.append(rest)
            rest = ''
        else:
            separator_next = f'\n({chr(66 + index)})  '
            if rest.count(separator_next) != 1:
                raise ValueError(f"{row['id']}: ambiguous or missing option separator")
            current, following = rest.split(separator_next)
            options.append(current)
            rest = f'({chr(66 + index)})  ' + following
    if not options or rest:
        raise ValueError(f"{row['id']}: invalid option transcript")
    clean = lambda text: re.sub(r'<.*?>', '', text)
    if clean(question).strip() != row['question'].strip() or [clean(x).strip() for x in options] != [x.strip() for x in row['options']]:
        raise ValueError(f"{row['id']}: OCR and traditional fields violate historical cleanup")
    rebuilt = prefix + question + separator + '\n'.join(
        f'({chr(65+i)})  {value}' for i, value in enumerate(options)
    )
    if rebuilt != transcript:
        raise ValueError(f"{row['id']}: transcript round-trip changed text")
    return {'question': question, 'options': options,
            'question_label': QUESTION_LABELS[language], 'option_label': OPTION_LABELS[language]}


def visible_text(text: str) -> str:
    """Remove recognized image placeholders only; keep inequalities and TeX literal."""
    return MARKERS.sub('', text)
