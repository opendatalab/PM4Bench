"""Pack generated OCR image/GT pairs into portable veRL Parquet shards."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ..data.benchmark import sha256
from ..vision.browser import new_output, save_json

PROMPT = (r'<image> Please identify the text in the image and output each word in sequence, '
          r'separated by spaces, without line breaks in \boxed{}. Please follow these rules: '
          r'1. For Arabic text lines, read from right to left. For other languages, read normally '
          r'(left to right). 2. Ignore any background text, webpage UI elements, or irrelevant '
          r'English characters that are not part of the main foreground content.')


def split_files(names: list[str], validation_size: int, seed: int) -> dict:
    if not names or len(names) != len(set(names)):
        raise ValueError('No images or duplicate image names')
    if not 0 <= validation_size < len(names):
        raise ValueError('Validation size must leave at least one training sample')
    ordered = sorted(names)
    random.Random(seed).shuffle(ordered)
    return {'test': ordered[:validation_size], 'train': ordered[validation_size:]}


def load_split(path: Path, names: list[str]) -> dict:
    split = json.loads(path.read_text(encoding='utf-8'))
    if set(split) != {'train', 'test'}:
        raise ValueError('Split manifest must have exactly train and test lists')
    all_names = split['train'] + split['test']
    if len(all_names) != len(set(all_names)) or set(all_names) != set(names):
        raise ValueError('Split manifest must cover each input image exactly once')
    return split


def row_for(image: Path, gt: dict, split: str) -> dict:
    if gt.get('source') == 'gui_v2':
        raise ValueError('This recipe is the OCR-only QGO corpus, not the later GUI mixture')
    target = gt['gt_text']
    if not target or not isinstance(target, str):
        raise ValueError('Missing OCR target')
    if 'word_boxes' in gt and ' '.join(b['text'] for b in gt['word_boxes']) != target:
        raise ValueError(f'{image.name}: word boxes and OCR target disagree')
    from PIL import Image

    with Image.open(image) as decoded:
        decoded.verify()
    return {'data_source': 'ocr_custom', 'prompt': [{'role': 'user', 'content': PROMPT}],
            'images': [{'bytes': image.read_bytes(), 'path': image.name}], 'ability': 'ocr',
            'reward_model': {'style': 'rule', 'ground_truth': target},
            'extra_info': {'split': split, 'index': -1, 'answer': target, 'image_path': image.name}}


def pack(source: Path, output: Path, *, validation_size: int = 500, seed: int = 42,
         shard_size: int = 500, split_manifest: Path | None = None) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if shard_size < 1:
        raise ValueError('Shard size must be positive')
    source = source.resolve()
    names = [p.name for p in (source / 'img').glob('*.png')]
    split = (load_split(split_manifest, names) if split_manifest else
             split_files(names, validation_size, seed))
    for name in names:
        if not (source / 'gt' / Path(name).with_suffix('.json')).is_file():
            raise FileNotFoundError(f'Missing ground truth: {name}')
    output = new_output(output, source)
    save_json(output / 'split.json', split)
    shards = []
    # Explicit schema retains the training data types even for small fixtures.
    string = pa.string()
    schema = pa.schema([
        ('data_source', string),
        ('prompt', pa.list_(pa.struct([('role', string), ('content', string)]))),
        ('images', pa.list_(pa.struct([('bytes', pa.binary()), ('path', string)]))),
        ('ability', string),
        ('reward_model', pa.struct([('style', string), ('ground_truth', string)])),
        ('extra_info', pa.struct([('split', string), ('index', pa.int64()),
                                  ('answer', string), ('image_path', string)])),
    ])
    value = {'dtype': 'string', '_type': 'Value'}
    features = {
        'data_source': value, 'prompt': [{'role': value, 'content': value}],
        'images': {'feature': {'_type': 'Image'}, 'length': -1, '_type': 'Sequence'},
        'ability': value, 'reward_model': {'style': value, 'ground_truth': value},
        'extra_info': {'split': value, 'index': {'dtype': 'int64', '_type': 'Value'},
                       'answer': value, 'image_path': value},
    }
    schema = schema.with_metadata({
        b'huggingface': json.dumps({'info': {'features': features}}).encode('utf-8')})
    for label, files in split.items():
        folder = output / 'data' / label
        folder.mkdir(parents=True)
        for offset in range(0, len(files), shard_size):
            rows = []
            for name in files[offset:offset + shard_size]:
                gt = json.loads((source / 'gt' / Path(name).with_suffix('.json')).read_text())
                rows.append(row_for(source / 'img' / name, gt, label))
            destination = folder / f'{label}_part_{offset // shard_size:04d}.parquet'
            pq.write_table(pa.Table.from_pylist(rows, schema=schema), destination)
            shards.append({'path': str(destination.relative_to(output)),
                           'rows': len(rows), 'sha256': sha256(destination)})
    report = {'train': len(split['train']), 'validation': len(split['test']),
              'seed': seed, 'split_sha256': sha256(output / 'split.json'),
              'shards': shards, 'pyarrow': pa.__version__}
    save_json(output / 'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--validation-size', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--shard-size', type=int, default=500)
    parser.add_argument('--split-manifest', type=Path)
    args = parser.parse_args()
    print(json.dumps(pack(args.input_root, args.output_root,
                          validation_size=args.validation_size, seed=args.seed,
                          shard_size=args.shard_size, split_manifest=args.split_manifest), indent=2))


if __name__ == '__main__':
    main()
