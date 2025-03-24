import os
import json
from joblib import Parallel, delayed
from tqdm import tqdm
import re
import string


def find_first_failure_font_size(ocr_string, reference_list):
    _ocr_words = re.split(r'\s+', ocr_string.split('<start>')[-1].split('<Start>')[-1].split('<end>')[0].split('<End>')[0].strip(string.punctuation).strip())
    ocr_words = []
    for word in _ocr_words:
        if word:
            ocr_words.append(word)
    current_position = 0

    for ref_dict in reference_list:
        ref_line = ref_dict['text']
        ref_font_size = ref_dict['font_size']
        ref_words = ref_line.split()
        if current_position >= len(ocr_words):
            return ref_font_size
        for expected_word in ref_words:
            if current_position >= len(ocr_words):
                return ref_font_size + 2
            if ocr_words[current_position] != expected_word:
                return ref_font_size + 2
            current_position += 1
    return 2

def judge_MSOCR_ocr():
    src_root = r'VLM_output'
    dst_root = r'VLM_output_judge'
    ref_root = r'data/ref_answers/MSOCR'
    for model in os.listdir(src_root):
        if 'err' in model or 'origin' in model:
            continue
        model_dir = os.path.join(src_root, model, 'MSOCR', 'OCR')
        dst_dir = os.path.join(dst_root, model, 'MSOCR', 'OCR')
        os.makedirs(dst_dir, exist_ok=True)
        for file in os.listdir(model_dir):
            file_pth = os.path.join(model_dir, file)
            out_pth = os.path.join(dst_dir, file)
            lang = file.split('.')[-2].split('_')[-1].lower()
            ref_dir = os.path.join(ref_root, lang)
            out_file = open(out_pth, 'w', encoding='utf-8')
            for line in open(file_pth, 'r', encoding='utf-8').readlines():
                obj = json.loads(line)
                name = str(obj["index"]).rjust(3, '0')
                ref_meta = json.load(open(os.path.join(ref_dir, f'{lang}_{name}.json'), 'r', encoding='utf-8'))
                ref_info = ref_meta['lines']
                try:
                    score = find_first_failure_font_size(obj['response'], ref_info)
                except:
                    print(f"{lang} {obj['index']} error")
                    continue
                entry = {'index': obj['index'], 'score': score}
                out_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
judge_MSOCR_ocr()
