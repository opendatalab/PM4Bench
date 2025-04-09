import os
import json
import re


def clean_string(s):
    if not s:
        return ""
    s = re.sub(r'[^\w\u0600-\u06FF\u0E00-\u0E7F]', '', s, flags=re.UNICODE)
    s = re.sub(r'<.*?>', '', s)
    return s


def find_first_failure_font_size(ocr_string, reference_list):
    ocr_string = clean_string(ocr_string)
    if not ocr_string:
        return 0
    ref_string = ""
    font_size_map = []
    
    for item in reference_list:
        text = clean_string(item["text"])
        ref_string += text
        for _ in text:
            font_size_map.append(item["font_size"])
    
    for i, (ocr_char, ref_char) in enumerate(zip(ocr_string, ref_string)):
        if ocr_char != ref_char:
            size = max(0, i-1)
            return 40 - font_size_map[size]
    
    if len(ocr_string) != len(ref_string):
        min_len = min(len(ocr_string), len(ref_string))
        if min_len < len(font_size_map):
            return 40 - font_size_map[min_len]
    
    return 40
    

def judge_MSOCR_ocr():
    src_root = r'VLM_output'
    dst_root = r'VLM_output_judge'
    ref_root = r'data/ref_answers/MSOCR'

    for model in os.listdir(src_root):
        if 'err' in model or 'origin' in model:
            continue
        model_dir = os.path.join(src_root, model, 'SizeBench', 'OCR')
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
                score = find_first_failure_font_size(obj['response'], ref_info)
                entry = {'index': obj['index'], 'score': score}
                out_file.write(json.dumps(entry, ensure_ascii=False) + '\n')

        
judge_MSOCR_ocr()
