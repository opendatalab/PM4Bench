import os
import json
from joblib import Parallel, delayed
from tqdm import tqdm
import string
import re
import unicodedata
def proc_MIQA_ocr(entry, out_pth):
    resp = entry['ret'].replace('\n', ' ')
    text = entry['instruction']
    last_colon_newline = text.rfind(":\n")
    last_newline = text.rfind("\n")
    
    if last_colon_newline != -1 and last_newline != -1 and last_newline > last_colon_newline:
        extracted = text[last_colon_newline + 2 : last_newline]
        instruction = extracted.strip()
    else:
        instruction = ""
    resp = unicodedata.normalize('NFKC', resp)
    resp = re.sub(r'\s+', '', resp)
    instruction = unicodedata.normalize('NFKC', instruction)
    instruction = re.sub(r'\s+', '', instruction)
    if len(resp) <= 10:
        judge = 0
    elif instruction in resp:
        judge = 1
    else:
        judge = 0
    out_file = open(out_pth, 'a', encoding='utf-8')
    out_file.write(json.dumps({'index': entry['index'], 'judge': judge, 'resp': resp, 'answer': instruction}, ensure_ascii=False) + '\n')
    out_file.close()

def judge_MIQA_ocr():
    src_root = r'VLM_output'
    dst_root = r'VLM_output_judge'
    ref_dir = r'data/ref_answers/MIQA/OCR'
    entries = []
    for model in os.listdir(src_root):
        if 'err' in model:
            continue
        model_dir = os.path.join(src_root, model, 'MIQA', 'OCR')
        dst_dir = os.path.join(dst_root, model, 'MIQA', 'OCR')
        os.makedirs(dst_dir, exist_ok=True)
        for file in os.listdir(model_dir):
            file_pth = os.path.join(model_dir, file)
            out_pth = os.path.join(dst_dir, file)
            lang = file.split('.')[-2].split('_')[-1].lower()

            # load question
            ref_pth = os.path.join(ref_dir, 'final_text_{}.jsonl'.format(lang))
            ref_info = {}

            with open(ref_pth, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    ref_info[str(data['id'])] = data["text"]

            # load prediction
            resp_info = {}
            for line in open(file_pth, 'r', encoding='utf-8').readlines():
                obj = json.loads(line)
                resp_info[obj['index']] = obj['response']

            f = open(out_pth, 'w')
            f.close()
            for idx in resp_info.keys():
                entry = {'index': idx, 'ret': resp_info[idx], 'instruction': ref_info[idx], 'lang': lang, 'out_pth': out_pth}
                entries.append(entry)
    print('Total entries:', len(entries))
    # print(entries[0])
    Parallel(n_jobs=32)(delayed(proc_MIQA_ocr)(entry, entry['out_pth']) for entry in tqdm(entries))

judge_MIQA_ocr()
