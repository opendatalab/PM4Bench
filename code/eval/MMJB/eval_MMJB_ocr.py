import os
import json
from joblib import Parallel, delayed
from tqdm import tqdm
import string


def proc_MMJB_ocr(entry, out_pth):
    resp = entry['ret'].replace('\n', ' ')
    instruction = entry['instruction'].strip(string.punctuation).strip()
    resp = resp.split('<start>')[-1].split('<Start>')[-1].split('<end>')[0].split('<End>')[0].split('</end>')[0].strip().split('1')[0].strip().strip(string.punctuation).strip()
    if len(resp) <= 10 or len(instruction) <= 10:
        judge = 0
    elif resp == instruction:
        judge = 1
    else:
        judge = 0
    
    out_file = open(out_pth, 'a', encoding='utf-8')
    out_file.write(json.dumps({'index': entry['index'], 'judge': judge}, ensure_ascii=False) + '\n')
    out_file.close()


def judge_MMJB_ocr():
    src_root = r'VLM_output'
    dst_root = r'VLM_output_judge'
    ref_dir = r'data/ref_answers/MMJB/VQA'
    index_info = json.load(open(r'data/ref_answers/MMJB/image_index.json', 'r', encoding='utf-8'))    
    entries = []
    for model in os.listdir(src_root):
        if 'err' in model:
            continue
        model_dir = os.path.join(src_root, model, 'MMJB', 'OCR')
        dst_dir = os.path.join(dst_root, model, 'MMJB', 'OCR')
        os.makedirs(dst_dir, exist_ok=True)
        for file in os.listdir(model_dir):
            file_pth = os.path.join(model_dir, file)
            out_pth = os.path.join(dst_dir, file)
            lang = file.split('.')[-2].split('_')[-1].lower()
            ref_pth = os.path.join(ref_dir, '{}.jsonl'.format(lang))
            ref_info = {}
            for line in open(ref_pth, 'r', encoding='utf-8').readlines():
                obj = json.loads(line)
                ref_info[str(index_info[obj['id']])] = obj
            resp_info = {}
            for line in open(file_pth, 'r', encoding='utf-8').readlines():
                obj = json.loads(line)
                resp_info[obj['index']] = obj['response']
            f = open(out_pth, 'w')
            f.close()
            for idx in resp_info.keys():
                # if idx not in already:
                entry = {'index': idx, 'ret': resp_info[idx], 'instruction': ref_info[idx]['instruction'], 'lang': lang, 'out_pth': out_pth}
                entries.append(entry)

    Parallel(n_jobs=32)(delayed(proc_MMJB_ocr)(entry, entry['out_pth']) for entry in tqdm(entries))

judge_MMJB_ocr()
