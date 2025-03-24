from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import json
from joblib import Parallel, delayed
from tqdm import tqdm
import random
random.seed(42)


env_file = ".env"
load_dotenv(env_file)

def gpt_chat(
    prompt,
    *,
    image=None,
    system_prompt=None,
    model_name="gpt-4o-2024-11-20",
    temperature=0.1,
    max_tokens=16384,
    max_num_retries=5,
    top_p=0.001,
):

    OPENAI_API_BASE = "https://api.openai.com/v1"
    client = OpenAI(
        api_key = os.getenv('gpt-4o-2024-11-20'),
        base_url = OPENAI_API_BASE
    )
    
    if system_prompt is not None:
        messages = [{
            "role": "system",
            "content": system_prompt,
        }, {
            "role": "user",
            "content": prompt,
        }]
    else:
        messages = [{
            "role": "user",
            "content": prompt,
        }]
    if image:
        messages[-1]['content'] = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}
        ]

    retry = 0
    while retry < max_num_retries:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            content = completion.choices[0].message.content
            return content

        except Exception as e:
            retry += 1
            if retry >= max_num_retries:
                print(f"Error: {e}", flush=True)
            
    return None


def proc_MDUR_vqa(lang, entry, out_pth):
    resp = entry['ret']
    meta_prompt = """
        You are a judgmental evaluation model. Now I will give you the correct answer to a multiple-choice question and the predicted answer from a language model (LLM). Please carefully read the predicted answer and then decide whether it matches the correct answer option.
        Correct answer: {answer}
        Model prediction: {response}

        If you believe the model's predicted answer matches the correct answer, return: hit:1  
        If you believe the model's predicted answer does not match the correct answer, return: hit:0

        Please return only hit:0 or hit:1, without any extra content.
    """ 
    prompt = meta_prompt.format(answer=entry['instruction'], response=resp)
    judge = gpt_chat(prompt)
    if judge:
        out_file = open(out_pth, 'a', encoding='utf-8')
        out_file.write(json.dumps({'index': entry['index'], 'judge': judge, 'answer': entry['instruction'], 'prediction': resp}, ensure_ascii=False) + '\n')
        out_file.close()

WORKERS = 500
langs = ['en', 'zh', 'hu', 'ru', 'sr', 'cs', 'ar', 'vi', 'th', 'ko']
def judge_MDUR_vqa():
    src_root = r'VLM_output'
    dst_root = r'VLM_output_judge'
    ref_dir = r'data/ref_answers/MDUR/VQA'
    entries = []
    for model in os.listdir(src_root):
        if 'err' in model: # err is not a model name
            continue
        model_dir = os.path.join(src_root, model, 'MDUR', 'VQA')
        dst_dir = os.path.join(dst_root, model, 'MDUR', 'VQA')
        os.makedirs(dst_dir, exist_ok=True)
        for file in os.listdir(model_dir):
            file_pth = os.path.join(model_dir, file)
            out_pth = os.path.join(dst_dir, file)
            lang = file.split('.')[-2].split('_')[-1].lower()
            ref_pth = os.path.join(ref_dir, 'MDUR_ref_{}.json'.format(lang))
            with open(ref_pth, 'r', encoding='utf-8') as f:
                ref_data = json.load(f)
            resp_info = {}
            for line in open(file_pth, 'r', encoding='utf-8').readlines():
                obj = json.loads(line)
                resp_info[obj['index']] = obj['response']
            already = []
            if os.path.exists(out_pth):
                for line in open(out_pth, 'r', encoding='utf-8').readlines():
                    obj = json.loads(line)
                    already.append(obj['index'])
            for idx in resp_info.keys():
                if idx not in already:
                    entry = {'index': idx, 'ret': resp_info[idx], 'instruction': ref_data[str(idx)], 'lang': lang, 'out_pth': out_pth}
                    entries.append(entry)
    Parallel(n_jobs=WORKERS)(delayed(proc_MDUR_vqa)(entry['lang'], entry, entry['out_pth']) for entry in tqdm(entries))

judge_MDUR_vqa()
