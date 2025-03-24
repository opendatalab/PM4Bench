from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import re
import ast
import json
from joblib import Parallel, delayed
import requests
from tqdm import tqdm
import uuid
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
    max_num_retries=3,
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
            print(f"Error: {e}", flush=True)
            
    return None


def proc_MIQA_vqa(lang, entry, out_pth):

    resp = entry['ret']
    meta_prompt = """
        You are an assistant skilled at evaluating the quality of creative text.
        Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. You'll need to assess the response on the following dimensions: Creativity, Richness, Visual Perception, Logical Coherence, Answer Accuracy and Image Relationship Understanding. We will provide you with a creative question and the AI model's response and a reference answer for your evaluation. As you begin your assessment, follow this process:
        1. Evaluate the AI model's answers on different dimensions, pointing out its strengths or weaknesses in each dimension and assigning a score of 1 to 10 for each.
        2. Finally, based on the assessments across dimensions, provide an overall score of 1 to 10 for the AI model's response.
        3. Your scoring should be as stringent as possible and follow the scoring rules below:

        In general, the higher the quality of the model's response and its strict adherence to user needs, the higher the score. Responses that do not meet user needs will receive lower scores.

        Scoring rules:
        Creativity:
        Scores 1-2 when there is no innovation or uniqueness in the content.
        Scores 3-4 when providing partially original content but with low creative quality.
        Scores 5-6 when mostly creative but lacks significant novelty, with moderate quality.
        Scores 7-8 when having novelty and high-quality content.
        Scores 9-10 when highly novel and of exceptional quality compared to the reference answer.

        Richness:
        Scores 1-2 when lacking depth and breadth, with very limited information.
        Scores 3-4 when limited in depth and breadth, with fewer explanations and examples, showing low diversity.
        Scores 5-6 when limited in depth and breadth but provides basic necessary information.
        Scores 7-8 when providing depth and useful additional information.
        Scores 9-10 when providing exceptional depth, breadth, and high diversity compared to the reference answer.

        Visual Perception:
        Scores 1-2 when the description of the visual information in the image contains errors or is significantly inconsistent with the content of the image.
        Scores 3-4 When the description of the visual information in the image reflects only a small amount of the image's information and contains some errors.
        Scores 5-6 when the description of the visual information in the image includes the basic information of the image but contains minimal information.
        Scores 7-8 when the description of the visual information in the image matches the image well and is rich in content, providing a substantial amount of information about the image.
        Scores 9-10 when the description of the visual information in the image not only matches the image but also is more detailed and informative compared to the reference answer, providing more information about the image.

        Logical Coherence:
        Scores 1-2 when entirely incoherent, lacking any logic, and not matching the question or known information.
        Scores 3-4 when somewhat coherent but with many logical errors or inconsistencies.
        Scores 5-6 when mostly coherent, with few errors, but may struggle to maintain complete coherence in complex situations.
        Scores 7-8 when excellent logical handling, very few errors.
        Scores 9-10 when flawless logic, impeccable in handling complexity, and significantly higher logical coherence compared to the reference answer.

        Answer Accuracy
        Scores 1-2 when the answer is significantly inconsistent with the question or contains obvious errors.
        Scores 3-4 when the answer is partially correct but contains some errors or is incomplete.
        Scores 5-6 when the answer is basically correct but lacks details or is not sufficiently detailed.
        Scores 7-8 when the answer is accurate and detailed, fully corresponding to the question.
        Scores 9-10 when the answer is not only accurate and detailed but also provides additional useful information, exceeding expectations.

        Image Relationship Understanding:
        Scores 1-2 when there are significant errors or confusion in distinguishing and describing different images, unable to correctly identify and relate the content of the images.
        Scores 3-4 when the description of different images reflects only minimal distinguishing information, contains some errors and confusion, and fails to clearly differentiate and relate the images.
        Scores 5-6 when the description of different images includes basic distinguishing information, is able to correctly identify and relate the images in a basic manner, but the information provided is minimal and lacks detail.
        Scores 7-8 when the description of different images is accurate and detailed, clearly distinguishing and relating the images, with rich content that points out the main commonalities and differences between the images.
        Scores 9-10 when the description of different images is not only accurate and detailed but also provides richer information and analysis, clearly distinguishing and relating the images, more comprehensively pointing out the commonalities and differences between the images compared to the reference answer.

        Overall Score:
        Scores 1-2 when irrelevant to the question, factually incorrect, or generates harmful content.
        Scores 3-4 when no serious errors, mostly harmless, but of low quality and does not meet requirements.
        Scores 5-6 when basically meeting requirements but performing poorly in some dimensions, with moderate quality.
        Scores 7-8 when performing well in all dimensions.
        Scores 9-10 when fully addressing user questions and all requirements, significantly surpassing the reference answer.

        Please remember, you must evaluate and explain before scoring. After your explanation for each dimension, add the score for that dimension. Finally, at the end of your response, in the format of the dictionary (including brackets), return all your scoring results, ensuring your scores are integers:
        {'Dimension One': Score, 'Dimension Two': Score, ..., 'Overall Score': Score}, for example: {'Creativity': 9, 'Richness': 6, ..., 'Overall Score': 7}.\n
    """
    question_begin_prompt = "[Question]"
    reference_begin_prompt = "[The Start of Reference Answer]"
    reference_end_prompt = "[The End of Reference Answer]"
    answers_begin_prompt = "[The Start of Assistant’s Answer]"
    answers_end_prompt = "[The End of Assistant’s Answer]"
    prompt = (
        meta_prompt +
        question_begin_prompt + '\n' + entry['instruction'][0]['value'] + '\n\n' +
        reference_begin_prompt + '\n' + entry['instruction'][1]['value'] + '\n' +
        reference_end_prompt + '\n\n' +
        answers_begin_prompt + '\n' + resp + '\n' +
        answers_end_prompt
    )
    judge = gpt_chat(prompt)
    if judge:
        with open(out_pth, 'a', encoding='utf-8') as out_file:
            out_file.write(json.dumps({'index': entry['index'], 'judge': judge}, ensure_ascii=False) + '\n')

WORKERS = 500
langs = ['en', 'zh', 'hu', 'ru', 'sr', 'cs', 'ar', 'vi', 'th', 'ko']

def judge_MIQA_vqa():
    src_root = r'VLM_output'
    dst_root = r'VLM_output_judge'
    ref_dir = r'data/ref_answers/MIQA/VQA'
    entries = []
    for model in os.listdir(src_root):
        if 'err' in model:  # err is not a model name
            continue
        model_dir = os.path.join(src_root, model, 'MIQA', 'VQA')
        dst_dir = os.path.join(dst_root, model, 'MIQA', 'VQA')
        os.makedirs(dst_dir, exist_ok=True)
        for file in os.listdir(model_dir):
            file_pth = os.path.join(model_dir, file)
            out_pth = os.path.join(dst_dir, file)
            lang = file.split('.')[-2].split('_')[-1].lower()

            # load reference data
            ref_pth = os.path.join(ref_dir, '{}.json'.format(lang))
            ref_info = {}
            with open(ref_pth, 'r', encoding='utf-8') as f:
                ref_data = json.load(f)
            for item in ref_data:
                ref_info[str(item['id'])] = item

            # load prediction data
            resp_info = {}
            with open(file_pth, 'r', encoding='utf-8') as f:
                for line in f:
                    obj = json.loads(line)
                    resp_info[obj['index']] = obj['response']
            already = []
            if os.path.exists(out_pth):
                with open(out_pth, 'r', encoding='utf-8') as f:
                    for line in f:
                        obj = json.loads(line)
                        already.append(obj['index'])
            for idx in resp_info.keys():
                if idx not in already:
                    entry = {
                        'index': idx,
                        'ret': resp_info[idx],
                        'instruction': ref_info[idx]['conversations'],
                        'lang': lang,
                        'out_pth': out_pth
                    }
                    entries.append(entry)
    Parallel(n_jobs=WORKERS)(
        delayed(proc_MIQA_vqa)(entry['lang'], entry, entry['out_pth']) for entry in tqdm(entries)
    )

def re_run_judge(idx, model, task, setting, file):
    """
    rerun judge for predictions which cannot extract score
    """
    lang = file.split('.')[-2].split('_')[-1].lower()
    original_src = r'VLM_output'
    original_file = os.path.join(original_src, model, task, setting, file)
    
    # load reference data
    ref_dir = r'data/ref_answers/MIQA/VQA'
    ref_pth = os.path.join(ref_dir, f'{lang}.json')
    ref_info = {}
    with open(ref_pth, 'r', encoding='utf-8') as f:
        ref_data = json.load(f)
    for item in ref_data:
        ref_info[str(item['id'])] = item

    # load prediction data
    response = None
    with open(original_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception as e:
                continue
            if obj.get('index') == idx:
                response = obj.get('response')
                break
    if response is None:
        print(f"Cannot find response for index {idx} in file {original_file}")
        return None

    if idx not in ref_info:
        print(f"Cannot find reference info for index {idx} in file {ref_pth}")
        return None

    judge_file = os.path.join(r'VLM_output_judge', model, task, setting, file)
    os.makedirs(os.path.dirname(judge_file), exist_ok=True)
    entry = {
        'index': idx,
        'ret': response,
        'instruction': ref_info[idx]['conversations'],
        'lang': lang,
        'out_pth': judge_file
    }
    print(f"Rerun judge: model={model}, task={task}, setting={setting}, file={file}, index={idx}")
    proc_MIQA_vqa(lang, entry, entry['out_pth'])
    new_obj = None
    with open(entry['out_pth'], 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in reversed(lines):
        try:
            obj = json.loads(line)
            if obj.get('index') == idx:
                new_obj = obj
                break
        except Exception as e:
            continue
    return new_obj

def extract_score():
    src_root = r'VLM_output_judge'
    dst_root = r'MIQA_extracted_score'
    os.makedirs(dst_root, exist_ok=True)
    for model in os.listdir(src_root):
        if 'err' in model or 'temp' in model:
            continue
        model_dir = os.path.join(src_root, model)
        dst_model_dir = os.path.join(dst_root, model)
        os.makedirs(dst_model_dir, exist_ok=True)
        task = 'MIQA'
        task_dir = os.path.join(model_dir, task)
        setting = 'VQA'
        setting_dir = os.path.join(task_dir, setting)
        if not os.path.exists(setting_dir):
            continue
        dst_setting_dir = os.path.join(dst_model_dir, task, setting)
        os.makedirs(dst_setting_dir, exist_ok=True)
        for file in os.listdir(setting_dir):
            if 'DS_Store' in file:
                continue
            file_pth = os.path.join(setting_dir, file)
            out_pth = os.path.join(dst_setting_dir, file)
            with open(out_pth, 'w', encoding='utf-8') as out_file:
                with open(file_pth, 'r', encoding='utf-8') as f_in:
                    for line in f_in:
                        try:
                            obj = json.loads(line)
                        except Exception as e:
                            print(f"Extract score failed: {e}, file: {file_pth}")
                            continue
                        idx = obj.get('index')
                        judge_text = obj.get('judge', '')
                        # use regex to extract score
                        pattern = r"(\{.*?\})"
                        match = re.search(pattern, judge_text, flags=re.DOTALL)
                        if match:
                            dict_str = match.group(1)
                            try:
                                result_dict = ast.literal_eval(dict_str)
                            except Exception as e:
                                print(f"Transform score string to dict failed: {e}, file: {file_pth}, index: {idx}")
                                result_dict = {}
                            out_file.write(json.dumps({'index': idx, 'score': result_dict}, ensure_ascii=False) + '\n')
                        else:
                            print(f"Cannot match score dict, file: {file_pth}, index: {idx}, try to rerun judge...")
                            new_obj = re_run_judge(idx, model, task, setting, file)
                            if new_obj:
                                new_judge_text = new_obj.get('judge', '')
                                match_new = re.search(pattern, new_judge_text, flags=re.DOTALL)
                                if match_new:
                                    dict_str_new = match_new.group(1)
                                    try:
                                        result_dict = ast.literal_eval(dict_str_new)
                                    except Exception as e:
                                        print(f"Transform score string to dict failed after reruning: {e}, file: {file_pth}, index: {idx}")
                                        result_dict = {}
                                    out_file.write(json.dumps({'index': idx, 'score': result_dict}, ensure_ascii=False) + '\n')
                                    print(f"Rerun judge success", json.dumps(result_dict, ensure_ascii=False))
                                else:
                                    print(f"Cannot match score dict after reruning, file: {file_pth}, index: {idx}")
                            else:
                                print(f"Rerun judge failed, file: {file_pth}, index: {idx}")                             
if __name__ == '__main__':
    judge_MIQA_vqa()
    extract_score()
