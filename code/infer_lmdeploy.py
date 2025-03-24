import re
import ast
import os
import json
from PIL import Image
from tqdm import tqdm
import sys
import json
import sys
from openai import OpenAI
import base64
from io import BytesIO
import csv
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from joblib import Parallel, delayed
from dotenv import load_dotenv

csv.field_size_limit(sys.maxsize)


### arguments
if len(sys.argv) == 9:
    MODEL = sys.argv[1] # model name, official model name recommended, such as [gpt-4o-2024-11-20, step-1o-vision-32k]
    MODE = sys.argv[2] # [cot, direct] for normal VLMs, use direct; for thinking VLMs, use cot
    SETTING = sys.argv[3] # [traditional, vision] setting
    LANGUAGE = sys.argv[4] # 10 languages choices, [ZH, EN, AR, SR, TH, RU, KO, CS, HU, VI]
    TASK = sys.argv[5] # [OCR, VQA]
    DATASET = sys.argv[6] # [MDUR, MIQA, MMJB, MSOCR]
    MAX_TOKENS = int(sys.argv[7]) # for different models, the max_tokens should be different in case of cut off problems
    PORT = sys.argv[8] # localhost port number
else:
    print("Usage: python code/infer_lmdeploy.py [MODEL] [MODE] [SETTING] [LANGUAGE] [TASK] [DATASET] [MAX_TOKENS] [PORT], your input is invalid, please check it again.")
    sys.exit(1)
API_KEY = 'sk-123'
BASE_URL = f'http://0.0.0.0:{PORT}/v1' # localhost
WORKERS = 32

DATASET_ROOT = 'data/tsv' # root dictionary of dataset
RESULTS_ROOT = 'VLM_output' # root dictionary of results
ERR_LOG_DIR = os.path.join(RESULTS_ROOT, 'err') # error log dictionary
os.makedirs(ERR_LOG_DIR, exist_ok=True)
''' number of samples in each dataset, used to check if the dataset is finished. '''
DATASET_NUM_SAMPLES = {
    'MIQA': 109,
    'MDUR': 1730,
    'MMJB': 500,
    'MSOCR': 100
}

# Load prompts from YAML file based on the task, language and mode etc.
import yaml
if TASK == 'OCR':
    with open(f"code/prompts/OCR/{DATASET}_prompts.yaml", "r") as file:
        prompt_config = yaml.safe_load(file)["languages"][LANGUAGE][MODE]
elif TASK == 'VQA':
    with open(f"code/prompts/VQA/{DATASET}_prompts.yaml", "r") as file:
        prompt_config = yaml.safe_load(file)["languages"][LANGUAGE][MODE]


def MDUR_doc_to_text(doc):
    question = doc["question"]
    options = ast.literal_eval(str(doc["options"]))
    option_letters = [chr(ord("A") + i) for i in range(len(options))]
    parsed_options = "\n".join([f"({option_letter}). {option}" for option_letter, option in zip(option_letters, options)])
    question = f"{question}\n\n{parsed_options}\n\n{prompt_config['traditional']}"

    for i in range(1, 15):
        images_token = f"<image {i}>"
        query_text = "<image>"
        if images_token in question:
            question = question.replace(images_token, query_text)

    return question


def load_model(model_name="GPT4", base_url="", api_key="", model="gpt-4-turbo-preview"):
    model_components = {}
    model_components['model_name'] = model_name
    model_components['model'] = model
    model_components['base_url'] = base_url
    model_components['api_key'] = api_key
    return model_components


def encode_pil_image(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


def send_request(prompt, max_tokens=128, base_url="", api_key=""):
    if not isinstance(prompt, list):
        print("WRONG PROMPT TYPE")
        exit(0)
    
    message = [{
        'role': 'user',
        'content': [{
            'type': 'text',
            'text': prompt[0],
        }]
    }]
    for base64_image in prompt[1:]:
        message[0]['content'].append({
            'type': 'image_url',
            'image_url': {'url': f'data:image/jpeg;base64,{base64_image}'}
        })

    client = OpenAI(base_url=base_url, api_key=api_key)
    model_name = client.models.list().data[0].id
    response = client.chat.completions.create(
        model=model_name,
        messages=message,
        temperature=0.1,
        top_p=0.001,
        max_tokens=max_tokens,
    )
    return response


def infer(prompts, max_tokens=4096, **kwargs):
    base_url = kwargs.get('base_url')
    api_key = kwargs.get('api_key')
    
    max_retries = 5
    retries = 0

    while retries < max_retries:
        try:
            response_tmp = send_request(prompts, max_tokens=max_tokens, base_url=base_url, api_key=api_key)
            if MODEL == 'kimi-k1.5-preview':
                stream = response_tmp
                for chunk in stream:
                    if chunk.choices[0].delta:
                        if chunk.choices[0].delta.content:
                            response = chunk.choices[0].delta.content
                            # print(chunk.choices[0].delta.content, end="")
            else:   
                response = response_tmp.choices[0].message.content
            break
        except Exception as e:
            retries += 1
            if retries >= max_retries:
                response = {"error": str(e)}
                break
    return response

def process_prompt(data, output_path, model_components):
    if DATASET == 'MMJB' and SETTING == 'traditional':
        prompts = [data['question']]
    elif DATASET == 'MMJB' and SETTING == 'vision':
        prompt = prompt_config['vision']
        image = data['image']
        prompts = [prompt, image]
    elif DATASET == 'MDUR' and SETTING == 'traditional':
        prompt = prompt_config['traditional'] + MDUR_doc_to_text(data)
        prompts = [prompt]
        for i in range(1, 15):
            if f'image_{i}' not in data or not data[f'image_{i}']:
                break
            prompts.append(data[f'image_{i}'])
    elif DATASET == 'MDUR' and SETTING == 'vision':    
        prompt = prompt_config['vision']
        image = data['image']
        prompts = [prompt, image]
    elif DATASET == 'MIQA' and SETTING == 'traditional':
        if MODE == 'cot':
            prompts = [prompt_config['traditional'] + data['question']]
        else:
            prompts = [data['question']]
        for i in range(1, 15):
            if f'image_{i}' not in data or not data[f'image_{i}']:
                break
            prompts.append(data[f'image_{i}'])
    elif DATASET == 'MIQA' and SETTING == 'vision':
        prompt = prompt_config['vision']
        image = data['image']
        prompts = [prompt, image]
    elif DATASET == 'MSOCR' and SETTING == 'vision':
        prompt = prompt_config['vision']
        image = data['image']
        prompts = [prompt, image]
    else:
        print("WRONG DATASET OR SETTING")
        exit(1)
    result = infer(prompts, max_tokens=MAX_TOKENS, **model_components)
    if isinstance(result, dict) or 'internal error happened' in result.lower():
        err_file = open(f"{ERR_LOG_DIR}/{TASK}_{DATASET}_{SETTING}_{LANGUAGE}_{MODEL}.txt", 'a')
        err_file.write(json.dumps({'response': result, 'index': str(data['index'])}, ensure_ascii=False) + '\n')
        err_file.close()
    else:
        out_file = open(output_path, 'a', encoding='utf-8')
        out_file.write(json.dumps({'response': result, 'index': str(data['index'])}, ensure_ascii=False) + '\n')
        out_file.close()


# main function
def run_and_save():
    dataset = f"{DATASET_ROOT}/{DATASET}_{SETTING}_{LANGUAGE}.tsv" # dataset root
    model_components = load_model(model_name=MODEL, base_url=BASE_URL, api_key=API_KEY, model=MODEL) # load model components
    
    print(f"Begin processing {DATASET}_{SETTING}_{MODE}_{TASK}_{LANGUAGE}")
    output_filefolder = f"{RESULTS_ROOT}/{MODEL}/{DATASET}/{TASK}/"
    os.makedirs(output_filefolder, exist_ok=True)
    output_path = os.path.join(output_filefolder, f"{SETTING}_{MODEL}_{LANGUAGE}.jsonl")
    
    # If results already exist, skip
    already = []
    if os.path.exists(output_path):
        lines = open(output_path, 'r', encoding='utf-8').readlines()
        if len(lines) == DATASET_NUM_SAMPLES[DATASET]:
            print(f"{DATASET}_{SETTING}_{LANGUAGE} already finished")
            return
        elif len(lines) > DATASET_NUM_SAMPLES[DATASET]:
            print(f"{DATASET}_{SETTING}_{LANGUAGE} result number error")
            return
        for line in lines:
            data = json.loads(line)
            already.append(data['index'])

    tsv_reader = csv.DictReader(open(dataset, mode='r', encoding='utf-8'), delimiter='\t')
    all_data = list(tsv_reader)
    all_data = [data for data in all_data if data['index'] not in already]
    Parallel(n_jobs=WORKERS)(delayed(process_prompt)(data, output_path, model_components) for data in tqdm(all_data))

run_and_save()
