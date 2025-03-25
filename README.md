# PM<sup>4</sup>Bench: A Parallel Multilingual Multi-Modal Multi-task Benchmark for Large Vision Language Model
---

[Junyuan Gao*](), [Jiahe Song*](https://jiahe-song.webflow.io/), [Jiang Wu*†](), Runchuan Zhu, Guanlin Shen, Shasha Wang, Xingjian Wei, Haote Yang, Songyang Zhang, Weijia Li, Bin Wang, Dahua Lin, Lijun Wu, Conghui He‡
```
*Equal contribution.
†Project lead.
‡Corresponding author.
```
---

<!-- [**🌐 Homepage**](https://mmmu-benchmark.github.io/) | [**🏆 Leaderboard**](https://mmmu-benchmark.github.io/#leaderboard) | [**🤗 PM<sup>4</sup>Bench**](https://huggingface.co/datasets/MMMU/MMMU_Pro) | [**📖 MMMU-Pro arXiv**](https://arxiv.org/abs/2409.02813) | [**🤗 MMMU**](https://huggingface.co/datasets/MMMU/MMMU/) | [**📖 MMMU arXiv**](https://arxiv.org/pdf/2311.16502.pdf)  -->

[**🌐 Homepage**](https://songjhpku.github.io/PM4Bench/) | [**🤗 PM<sup>4</sup>Bench**](https://huggingface.co/datasets/songjhPKU/PM4Bench) | [**📖 PM<sup>4</sup>Bench arXiv**](https://arxiv.org/abs/2503.18484) 


## 📢 News

- **🔥[2025-03-25]: We uploaded PM<sup>4</sup>Bench to [HuggingFace](https://huggingface.co/datasets/songjhPKU/PM4Bench) and open-sourced the code, paper on GitHub and [arXiv](https://arxiv.org/abs/2503.18484).**

---

## 🧑‍💻 How to Run?
### 📁 Code Directory
- `code/`
  - `eval/`
  - `prompts/`
    - `EVAL/`
    - `OCR/`
    - `VQA/`
  - `infer_api.py`
  - `infer_lmdeploy.py`
  - `score.py`
- `data/`
  - `results/`
  - `tsv/`
    - Store tsv files downloaded from [HuggingFace](https://huggingface.co/datasets/songjhPKU/PM4Bench)
  - `ref_answers/`
    - `MDUR/`
    - `MIQA/`
    - `MMJB/`
    - `MSOCR/`
- `VLM_output/`
- `VLM_output_judge/`
- `logs/`
- `scripts/`
- `requirements.txt`
- `README.md`
### 🏠 Set Up
``` bash
conda env create -f requirements.txt
```
### ⚙️ Inference
#### API Inference
##### Step 0. Configure `.env` file
API inference requires an `API_KEY`. Please configure the `API_KEY` in the `.env` file in the following format: 
``` bash
model_name='sk-123'
```
The `API_KEY` will be loaded through the `infer_api.py` file using:
``` python
load_dotenv()  # load .env file to get API_KEY
API_KEY = os.getenv(MODEL)
```
##### Step 1. Start Inference!
🔴 **Attention: All codes and scripts files are executed in the root directory!**
**e.g.** `python code/infer_api.py [MODEL] [MODE] [SETTING] [LANGUAGE] [TASK] [DATASET] [MAX_TOKENS]`
* `MODEL`: Official model name, such as `gpt-4o-2024-11-20`, `qwen2.5-vl-72b-instruct`, etc.
* `MODE`: For normal VLMs, use `direct`; for reasoning VLMs, use `cot`.
* `SETTING`: `traditional` or `vision`, for detailed explanations please refer to our paper.
* `LANGUAGE`: 10 languages choices, `[ZH, EN, AR, SR, TH, RU, KO, CS, HU, VI]`
* `TASK`: `OCR` for OCR tasks, and `VQA` for VQA tasks under `traditional` or `vision` settings.
* `DATASET`: `[MDUR, MIQA, MMJB, MSOCR]`
* `MAX_TOKENS`: For different models, the `MAX_TOKENS` should be different in case of cut off problems.

Besides, we provide a standard script template `scripts/infer_api.sh`. You can modify parameters directly and run it using 
``` bash
nohup bash scripts/infer_api.sh > logs/infer_api.log 2>&1 &
```

#### Local VLMs Inference
##### Step 0. Use [LMDeploy](https://github.com/InternLM/lmdeploy) to serve models
A special thanks to [LMDeploy](https://github.com/InternLM/lmdeploy) for their work, which has greatly assisted in providing local inference for our work. Please refer to [LMDeploy docs](https://lmdeploy.readthedocs.io/en/latest/get_started/get_started.html) for detailed information of VLMs' deployment and serve. Before inference, you should make sure that **VLM is running** and you have a **local port (like `23333`)** to call it:
``` bash
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES nohup lmdeploy serve api_server $MODEL_PATH 
--backend turbomind --dtype $DTYPE --server-port $SERVER_PORT --tp $TP > $LOG_PATH 2>&1 &
```
We only provide a simplified command line here and if you want to know more paramters and their meanings, please run
``` bash
lmdeploy serve api_server --help
```
##### Step 1. Start Inference!
🔴 **Attention: All codes and scripts files are executed in the root directory!**
**e.g.** `python code/infer_lmdeploy.py [MODEL] [MODE] [SETTING] [LANGUAGE] [TASK] [DATASET] [MAX_TOKENS] [PORT]`
* `MODEL`: Model name, such as `InternVL2_5-78B-MPO`, `qwen2.5-vl-72b-instruct`, etc.
* `MODE`: For normal VLMs, use `direct`; for reasoning VLMs, use `cot`.
* `SETTING`: `traditional` or `vision`, for detailed explanations please refer to our paper.
* `LANGUAGE`: 10 languages choices, `[ZH, EN, AR, SR, TH, RU, KO, CS, HU, VI]`
* `TASK`: `OCR` for OCR tasks, and `VQA` for VQA tasks under `traditional` or `vision` settings.
* `DATASET`: `[MDUR, MIQA, MMJB, MSOCR]`
* `MAX_TOKENS`: For different models, the `MAX_TOKENS` should be different in case of cut off problems.
* `PORT`: Local port (like `23333`) for lmdeploy server to call.

Besides, we provide a standard script template `scripts/infer_lmdeploy.sh`. You can modify parameters directly and run it using 
``` bash
nohup bash scripts/infer_lmdeploy.sh > logs/infer_lmdeploy.log 2>&1 &
```

### 📉 Evaluation & Statistics
#### Step 0. Evaluation
We use `gpt-4o-2024-11-20` to judge VQA performance so you should configure `API_KEY` before evaluation. Besides, you can change base model in `code/eval/{DATASET}/eval_{DATASET}_vqa.py`:
``` python
OPENAI_API_BASE = "https://api.openai.com/v1"
client = OpenAI(
    api_key = os.getenv('gpt-4o-2024-11-20'),
    base_url = OPENAI_API_BASE
)
```
The evaluation codes are executed by:
``` bash
python code/eval/{DATASET}/eval_{DATASET}_{TASK}.py
```
where `DATASET` is chosen from `[MDUR, MIQA, MMJB, MSOCR]` and `TASK` is chosen from `[VQA, OCR]`.
#### Step 1. Statistics
The statistics codes are executed by:
``` bash
python code/score.py
```
and the results are stored in `data/results/{DATASET}_{TASK}_{SETTING}.csv`


## Citation
If you find this work helpful, please consider to star🌟 this repo. Thanks for your support!
```bibtex
@misc{gao2025pm4benchparallelmultilingualmultimodal,
      title={PM4Bench: A Parallel Multilingual Multi-Modal Multi-task Benchmark for Large Vision Language Model}, 
      author={Junyuan Gao and Jiahe Song and Jiang Wu and Runchuan Zhu and Guanlin Shen and Shasha Wang and Xingjian Wei and Haote Yang and Songyang Zhang and Weijia Li and Bin Wang and Dahua Lin and Lijun Wu and Conghui He},
      year={2025},
      eprint={2503.18484},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.18484}, 
}
```