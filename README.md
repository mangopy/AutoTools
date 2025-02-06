<div align="center">
   <h1>AutoTools <img src="assets/images/icon.png" alt="Logo" width="30px" height="30"/></h1>
</div>


🔧This project (AutoTools) aims to introduce an automatic framework to enable the large language models (LLMs) as autonomous agents. By integrating the AutoTools,  the LLM automatically transforms tool documentation into callable functions, verify syntax and runtime correctness, and flexibly compose these functions into executable programs to solve practical tasks.
![img.png](./assets/images/method.png)

## News

- **[2025.1.20]** The second version of our paper has been accepted by the [THE WEB CONFERENCE 2025](https://www2025.thewebconf.org/) (WWW2025) 🎉🎉🎉.
- **[2024. 12.7 ]** Our code was released, including main code for inference and supervised fine-tuning.
- **[2024.5.26]** The first version of our paper has been released in arxiv. See our paper in this [link](https://arxiv.org/abs/2405.16533).

## Quick start for AutoTools

We provide the following quick demo to learn about our project. The full demo is also provided in `./src/autools/run.py` for a clear illustration.

1. Build your own tool-augmented (func-augmented) LLMs
```python
from basellm import FuncLLM

llm_config = {
    "model_name_or_path": "<MODEL_NAME>", #
    "base_model": "gpt", # gpt or open-source llm
    "api_key": "<API-KEY>", # API KEY to access the LLM (openai API for gpt; VLLM local API key for open-source LLM
    "base_url": "<BASE_URL>", # The base url to access LLM
    "backend": "<BACKEND>", # backend of LLM, gpt or vllm engine
}
func_llm = FuncLLM(
    llm_config=llm_config,
    kernel_gateway_url="127.0.0.1:8888" # default gate way to execute python code
)
```

2. A demo for tool encapsulation (tool doc -> function).
```python
# TMDB dataset
import json
import glob
import tqdm
from tools import OpenToolDoc, OpenFunc, OpenAPIDoc

header_key = ''
base_url = ''
header_key = ''
output_folder = './funcs'
specification = json.load(open('<TOOL_DOCUMENTATION_FILE>'))
data = json.load(open('<EVALUATION_DATA_FILE>'))
notebooks = {}
tool_doc = OpenToolDoc(base_url=base_url, headers={'Authorization': f'Bearer {header_key}'}, specification=specification)
apis = tool_doc.api
apis = [api for api in apis if api.idx not in notebooks]
for api in tqdm(apis):
    idx = api.idx
    notebook = func_llm.encapsulate(api=api)
    if notebook is not None:
        notebooks[idx] = notebook
```

3. A demo for tool verification (tool doc -> function).
```python
import tqdm
import glob
cnt = 0
output_folder = ''  # the folder to store the generated functions
bar = tqdm(glob.glob(f"{output_folder}/*.json"))
for file in bar:
    func = OpenFunc(**json.load(open(file)))
    response = func_llm.verify(func)
    if response.state:
        cnt+=1
    print(response)
    bar.set_postfix({"successfully verified": cnt})
```


## AutoTools-Learning with Open-source LMs

To extend our framework into wide range of open-source LMs, we propose the AutoTools-Learning, which consists of three learning tasks improve the LLM’s expertise within AutoTools.

We also collect high-quality training datasets to facilitate our AutoTools-Learning process. A template bash command to `train your own LM` or `reproduce our experiment` is provide below.

```bash
cd ./learning

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7   nohup torchrun  --nproc_per_node=8 --master_port=11020 run.py \
    --model_name_or_path   mistralai/Mistral-7B-Instruct-v0.3 \
    --dataset_name_or_path  training.data.json \
    --deepspeed ./script/ds_z3_config.json \
    --output_dir  ./output \
    --overwrite_cache True \
    --warmup_ratio 0.1 \
    --report_to wandb \
    --run_name test_run \
    --logging_steps 1 \
    --cutoff_len 8192 \
    --max_samples 200000 \
    --save_steps  200 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2.0e-5 \
    --num_train_epochs 3 \
    --lr_scheduler_type cosine \
    --remove_unused_columns False \
    --bf16 True  &
```
You can also set the `resume_from_checkpoint` to restart your training from previous checkpoint.

## A New Benchmark -- AutoTools-Eval
In our work, we provide a more challenging benchmark to evaluate our framework under more complex scenarios. The tools in our dataset can be found in the './tools' dataset.  The evaluation example can also be found in `./data` folder. 

A concrete example for our evaluation dataset.
```json
 {
        "query": "Please help me find a pasta recipe, rice recipe, and steak recipe, each with a carbohydrate content between 10 and 50 grams per gram. Among these recipes, which one has the highest calorie content and what equipment is needed for it?",
        "api_list": [
            "GET_menu_item_information",
            "GET_ingredient_information",
            "GET_summarize_recipe",
            "GET_dish_pairing_for_wine",
            ...
        ],
        "solution": [
            [
                "GET_search_recipes",
                3
            ],
            ...
        ],
        "url": [
            "https://spoonacular-recipe-food-nutrition-v1.p.rapidapi.com/recipes/complexSearch",
            ...
        ],
        "dataset": "food",
        "qid": "q0"
    }
```

> The full evaluation dataset will be released after the review processing.


## Training data for AutoTools-Learning

We collect 7,243/12, 251/14,689 examples for the tool understanding, relevance learning, and function learning tasks in our AutoTools-Learning method.
To ensure data quality, we apply strict filtering strategies, such as removing examples with empty tool responses, unsolvable queries, or incorrect tool-calling parameters. 
We also reformat these datasets into a unified interactive format. Each formatted example begins with a system instruction describing the task and initial input, followed by interactions between two roles: the user and the LLM, or the LLM and the execution environment.
The demo training data (1,0000) can be found in the `./data` folder.

### 📊 Model Experiments Results

![img.png](./assets/images/results.png)


## Resource

The training dataset of the three learning tasks in AutoTools-learning can be downloaded from the following link.

| Learning task         | Note                                                                                              |       Link       |
|:----------------------|:--------------------------------------------------------------------------------------------------|:----------------:|
| Tool Understanding  | learning to encapsulate tool documentation in natural language into python functions.             | [Google drive](https://drive.google.com/file/d/1uYIwG1Qj0ut7A1mtjlyKVc_leCOa7hv2/view?usp=sharing) |
| Relevance Learning  | learning to select target tools by generating their identifiers, i.e., tool name.                 | [Google drive](https://drive.google.com/file/d/1qhhe3dviPSTynfbkvlxBhF6-Fk1_VaNx/view?usp=sharing) |
| Function Learning   | learning to programmatically use pre-encapsulated tools for task-solving.                         | [Google drive](https://drive.google.com/file/d/1AOcOh1OzvBJI_J0R3G5DWDtGIB4BC8-p/view?usp=sharing) |

## Todo
- [ ] More details about the experiments will be released
- [ ] The AutoTools-Eval is on the way!
- [ ] Model checkpoint will be available on Huggingface!

## Acknowledgement
We sincerely thank prior work, such as [CodeAct](https://github.com/xingyaoww/code-act) and [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main), which inspire this project or provide strong technique reference.

## Citation
```text
@inproceedings{autotools,
	title     = {Tool Learning in the Wild: Empowering Language Models as Automatic Tool Agents},
	author    = {Zhengliang Shi, Shen Gao, Lingyong Yan, Yue Feng, Xiuyi Chen, Zhumin Chen, Dawei Yin, Suzan Verberne, Zhaochun Ren},
	year      = 2025,
	booktitle = {WWW}
}
```

