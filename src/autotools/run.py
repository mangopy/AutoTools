"""
Before running this demo, please first run `pip install jupyter_kernel_gateway` to prepare the jupyter kernel and `jupyter kernelgateway` to start your local jupyter. Replace the port 8888 with your actual port.
"""
import glob
import os.path
from basellm import FuncLLM
from tools import OpenToolDoc, OpenFunc, OpenAPIDoc
import json
from utilize import load_data, write_file
from tqdm import tqdm

llm_config = {
    "model_name_or_path": "<MODEL_NAME>", #
    "base_model": "gpt", # gpt or open-source llm
    "api_key": "<API-KEY>", # API KEY to access the LLM (openai API for gpt; VLLM local API key for open-source LLM
    "base_url": "<BASE_URL>", # The base url to access LLM
    "backend": "<BACKEND>", # backend of LLM, gpt or vllm engine
}
func_llm = FuncLLM(
    llm_config=llm_config,
    kernel_gateway_url="127.0.0.1:8888"
)
output_folder = './funcs'
specification = json.load(open('<TOOL_DOCUMENTATION_FILE>'))
data = json.load(open('<EVALUATION_DATA_FILE>'))



def encapsulate_demo_TMDB(header_key: str):
    """

    :param header_key: the user's authentic api key to access tmdb platform
    :return:
    """
    base_url = 'https://api.themoviedb.org'
    headers = {'Authorization': f'Bearer {header_key}'}
    tool_doc = OpenToolDoc(base_url=base_url, headers=headers, specification=specification)
    notebooks = {}
    for file in glob.glob(f"{output_folder}/*.json"):
        notebook = OpenFunc(**json.load(open(file)))
        notebooks[notebook.api_idx] = notebook
    apis = tool_doc.api
    apis = [api for api in apis if api.idx not in notebooks]
    for api in tqdm(apis):
        idx = api.idx
        # if idx in notebooks:
        #     continue
        notebook = func_llm.encapsulate(api=api)
        if notebook is not None:
            notebooks[idx] = notebook
            write_file(notebook.to_dict(), os.path.join(output_folder, f'{idx}.json'))

def verify_demo():
    cnt = 0
    bar = tqdm(glob.glob(f"{output_folder}/*.json"))
    for file in bar:
        func = OpenFunc(**json.load(open(file)))
        response = func_llm.verify(func)
        if response.state:
            cnt+=1
        print(response)
        bar.set_postfix({"successfully verified": cnt})

# demo to encapsulation
# encapsulate_demo()

# demo for integration verification
# verify_demo()
