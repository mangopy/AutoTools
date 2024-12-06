from dataclasses import dataclass
import time
from utilize import *
from openai import OpenAI
from typing import Union
from src.template import register_template
import copy
from engine import FuncEngine
from tools import OpenFunc, OpenAPIDoc
from typing import List, Dict
from constant import ExecResponse, BEGIN, END

class BaseRequest:
    def __init__(self, model_name):
        self.model_name = model_name
        self.usage = []
        self.client = OpenAI(api_key='', base_url='')

    def generate(self,
                 messages=None,
                 stop=None,
                 max_len=1000,
                 temp=1,
                 n=1,
                 **kwargs):
        ...

    def __call__(self, *args, **kwargs):
        ...


class OpenLLMRequest(BaseRequest):

    def __init__(self, model_name, api_key, base_url):
        super().__init__(model_name)
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def __call__(
            self,
            messages=None,
            stop=None,
            remain_stop=False,
            max_tokens=16384,
            min_tokens=16,
            temp=0.9,
            top_p=0.95,
            n=1,
            **kwargs
    ):
        """
        support via VLLM
        """
        payload = {
            # "use_beam_search": True,
            "model": self.model_name,
            "messages": messages,
            'max_tokens': max_tokens,
            "temperature": temp,
            'stop': stop,
            "top_p": top_p,
        }
        payload.update(kwargs)
        # for i in range(10):
        #     try:
        begin = time.time()
        response = self.client.chat.completions.create(**payload)
        end = time.time()
        cost = end - begin
        content = response.choices[0].message.content # if n == 1 else [res.message.content for res in response.choices]
        if stop != [] and remain_stop:
            print(f'--------\n{content}\n{response.choices[0].model_extra}\n----------')
            if 'stop_reason' in response.choices[0].model_extra and type(response.choices[0].model_extra['stop_reason'])==str:
                content += response.choices[0].model_extra['stop_reason']
        results = {
            "content": content,
            "time": cost,
            "usage": [response.usage.completion_tokens, response.usage.prompt_tokens, response.usage.total_tokens]
        }
        return results
            # except:
            #     error = sys.exc_info()[0]
            #     print("API error:", error)
            #     time.sleep(30)
        return {"content": 'no response from openai model...', "time": -1, "usage": [0, 0, 0]}


class GPTRequest(BaseRequest):
    def __init__(self, model_name, api_key, base_url):
        super().__init__(model_name)
        self.api_key = api_key
        self.base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def __call__(
            self,
            messages=None,
            stop=None,
            max_tokens=16384,
            min_tokens=16,
            temp=0.9,
            top_p=0.95,
            n=1,
            json_format=False,
            **kwargs
    ):
        payload = {
            "model": self.model_name,
            "messages": messages,
            'max_tokens': max_tokens,
            "temperature": temp,
            "n": n, 'stop': stop,
            "top_p": top_p,
        }
        payload.update(kwargs)
        if json_format:
            payload['json_format'] = True
        for i in range(10):
            try:
                begin = time.time()
                response = self.client.chat.completions.create(**payload)
                end = time.time()
                cost = end - begin
                content = response.choices[0].message.content if n == 1 else [res.message.content for res in response.choices]
                results = {
                    "content": content,
                    "time": cost,
                    "usage": [response.usage.completion_tokens, response.usage.prompt_tokens, response.usage.total_tokens]
                }
                return results
            except:
                error = sys.exc_info()[0]
                print("API error:", error)
                time.sleep(30)
        return {"content": 'no response from openai model...', "time": -1, "usage": [0, 0, 0]}


def llm_request_wrapper(
        model_name,
        api_key,
        base_url,
        backend: str = 'openai',
        **kwargs
) -> Union[OpenLLMRequest, GPTRequest]:
    params = {
        "model_name": model_name,
        "api_key": api_key,
        "base_url": base_url
    }
    if backend.lower() == 'customized':
        return OpenLLMRequest(**params, **kwargs)
    elif backend.lower() == 'openai':
        return GPTRequest(**params, **kwargs)
    else:
        raise NotImplementedError


@dataclass
class LLMConfig:
    model_name_or_path: str
    base_model: str
    api_key: str
    base_url: str
    backend: str
    begin: str = BEGIN
    end: str = END



class FuncLLM:
    def __init__(
        self,
        llm_config: Union[LLMConfig, Dict],
        kernel_gateway_url: str
    ):
        if type(llm_config) == dict:
            llm_config = LLMConfig(**llm_config)

        self.llm: Union[GPTRequest, OpenLLMRequest] = llm_request_wrapper(
            llm_config.model_name_or_path,
            llm_config.api_key,
            llm_config.base_url,
            llm_config.backend,
        )
        self.begin = llm_config.begin
        self.end = llm_config.end

        self.template = register_template(
            begin=llm_config.begin,
            end=llm_config.end,
        )

        self.engine = FuncEngine(
            begin=llm_config.begin,
            end=llm_config.end,
            kernel_gateway_url=kernel_gateway_url,
            headers='',
            backend=''
        )


    def restart_session(self):
        pass

    def _encapsulate(self, messages, n=3) -> Union[OpenFunc, None]:
        _message = copy.deepcopy(messages)
        for i in range(n):
            output = self.llm(
                messages=messages+[{"role": "assistant", "content": f"{self.begin}\n"}],
                stop=self.end
            )
            content = output['content']
            response = self.engine.run(content)
            if not response.state:
                continue
            notebook = self.engine.extract_functions_from_str(content)
            if notebook is not None and len(notebook)>0:
                return notebook[0]
            # messages.append({"role": "assistant", "content": content})
            # messages.append({"role": "assistant", "content": f"Here is the response: {response}. May be you can change your parameters and try again. \nYour output: ```python"})
        return None

    def encapsulate(self, api: OpenAPIDoc, mode='text') -> Union[OpenFunc, None]:
        doc = api.formulate(mode, extra_kv=None)
        message = self.template.initial_message_for_encap(docs=doc, headers=api.headers)
        # 请求调用
        notebook: OpenFunc = self._encapsulate(messages=message, n=3)
        if notebook is not None:
            notebook.update(api_idx=api.idx, tool=api.tool, category=api.category, response=api.response)
            return notebook
        else:
            return None

    def _verify(self, func, messages, n=3):
        _engine = copy.deepcopy(self.engine)
        _engine.load(func)
        _message = copy.deepcopy(messages)
        for i in range(n):
            output = self.llm(
                messages=messages+[{"role": "assistant", "content": f"Test code: ```python\n"}],
                stop=[self.end+'\n']
            )
            content = output['content']
            response: "ExecResponse"= _engine.run(content)
            if response.state:
                return response
            # 找到定义的函数, 以及函数的签名，内容等
            # messages.append({"role": "assistant", "content": content})
            # messages.append({"role": "assistant", "content": f"Here is the response: {response}. May be you can change your parameters and try again. \nYour output: ```python"})
        return ExecResponse(
            code=None,
            response=None,
            state=False,
            urls=None,
        )
    def verify(
            self,
            func: OpenFunc,
            n=3,
            add_instance: bool = False,
            add_response: bool = False
    ) -> ExecResponse:
        func_str = func.str(add_instance, add_response)
        messages = self.template.initial_message_for_verify(func_str)
        result = self._verify(func=func, messages=messages, n=n)
        return result

    def _generate(self, query, func_str, tool_docs, n=3, max_trunc=1000):
        usage = []
        messages = [
            {"role": "user", 'content': self.template.warp_system_prompt(funcs=func_str, docs=tool_docs)},
            {"role": "assistant", "content": "Sure, I will generate the functions step-by-step and end up with `print` statement to show myself their execution results."},
            {"role": "user", 'content': self.template.warp_user_initial_prompt(query=query)},
        ]
        print(f"User: {query}")
        select_tools = []
        step_state = 1
        for i in range(0, n):
            if i==0:
                prefix = f"In this step, I should call this function and end with `print` to print the results:\n{self.begin}"
            else:
                if step_state==1:
                    prefix = f"Receive the correct response, I should move to next step. In this step, I will call this function and end with `print` to print the results: {self.begin}"
                else:
                    prefix = f"Sorry, I will fix my bug and call functions correctly. Here is my code: {self.end}"

            response = self.llm(
                messages=messages+[{"role": "assistant", "content": prefix}],
                max_tokens=1536,
                stop=[self.end],
                remain_stop=True
            )
            usage.append(response['usage'])
            content = response['content']

            messages.append({"role": 'assistant', "content": content})
            print(content)

            norm_code, feedback, selected_tool, state = self.engine.run(content)
            print(feedback)

            select_tools.extend(selected_tool)
            messages.append({"role": 'user', "content": f"[Function]: The results is: {feedback[:max_trunc]}..."})

            if 'FINISH' in feedback:
                return usage, messages, select_tools

        print(select_tools)
        return usage, messages, select_tools

    def load(self, funcs: Union[List[OpenFunc], OpenFunc], debug=False):
        if type(funcs)!=list:
            funcs = [funcs]
        for func in funcs:
            self.engine.load(func)
        if debug:
            print('The environment...')
            print(self.engine.env['library'])
            print(self.engine.env['func'])

    def restart(self, debug=False):
        self.engine.restart()
        if debug:
            print('Restart the environment...')
            print(self.engine.env['library'])
            print(self.engine.env['func'])

    def generate(
            self,
            query: str,
            funcs: List[OpenFunc],
            tools: List[str]
    ):
        # load环境
        func_str, library, functions = [], [], []
        for func in funcs:
            func_str.append(func.signature+func.docstring)
            library.extend(func.library)
            functions.append(func.source)
        library = '\n'.join(list(set(library)))
        context = library + '\n\n' + '\n\n'.join(functions)
        self.engine.initialize(context)
        usage, message, select_tools = self._generate(
            query=query,
            func_str=func_str,
            tool_docs=tools
        )

        return message

    def chat(self):
        ...

def integrate_verify():
    pass


