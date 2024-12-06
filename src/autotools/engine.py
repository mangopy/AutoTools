import copy
import re
import json
from constant import ExecResponse
import requests
import websocket
import ast
import inspect
from tools import OpenFunc
from typing import List
from uuid import uuid4
import os
import time
from tornado.escape import json_encode, json_decode, url_escape

def _simplify_json(raw_json: dict):
    if isinstance(raw_json, dict):
        for key in raw_json.keys():
            raw_json[key] = simplify_json(raw_json[key])
        return raw_json
    elif isinstance(raw_json, list):
        if len(raw_json) == 0:
            return raw_json
        else:
            return [_simplify_json(raw_json[0])]
    else:
        return type(raw_json).__name__


def simplify_json(raw_json: dict):
    return _simplify_json(copy.deepcopy(raw_json))

def get_yaml(value, name, indent=0):
    result = ['\t' * indent + f"- {name}: {type(value).__name__}"]
    if list == type(value):
        element = f'{name}[0]'
        indent += 1
        if value != []:
            result.extend(get_yaml(value[0], element, indent))

    elif dict == type(value):
        for k, v in value.items():
            result.extend(get_yaml(v, k, indent + 1))
    return result



class FuncEngine:

    def  __init__(
            self,
            begin,
            end,
            kernel_gateway_url,
            headers: str,
            backend='rest',):

        self.headers = headers
        self.backend = backend
        self.begin = begin
        self.end = end

        self.kernel_gateway_url = kernel_gateway_url
        self.kernel_id = self.create_kernel(kernel_gateway_url)

        self.env = {
            "library": [],
            "func": []
        }

        self.ceil: List[ExecResponse] = []

    def create_kernel(self, kernel_gateway_url):
        """
        创建一个 Jupyter Kernel 并返回 Kernel ID
        """
        response = requests.post(f"http://{kernel_gateway_url}/api/kernels")
        if response.status_code != 201:
            raise Exception(f"Error creating kernel: {response.text}")

        kernel = response.json()
        kernel_id = kernel['id']
        return kernel_id

    def execute_code_via_websocket(self, code, timeout=60):
        """
        通过 WebSocket 向指定内核发送代码并接收执行结果
        """
        # WebSocket 连接URL
        ws_url = f"ws://{self.kernel_gateway_url}/api/kernels/{self.kernel_id}/channels"
        ws = websocket.create_connection(ws_url)
        msg_id = uuid4().hex

        # 要执行的代码的消息格式
        code_to_execute = {
            "header": {
                # "msg_id": "execute_code",
                "msg_id": msg_id,
                "msg_type": "execute_request",
                "username": "",
                "session": "",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,  # 需要执行的代码
                "silent": False,
                "store_history": True,
            },
            "buffers": {}
        }

        # 发送代码到内核
        ws.send(json.dumps(code_to_execute))
        execution_done = False
        outputs = []
        start_time = time.time()

        while not execution_done:
            msg = ws.recv()
            msg = json.loads(msg)
            msg_type = msg['msg_type']
            parent_msg_id = msg['parent_header'].get('msg_id', None)

            if parent_msg_id != msg_id:
                continue

            if os.environ.get("DEBUG", False):
                print(f"MSG TYPE: {msg_type.upper()} DONE:{execution_done}\nCONTENT: {msg['content']}")

            if msg_type == 'error':
                traceback = "\n".join(msg["content"]["traceback"])
                outputs.append(traceback)
                execution_done = True
            elif msg_type == 'stream':
                outputs.append(msg['content']['text'])
            elif msg_type in ['execute_result', 'display_data']:
                text_output = msg['content']['data'].get('text/plain', '')
                outputs.append(text_output)
                if 'image/png' in msg['content']['data']:
                    # use markdone to display image (in case of large image)
                    # outputs.append(f"\n<img src=\"data:image/png;base64,{msg['content']['data']['image/png']}\"/>\n")
                    outputs.append(f"![image](data:image/png;base64,{msg['content']['data']['image/png']})")

            elif msg_type == 'execute_reply':
                execution_done = True

            if time.time() - start_time > timeout:
                print("Timeout: No response from the kernel.")
                break
        # 关闭 WebSocket 连接
        ws.close()
        if execution_done:
            return '\n'.join(outputs)
        else:
            return None

    @staticmethod
    def is_json(json_str):
        try:
            tmp = json.loads(json_str)
            return True
        except:
            return False

    @staticmethod
    def compile(func_str) -> bool:

        # rule-base compile
        flag = ['def', 'return', 'requests']
        if any([e not in func_str for e in flag]):
            return False
        try:
            local_context = {}
            exec(func_str, globals(), local_context)
            return True
        except:
            return False

    def code_normalization(self, code):
        def remove_example_usage_comments(code_snippet):
            """
            Remove comments after 'Example usage:' in the given code snippet.
            :param code_snippet: A string containing the code snippet
            :return: A cleaned-up version of the code with comments removed after 'Example usage:'
            """
            lines = code_snippet.split('\n')
            cleaned_code = []
            in_example_usage = False

            for line in lines:
                # 检查是否到达 Example usage 部分
                if 'example usage' in line.lower():
                    in_example_usage = True
                    # cleaned_code.append(line)
                    continue

                # Example usage 部分的行处理
                if in_example_usage:
                    # 移除以 # 开头的注释
                    # code_line = line.split('#')[0].strip()
                    code_line = line.replace('#', '').strip()
                    if code_line:
                        cleaned_code.append(code_line)
                else:
                    cleaned_code.append(line)

            return '\n'.join(cleaned_code)

        def extract_code(string):
            pattern = rf"{self.begin}(.*?){self.end}"
            matches = re.findall(pattern, string, re.DOTALL)
            string = copy.deepcopy(string) if matches == [] else '\n\n'.join(matches)
            string = string.replace(self.begin, '')

            return string

        def pre_normalization(string):
            string = string.replace(f'{self.begin}\n\n{self.begin}', f"{self.begin}")
            string = string.replace(f'{self.begin}\n{self.begin}', f"{self.begin}")
            string = string.replace(f'{self.end}\n\n{self.end}', f'{self.end}')
            string = string.replace(f'{self.end}\n{self.end}', f'{self.end}')
            return string

        def post_normalization(string):
            string = string.replace(f'{self.begin}', '')
            string = string.replace(f'{self.end}', '')
            string = string.replace(f'```python', '')
            string = string.replace(f'```', '')
            string = string.replace('themoviemb', 'themoviedb')
            string = string.replace('.themoviesb', 'themoviedb')
            string = string.replace('\_', '-')
            string = remove_example_usage_comments(string)

            return string

        code = pre_normalization(code)
        code = extract_code(code)
        code = post_normalization(code)
        return code

    def extract_functions_from_str(self, func_str) -> List[OpenFunc]:
        """
        """
        try:
            func_str = self.code_normalization(func_str)
            parsed_code = ast.parse(func_str)
            namespace = {}
            function_sources = {}
            libraries = set()
            for node in parsed_code.body:
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        libraries.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    libraries.add(node.module)
                if isinstance(node, ast.FunctionDef):
                    func_code = ast.get_source_segment(func_str, node)
                    exec(func_code, namespace)
                    function_sources[node.name] = func_code
            functions = {name: obj for name, obj in namespace.items() if callable(obj)}
            result = []
            for func_name, func in functions.items():
                function_name = func.__name__
                docstring = func.__doc__
                source_code = function_sources[func_name]
                signature = str(inspect.signature(func))
                result.append(OpenFunc(
                    name=function_name,
                    source=source_code,
                    docstring=docstring,
                    signature=function_name+signature,
                    library=list(libraries),
                    instance=None,
                ))
        except:
            result = []
        return result

    def judge(self, response):
        if response == None:
            return 0

        if type(response) == str:
            keys = ['not found', '404 error', 'url 404','url: 404', 'could not be found','not be found','status code 404', 'HTTP Error', '404 Client Error','HTTP request error', 'Error: 404'
                    'success":false','No keywords found', 'not available', 'No results found', 'out of range','Client Error',
                    'is not subscriptable','Exception Value','is not defined',"n't exist",'api_key','YOUR_API_KEY']
            for e in  keys:
                if e.lower() in response.lower():
                    print(e)
                    return 0
            tmp = copy.deepcopy(response.split())
            tmp = [e.lower() for e in tmp]
            for e in ['fail', 'invalid', "n't"]: # 'None', '[]'
                if e.lower() in tmp:
                    print(e)
                    return 0
        return 1

    def wrap_rest(self,):
        wrap_code = f"""
import json
import requests
from collections import defaultdict
class Requests:
    def __init__(self):
        self.original_requests = requests
        self.headers= {self.headers}
        self.cnt = defaultdict(int)

    def get(self, url, **kwargs):
        if 'headers' in kwargs:
            kwargs['headers'] = {self.headers}
        self.cnt[url]+=1
        return self.original_requests.get(url, **kwargs)

    def post(self, url, data=None, json=None, **kwargs):
        if 'headers' in kwargs:
            kwargs['headers'] =  {self.headers}
        self.cnt[url]+=1
        return self.original_requests.post(url, data=data, json=json, **kwargs)

    def count(self):
        # print(self.cnt)
        return json.dumps(dict(self.cnt.items()))

    def __getattr__(self, name):
        return getattr(self.original_requests, name)
"""
        return wrap_code

    def wrap_toolbench(self, ):
        wrap_code = f"""
from collections import defaultdict
class Requests:
    def __init__(self):
        self.cnt = defaultdict(int)
        self.headers = {self.headers}

    def post(self, url, json=None, **kwargs):
        import requests
        json['strip'] = 'truncate'
        json['toolbench_key'] = self.headers['toolbench_key']
        if 'tool_input' not in json:
            json['tool_input'] = {{}}
        response = requests.post("http://8.130.32.149:8080/rapidapi", json=json, **kwargs)
        self.cnt[json['api_name']] += 1
        return response

    def count(self):
        print('----------------------')
        if len(self.cnt)==0:
            return 'No API call -> 0'
        res = []
        for k, v in self.cnt.items():
            res.append(k + '->' + str(v))
        res = '###'.join(res)
        return res
"""

        return wrap_code


    def is_code(self,code_str: str) -> bool:
        code = code_str.replace(self.begin, '').replace(self.end, '')
        try:
            local_context = {}
            exec(code, globals(), local_context)
            return True
        except:
            return False

    def load(self, func: OpenFunc):
        # lib
        result = self.run('\n'.join([f'import {lib}' for lib in func.library]))
        if not result.state:
            print(f'fail to load the library {func.library}...')
        else:
            self.env['library'].extend(func.library)

        result = self.run(func.source)
        if not result.state:
            print(f'fail to load the source code of function: {self.begin}\n{func.source}\n{self.end}')
        else:
            self.env['func'].extend(func.source)

    def restart(self):
        self.env = {"library": [], "func": []}

    def initialize(self, context='', backend='rest', warp=True):
        # lib

        # warp request
        if backend == 'rest':
            wrap = self.wrap_rest()
        else:
            wrap = self.wrap_toolbench()
        pass


    def run(self, code,):
        norm_code = self.code_normalization(code) # .replace('import requests', '')
        finish = '<Successfully Finish>'
        wrap_code = f"""
# ##############
headers = {self.headers}
requests = Requests()
{norm_code}

print('{finish}')
print('----------------------')
print(requests.count())
"""

        response = self.execute_code_via_websocket(norm_code)
        if response is not None:
            return ExecResponse(
                code=norm_code,
                response=response,
                urls=[],
                state=True
            )
        else:
            return ExecResponse(
                code=norm_code,
                response=response,
                urls=[],
                state=False
            )
        # if '----------------------' in result:
        #     try:
        #         response, logs = result.split('----------------------')
        #     except:
        #         response = result.split('----------------------')[0]
        #         logs = "{}"
        #
        #     if finish in response and self.judge(response):
        #         state=True
        #     else:
        #         state=False
        #
        #     response = response.replace(finish, '').strip()
        #     if state and response == '':
        #         response = "Successfully execute your code but print anything. To receive the execution results, please `print` the value your want."
        #
        #     logs = json.loads(logs.strip())
        #     urls = sum([[k]*v for k, v in logs.items()], [])
        #     return ExecResponse(
        #         code=norm_code,
        #         response=response,
        #         urls=urls,
        #         state=state
        #     )
        # else:
        #     return ExecResponse(
        #         code=norm_code,
        #         response=result,
        #         urls=[],
        #         state=False
        #     )
