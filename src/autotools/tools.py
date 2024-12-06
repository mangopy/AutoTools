import copy
import json
from typing import List, Dict, Union, Any, Iterator, Optional
import yaml
from src.utilize import load_data, write_file, change_name, standardize
from src.constant import BEGIN, END
from dataclasses import dataclass, asdict


def normalize(sss):
    for s in ['<br />', '<br/>', '_**NOTE**:']:
        sss = sss.replace(s, '\n')
    sss = sss.split('\n')[0]
    tmp = [
        '(/documentation/web-api/#spotify-uris-and-ids)',
        '(https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2)',
        '(https://www.spotify.com/se/account/overview/)',
        '(http://en.wikipedia.org/wiki/ISO_3166-1_alpha-2)',
        '<br/>',
        '<br>',
        '<br />',
        '\n',
        '/documentation/general/guides/track-relinking-guide/',
        '(http://en.wikipedia.org/wiki/Universal_Product_Code)',
        '(http://en.wikipedia.org/wiki/International_Standard_Recording_Code)',
        '/documentation/web-api/#spotify-uris-and-ids'
    ]
    for s in tmp:
        sss = sss.replace(s, '')

    for i in range(10):
        sss = sss.replace(f'[{i}].', '')
        sss = sss.replace(f'[{i}]', '')
    return sss.strip()


def simplify_response_template(data):
    if 'required' in data and 'properties' in data:
        for k, v in data['properties'].items():
            if k not in data['required']:
                data.pop(k)
    if 'type' in data and data['type'] == 'object' and 'properties' in data:
        for k, v in data['properties'].items():
            data['properties'][k] = simplify_response_template(v)
    else:
        for k, v in data.items():
            if k in ['example', 'nullable', 'x-spotify-docs-type']:
                data.pop(k)
            if k == 'description':
                data[k] = normalize(v)
    return data


def simplify_spec(data: Dict):
    """
    Recursively simplify the dictionary by removing specific keys.

    :param data: The input dictionary to be simplified.
    :return: A simplified dictionary with specified keys removed.
    """
    keys_to_remove = ['example', 'nullable', 'x-spotify-docs-type', 'required', 'default', 'minimum', 'maximum', 'examples']

    if isinstance(data, dict):
        results = {}
        for k, v in data.items():
            if k in keys_to_remove:
                continue
            # if k == 'description':
            #     results[k] = normalize(simplify_spec(v))
            # else:
            results[k] = simplify_spec(v)
        return results
    elif isinstance(data, list):
        return [simplify_spec(item) for item in data]
    else:
        if type(data) == str:
            return normalize(data)
        return data

@dataclass
class OpenAPIDoc:
    base_url: str
    path: str
    method: str
    description: str
    parameter: List[Dict]
    response: Dict
    doc: Dict
    headers: Optional[Dict] = None

    category: Optional[str] = None
    tool: Optional[str] = None


    def __post_init__(self):
        self.url = self.get_url()
        self.idx = self.get_id()

    def get_id(self):
        idx = self.path.replace('/', '_').replace('(',' ').replace(')', ' ').replace('{','').replace('}','')
        idx = self.method + '_' + idx
        idx = idx.replace('__','_')
        return idx

    def get_url(self):
        if self.base_url.endswith('/') and self.path.startswith('/'):
            return self.base_url + self.path[1:]
        elif not self.base_url.endswith('/') and not self.path.startswith('/'):
            return self.base_url + '/' + self.path
        else:
            return self.base_url + self.path

    def formulate(self, mode: str = Union['json', 'text', 'yaml'], extra_kv=None):
        # url, headers, parameters, results type
        template = f"""API path: {self.path}\n"""
        if mode == 'json':
            template += f"```json\n{json.dumps(self.doc, indent=4)}\n```"
        elif mode == 'yaml':
            # 将 Python 对象转换为 YAML
            template += yaml.dump(self.doc, allow_unicode=True)
        else:
            params = [f"    - {param['name']} (in: {param['in']}, required: {param.get('required', False)})" for param in self.parameter]
            params = '\n'.join(params)
            response = json.dumps(self.response, indent=4)
            template += f"""- Description: {self.description}
- Parameters: \n{params}
- Usages: {BEGIN}
import requests
url = {self.url}
headers = {self.headers}
response = requests.{self.method.lower()}(...)
{END}
- Response: {BEGIN}
{response}
{END}"""
        template += extra_kv if type(extra_kv) == str else ''
        return template

@dataclass
class OpenToolDoc:

    def __init__(self, base_url: str, specification: Union[Dict, str], headers=None):
        self.base_url = base_url
        self.category = specification.get("category", "")
        self.tool_name = specification.get("tool_name", "")
        self.tool_name = standardize(self.tool_name)
        self.description = f"This is the sub-function for tool \"{self.tool_name}\" in {self.category} category."
        self.spec = specification if type(specification)==dict else load_data(specification)
        self.headers = headers

        paths = self.spec.get("paths", {})
        self.api = []
        for path, methods in paths.items():
            for method, doc in methods.items():
                _api = OpenAPIDoc(
                    base_url=base_url,
                    path=path,
                    method=method,
                    description=doc.get('description', ""),
                    parameter=self.get_parameters(path, method),
                    response=self.get_response_type(path, method),
                    doc=doc,
                    headers=headers,
                )
                self.api.append(_api)


    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for api in self.api:
            yield api

    def get_global_parameters(self):
        return self.spec.get('components', {}).get('parameters', {})

    def get_response_type(self, path, method):
        path_item = self.spec['paths'].get(path, {})
        responses = path_item.get(method, {}).get('responses', {})

        if '200' in responses:
            response_content = responses['200'].get('content', {})
            if response_content:
                media_type = next(iter(response_content))
                return response_content[media_type].get('schema', {})

            return None

    def get_parameters(self, path, method):
        path_item = self.spec['paths'].get(path, {})
        path_parameters = path_item.get('parameters', [])
        method_parameters = path_item.get(method, {}).get('parameters', [])
        all_parameters = path_parameters + method_parameters
        global_parameters = self.get_global_parameters()
        for param in all_parameters:
            if '$ref' in param:  # 如果参数是引用类型
                ref_key = param['$ref'].replace('#/components/parameters/', '')
                if ref_key in global_parameters:
                    all_parameters.remove(param)  # 移除引用参数
                    all_parameters.append(global_parameters[ref_key])  # 添加全局参数
        return all_parameters

#     def formulate(self, path, method: str, mode: str = Union['json', 'text', 'yaml'], extra_kv=None):
#         # url, headers, parameters, results type
#         params = self.get_parameters(path=path, method=method)
#         url = self.get_path_url(path=path)
#         response = self.get_response_type(path=path, method=method)
#         template = f"""API path: {path}\n"""
#         if mode == 'json':
#             doc = self.spec['paths'].get(path, {})
#             template += f"```json\n{json.dumps(doc, indent=4)}\n```"
#         elif mode == 'yaml':
#             doc = self.spec['paths'].get(path, {})
#             # 将 Python 对象转换为 YAML
#             template += yaml.dump(doc, allow_unicode=True)
#         else:
#             params = [f"    - {param['name']} (in: {param['in']}, required: {param.get('required', False)})" for param in params]
#             params = '\n'.join(params)
#             response = json.dumps(response, indent=4)
#             template += f"""- Description: {self.description}
# - Parameters: \n{params}
# - Usages: {BEGIN}
# import requests
# url = {url}
# headers = {self.headers}
# response = requests.{method.lower()}(...)
# {END}
# - Response: {BEGIN}
# {response}
# {END}"""
#         template += extra_kv if type(extra_kv) == str else ''
#         return template


class RestAPIDoc:
    def __init__(self, spec: dict):

        self.api_name = spec['api_name']
        self.tool_name = spec['tool_name']
        self.category = spec['category']
        self.method = spec['method']
        self.url = spec['url']
        self.description = spec['api_description']
        self.parameter = spec['parameters'] if 'parameters' in spec else []
        self.headers = spec.get('headers', {'Authorization': 'Bearer eyJhbGciOiJIUzI1NiJ9'})
        if 'requestBody' in spec and spec['requestBody'] is not None:
            self.requestBody = simplify_spec(spec['requestBody']['content']['application/json']["schema"]['properties'])
        else:
            self.requestBody = 'This API do not need the request body when calling.'

        self.responses = {}
        if 'template' in spec and spec['template'] is not None:
            if type(spec['_template']) == dict:
                self.responses['template'] = simplify_spec(spec['template'])
                self.responses['template'] = json.dumps(self.responses['template'], indent=4)
            else:
                self.responses['template'] = spec['template']
        else:
            self.responses['template'] = 'No information about the type of the execution results'

        if '_template' in spec and spec['_template'] is not None:
            if type(spec['_template']) == dict:
                self.responses['_template'] = json.dumps(spec['_template'], indent=4)
            else:
                self.responses['_template'] = spec['_template']
        else:
            self.responses['_template'] = 'No information about the type of the execution results'

        self.raw = copy.deepcopy(spec)

    def get_parameters(self) -> str:
        parameter = []
        if self.parameter == []:
            parameter.append('  -- No extra parameters')
        else:
            for p in self.parameter:
                tmp = "  -- " + p['name'] + ": " + normalize(p['description'])
                if 'schema' in p and 'type' in p['schema']:
                    tmp += " (type: " + p['schema']['type'] + ")"
                parameter.append(tmp)
        if '{' in self.url:
            sentence = '  -- The `{variable}` in the url path should also be replaced with actual value based on the context.'
            parameter.append(sentence)
        parameter = '\n'.join(parameter)
        return parameter

    def formulate(self, is_description=True, is_parameters=True, is_request_type=True, is_url=True, is_usage=True,
                  is_template=False, is_request_body=False):
        text_doc = ["""API name: """ + self.api_name]
        if is_url:
            text_doc.append('- API url: ' + self.url)
        if is_request_type:
            method = """- Request type: """ + self.method
            text_doc.append(method)
        if is_description:
            description = f"- Description: This API belongs to the {self.tool_name} tool of the {self.category} category. Its functionality is `{normalize(self.description)}`"
            text_doc.append(description)
        if is_parameters:
            parameters = '- Parameter:\n' + self.get_parameters()
            text_doc.append(parameters)
        if is_usage:
            if self.method.lower() == 'get':
                usage = f"""{BEGIN}
headers={self.headers}
params="<The params dict>" 
requests.get(url='{self.url}', headers=headers, params=params) # requests the tools by passing corresponding url and parameters
{END}"""
            elif self.method.lower() == 'post':
                usage = f"""{BEGIN}
headers={self.headers}
data="<The data dict>" 
requests.post(url='{self.url}', headers=headers, data=data) # requests the tools by passing corresponding url and parameters
{END}"""
            else:
                usage = ""
            if usage != '':
                text_doc.append(f"""- Usage:\n{usage}""")
        if is_template:
            response = f'- Execution result structure:\n{self.responses["template"]}'
            text_doc.append(response)
        if is_request_body:
            requestBody = '- Request body:\n' + json.dumps(self.requestBody, indent=4)
            text_doc.append(requestBody)
        text_doc = '\n'.join(text_doc)
        return text_doc


@dataclass
class OpenFunc:
    name: str
    signature: str
    docstring: str
    source: str
    library: Optional[List[str]] = ""
    instance: Optional[str] = None
    #
    category: Optional[str] = None
    tool: Optional[str] = None
    api_idx: Optional[str] = None
    response: Optional[Dict] = None

    verified: bool = False

    begin: str = BEGIN
    end: str = END

    def str(self, add_instance=False, add_response=False):
        prefix = '"""'
        func_str = f"""{self.name}{self.signature}:
    {prefix}
{self.docstring}
    {prefix}"""
        if add_instance:
            func_str += f"{self.begin}\n{self.instance}\n{self.end}"
        if add_response:
            _response = json.dumps(self.response, indent=4)
            func_str += f"{self.begin}\n{_response}\n{self.end}"

        return func_str

    def update(self, **kwargs):
        """
        :param kwargs:
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self):
        return asdict(self)

