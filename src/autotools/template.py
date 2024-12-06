from constant import BEGIN, END
from typing import List, Dict
from dataclasses import dataclass
from tools import OpenFunc

@dataclass
class PromptTemplate:
    # encap_system: str
    # encap_user: str
    begin: str = BEGIN
    end: str = END

    def initial_message_for_encap(self,  docs, headers=None):
        encap_system = f"""I have a set of customized tools. Each API has a usage in its documentation to demonstrate how to access it.
According its usage, your task is to encapsulate them into well-structured Python functions, along with a testing instance to demonstrate how to call these functions.

Your encapsulated functions should follow these key points:
1. Self-Contained: Each function must handle the API request (including making the call and processing the response) and return the result. All required constants, such as `BASE_URL` and `headers`, must be included within the function itself, rather than relying on external, global variables, or function arguments.
2. Function Flexibility: Ensure the function is flexible enough to accept necessary parameters based on the API’s requirements and return structured results (such as JSON responses). The parameters should adapt dynamically depending on the user's query.
3. Error Handling: The function should be robust enough to handle HTTP request errors. This includes checking for unsuccessful status codes or managing exceptions, and faithfully returning the error message or exceptions information. BUT do not print anything inside the function.

Here is an output template. Please include the function and the testing instance using the following format.
{self.begin}
import requests # import necessary lib

def API_NAME(PARAMs: type):
    \"\"\"
    Description: add the description of the functionality
    Args:
    - PARAM 1 (type): explain the params
    - ...
    \"\"\"
    # define the variable constants, like header or base url 
    ...
    # request get/post/...
    ...
    # Error Handling for state code
    ...
    return response

# begin your testing instance
...
{self.end}

Staring Wrap your Python code between the {self.begin} and {self.end} tags like the above template for clear illustration."""

        user = f"""Here is the detailed development documentation of an API: 
{docs}

Starting below, please encapsulate it into a well-structured Python functions and carefully design a testing instance."""
        if headers!=None:
            user+='You should use this header to access it: `headers = {headers}`'

        user+="""\nYour output:"""

        message = [
            {"role": "system", "content": encap_system},
            {"role": "user", "content": user},
        ]
        return message

    def initial_message_for_generate(self, query, funcs, docs):
        message = [
            {"role": "system", "content": self.warp_system_prompt(funcs, docs)},
            {"role": "user", "content": self.warp_user_initial_prompt(query=query)}
        ]
        return message
    def warp_system_prompt(self,funcs, docs):
        docstring = '\n'.join([f"{func}" for i, func in enumerate(funcs, 1)])
        system_prompt = f"""You are a helpful assistant assigned the task of problem-solving. You will use an interactive coding environment, specifically the interactive Python Interpreter (CPython), to complete the task.
The Python Interpreter provides a variety of real-world tools, including Web REST APIs and customized APIs. These tools have been encapsulated into Python functions that can be directly called.
{self.begin}
{docstring}
{self.end}"""
        system_prompt += '\n'

        if docs == []:
            docs = '\n'.join([f"[{i}] {doc}" for i, doc in enumerate(docs, 1)])
            system_prompt += f"""Besides, you can also call the following APIs via Python http requests. \n{docs}\n"""

        system_prompt += f"""The return values of these functions or APIs may be lengthy or complex, often in JSON format. You are encouraged to break down your coding into several code snippets and interact with the notebook **multiple turns**. At each turn, generate executable Python code to call a tool and `print` the variable's value. After receiving the execution results, extract useful intermediate results as parameters to invoke subsequent tools in the next turns.
    
Starting below, you can only give me Python code. If your need to interact with Interpreter, please enclose your code within {self.begin} and {self.end} tags to ensure the Python interpreter can recognize your code. 

Please note that: 
1. Since the Python Interpreter is Cpython, it only return what you print in your code. Thus, please always `print` the key variable you need using `print` statement.
2. If you have got the answer and do not need to interact Python Interpreter any more, please use the `print("FINISH: [PLACEHOLDER_FOR_ANSWER]")` statement as a signal to print me your answer and finish the task."""

        return system_prompt

    def warp_user_initial_prompt(self, query):
        USER = f"""Please answer my question: "{query}" using the provide functions or APIs. You should break down my question into several simpler sub-questions and interact with Python Interpreter for multiple times to get execution response using `print(variable)` statement.
During the following interaction, you should give me **Python code snippet** enclosed in {self.begin} and {self.end} and I will run your code and return your the execution results. After receiving my response, decide further calling APIs or giving the final answer."""
        return USER


    def initial_message_for_verify(self, func_str):
        # The instance contains an API-related query (like practical user's query) and the code to call the API.
        system_prompt = ("In this task, you should help me to brainstorm an instance for the provided Python function to test its correctness."
                        "The Python function has been defined in advanced as a built-in function, which can be directly used in your code.")

        user_prompt = f"""Here is the details of the function and its docstring: {self.begin}
{func_str}
{self.end}
To test its correctness, please help me to first brainstorm a function-calling instance using commonsense query and then pass required parameters to call the API. You should use Python language and enclose your instance in {self.begin} and {self.end} to ensure the code can be identified by my Python interpreter."""
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            # {"role": "assistant", "content": ""},
        ]
        return message

def register_template(
        begin,
        end
) -> PromptTemplate:
    template = PromptTemplate(begin=begin, end=end)
    return template


