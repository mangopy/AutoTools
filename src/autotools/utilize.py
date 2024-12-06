import sys
import pickle
import multiprocessing
import json
import os
from tqdm import tqdm
import re



def standardize(string):
    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_]")
    string = res.sub("_", string)
    string = re.sub(r"(_)\1+", "_", string).lower()
    while True:
        if len(string) == 0:
            return string
        if string[0] == "_":
            string = string[1:]
        else:
            break
    while True:
        if len(string) == 0:
            return string
        if string[-1] == "_":
            string = string[:-1]
        else:
            break
    if string[0].isdigit():
        string = "get_" + string
    return string


def change_name(name):
    change_list = ["from", "class", "return", "false", "true", "id", "and"]
    if name in change_list:
        name = "is_" + name
    return name


def multi_load_jsonl(filename,num_processes=10):
    """

    :param filename: the jsonl file with big size
    :param num_processes:
    :return:
    """
    with open(filename,'r',encoding='utf-8') as f:
        data=[line.strip() for line in f]
        if len(data)<=20000:

            _,data=load_jsonl(0,data)
            return data

    length = len(data) // num_processes + 1
    pool=multiprocessing.Pool(processes=num_processes)
    collects=[]
    for ids in range(num_processes):
        collect = data[ids * length:(ids + 1) * length]
        collects.append(pool.apply_async(load_jsonl,(ids,collect)))

    pool.close()
    pool.join()
    results=[]
    for i,result in enumerate(collects):
        ids,res=result.get()
        assert ids==i
        results.extend(res)
    return results

def load_jsonl(ids,data):
    data=[json.loads(line) for line in tqdm(data)]
    return ids,data

def write_file(data,filename,num_processes=20,default_name='train',indent=4):
    if filename.endswith('.json'):
        json.dump(data,open(filename,'w'),indent=indent)
    elif filename.endswith('.jsonl') :
        with open(filename, 'w') as f:
            for line in data:
                f.write(json.dumps(line) + '\n')
    elif filename.endswith('.txt'):
        with open(filename, 'w') as f:
            for line in data:
                f.write(str(line) + '\n')
    elif filename.endswith('.pkl'):
        pickle.dump(data,open(filename,'w'))
    elif '.' not in filename:
        multi_write_jsonl(data,filename,num_processes=num_processes,default_name=default_name)
    else:
        raise "no suitable function to write data"

def write_jsonl(data,filename,ids=None):
    with open(filename,'w') as f:
        for line in tqdm(data):
            f.write(json.dumps(line)+'\n')
    return ids,len(data)

def multi_write_jsonl(data,folder,num_processes=10,default_name='train'):
    """

    :param filename:
    :param num_processes:
    :return:
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    length = len(data) // num_processes + 1
    pool=multiprocessing.Pool(processes=num_processes)
    collects=[]
    for ids in range(num_processes):
        filename=os.path.join(folder,f"{default_name}{ids}.jsonl")
        collect = data[ids * length:(ids + 1) * length]
        collects.append(pool.apply_async(write_jsonl,(collect,filename,ids)))

    pool.close()
    pool.join()
    cnt=0
    for i,result in enumerate(collects):
        ids,num=result.get()
        assert ids==i
        cnt+=num
    return cnt

def load_data(filename,num_processes=10):
    if filename.endswith('.jsonl'):
        return multi_load_jsonl(filename,num_processes)
    elif filename.endswith('.json'):
        return json.load(open(filename,'r'))
    elif filename.endswith('.pkl'):
        return pickle.load(filename)
    elif filename.endswith('.txt'):
        with open(filename,'r') as f:
            data=[line.strip() for line in f]
            return data
    else:
        raise "no suitable function to load data"

