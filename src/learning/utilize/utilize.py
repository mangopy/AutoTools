import sys
sys.path.append('../../')
import pickle
import multiprocessing
import math
import json
import copy
import os
import re
import string
import regex
import unicodedata
from tqdm import tqdm
import logging

logger = logging.getLogger()

def calculate_probability(cumulative_logprob):
    """"""
    probability = math.exp(cumulative_logprob)
    return probability


def str_index_docs(text, candidates, start=1):
    rank = extract_numbers_from_ordered_brackets(text)
    rank = [candidates[i-start] for i in rank if i<=len(candidates)]
    return rank

def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response

def extract_numbers_from_ordered_brackets(text: str):
    new_response = ''
    for c in text:
        if not c.isdigit():
            new_response += ' '
        else:
            new_response += c
    new_response = new_response.strip()
    new_response = [int(x) for x in new_response.split()]
    return remove_duplicate(new_response)

def mean(li, r=4):
    return round((sum(li)) / (len(li) + 0.0001), r)

def multi_load_jsonl(filename, num_processes=5):
    """

    :param filename: the jsonl file with big size
    :param num_processes:
    :return:
    """
    with open(filename, 'r', encoding='utf-8') as f:
        data = [line.strip() for line in f]
        if len(data) <= 20000:
            _, data = load_jsonl(0, data)
            return data
    multiprocessing.freeze_support()
    length = len(data) // num_processes + 1
    pool = multiprocessing.Pool(processes=num_processes)
    collects = []
    for ids in range(num_processes):
        collect = data[ids * length:(ids + 1) * length]
        collects.append(pool.apply_async(load_jsonl, (ids, collect)))

    pool.close()
    pool.join()
    results = []
    for i, result in enumerate(collects):
        ids, res = result.get()
        assert ids == i
        results.extend(res)
    # print(f"*************************** total {len(results)}  examples ****************************")
    return results


def load_jsonl(ids, data):
    data = [json.loads(line) for line in (data)]
    return ids, data


def write_file(data, filename, num_processes=20, default_name='train', indent=4):
    if filename is None:
        print(f"targeted file is None")
        return False
    print(f"================== begin to write data to {filename} ==================")
    if filename.endswith('.json'):
        json.dump(data, open(filename, 'w'), indent=indent, ensure_ascii=False)
    elif filename.endswith('.jsonl'):
        with open(filename, 'w') as f:
            for line in data:
                f.write(json.dumps(line, ensure_ascii=False) + '\n')
    elif filename.endswith('.txt'):
        with open(filename, 'w') as f:
            for line in data:
                f.write(str(line) + '\n')
    elif filename.endswith('.pkl'):
        pickle.dump(data, open(filename, 'wb'))
    elif '.' not in filename:
        multi_write_jsonl(data, filename, num_processes=num_processes, default_name=default_name)
    else:
        raise "no suitable function to write data"
    print(f"================== totally {len(data)} writing data to {filename} ==================")
    return True


def write_jsonl(data, filename, ids=None):
    with open(filename, 'w') as f:
        for line in tqdm(data):
            f.write(json.dumps(line) + '\n')
    return ids, len(data)


def multi_write_jsonl(data, folder, num_processes=10, default_name='train'):
    """

    :param filename:
    :param num_processes:
    :return:
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    length = len(data) // num_processes + 1
    pool = multiprocessing.Pool(processes=num_processes)
    collects = []
    for ids in range(num_processes):
        filename = os.path.join(folder, f"{default_name}{ids}.jsonl")
        collect = data[ids * length:(ids + 1) * length]
        collects.append(pool.apply_async(write_jsonl, (collect, filename, ids)))

    pool.close()
    pool.join()
    cnt = 0
    for i, result in enumerate(collects):
        ids, num = result.get()
        assert ids == i
        cnt += num
    print(f"** total {cnt}  examples have been writen to {folder} **")
    return cnt


def load_data(filename, num_processes=10):
    print(f"************************** begin to load data of {filename} *******************************")
    if filename.endswith('.jsonl'):
        return multi_load_jsonl(filename, num_processes)
    elif filename.endswith('.json'):
        return json.load(open(filename, 'r'))
    elif filename.endswith('.pkl'):
        return pickle.load(filename)
    elif filename.endswith('.txt'):
        with open(filename, 'r') as f:
            data = [line.strip() for line in f]
            return data
    else:
        raise "no suitable function to load data"
    print(f"************************** end load data of {filename} *******************************")


def fix_json_error(data: str, return_str=True):
    data = data.strip().strip('"').strip(",").strip("`")
    try:
        json.loads(data)
        return data
    except json.decoder.JSONDecodeError:
        data = data.split("\n")
        data = [line.strip() for line in data]
        for i in range(len(data)):
            line = data[i]
            if line in ['[', ']', '{', '}']:
                continue
            if line.endswith(('[', ']', '{', '}')):
                continue
            if not line.endswith(',') and data[i + 1] not in [']', '}', '],', '},']:
                data[i] += ','
            if data[i + 1] in [']', '}', '],', '},'] and line.endswith(','):
                data[i] = line[:-1]
        data = " ".join(data)

        if not return_str:
            data = json.loads(data)
        return data



def has_answer(answers, text, match_type="string"):
    class Tokens(object):
        """A class to represent a list of tokenized text."""
        TEXT = 0
        TEXT_WS = 1
        SPAN = 2
        POS = 3
        LEMMA = 4
        NER = 5

        def __init__(self, data, annotators, opts=None):
            self.data = data
            self.annotators = annotators
            self.opts = opts or {}

        def __len__(self):
            """The number of tokens."""
            return len(self.data)

        def slice(self, i=None, j=None):
            """Return a view of the list of tokens from [i, j)."""
            new_tokens = copy.copy(self)
            new_tokens.data = self.data[i: j]
            return new_tokens

        def untokenize(self):
            """Returns the original text (with whitespace reinserted)."""
            return ''.join([t[self.TEXT_WS] for t in self.data]).strip()

        def words(self, uncased=False):
            """Returns a list of the text of each token
            Args:
                uncased: lower cases text
            """
            if uncased:
                return [t[self.TEXT].lower() for t in self.data]
            else:
                return [t[self.TEXT] for t in self.data]

        def offsets(self):
            """Returns a list of [start, end) character offsets of each token."""
            return [t[self.SPAN] for t in self.data]

        def pos(self):
            """Returns a list of part-of-speech tags of each token.
            Returns None if this annotation was not included.
            """
            if 'pos' not in self.annotators:
                return None
            return [t[self.POS] for t in self.data]

        def lemmas(self):
            """Returns a list of the lemmatized text of each token.
            Returns None if this annotation was not included.
            """
            if 'lemma' not in self.annotators:
                return None
            return [t[self.LEMMA] for t in self.data]

        def entities(self):
            """Returns a list of named-entity-recognition tags of each token.
            Returns None if this annotation was not included.
            """
            if 'ner' not in self.annotators:
                return None
            return [t[self.NER] for t in self.data]

        def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
            """Returns a list of all ngrams from length 1 to n.
            Args:
                n: upper limit of ngram length
                uncased: lower cases text
                filter_fn: user function that takes in an ngram list and returns
                True or False to keep or not keep the ngram
                as_string: return the ngram as a string vs list
            """

            def _skip(gram):
                if not filter_fn:
                    return False
                return filter_fn(gram)

            words = self.words(uncased)
            ngrams = [(s, e + 1)
                    for s in range(len(words))
                    for e in range(s, min(s + n, len(words)))
                    if not _skip(words[s:e + 1])]

            # Concatenate into strings
            if as_strings:
                ngrams = ['{}'.format(' '.join(words[s:e])) for (s, e) in ngrams]

            return ngrams

        def entity_groups(self):
            """Group consecutive entity tokens with the same NER tag."""
            entities = self.entities()
            if not entities:
                return None
            non_ent = self.opts.get('non_ent', 'O')
            groups = []
            idx = 0
            while idx < len(entities):
                ner_tag = entities[idx]
                # Check for entity tag
                if ner_tag != non_ent:
                    # Chomp the sequence
                    start = idx
                    while (idx < len(entities) and entities[idx] == ner_tag):
                        idx += 1
                    groups.append((self.slice(start, idx).untokenize(), ner_tag))
                else:
                    idx += 1
            return groups


    class Tokenizer(object):
        """Base tokenizer class.
        Tokenizers implement tokenize, which should return a Tokens class.
        """

        def tokenize(self, text):
            raise NotImplementedError

        def shutdown(self):
            pass

        def __del__(self):
            self.shutdown()


    class SimpleTokenizer(Tokenizer):
        ALPHA_NUM = r'[\p{L}\p{N}\p{M}]+'
        NON_WS = r'[^\p{Z}\p{C}]'

        def __init__(self, **kwargs):
            """
            Args:
                annotators: None or empty set (only tokenizes).
            """
            self._regexp = regex.compile(
                '(%s)|(%s)' % (self.ALPHA_NUM, self.NON_WS),
                flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE
            )
            if len(kwargs.get('annotators', {})) > 0:
                logger.warning('%s only tokenizes! Skipping annotators: %s' %
                            (type(self).__name__, kwargs.get('annotators')))
            self.annotators = set()

        def tokenize(self, text):
            data = []
            matches = [m for m in self._regexp.finditer(text)]
            for i in range(len(matches)):
                # Get text
                token = matches[i].group()

                # Get whitespace
                span = matches[i].span()
                start_ws = span[0]
                if i + 1 < len(matches):
                    end_ws = matches[i + 1].span()[0]
                else:
                    end_ws = span[1]

                # Format data
                data.append((
                    token,
                    text[start_ws: end_ws],
                    span,
                ))
            return Tokens(data, self.annotators)

    tokenizer = SimpleTokenizer()
    text = unicodedata.normalize('NFD', text)
    if match_type == 'string':
        text = tokenizer.tokenize(text).words(uncased=True)
        for single_answer in answers:
            single_answer = unicodedata.normalize('NFD', single_answer)
            single_answer = tokenizer.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)
            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i: i+ len(single_answer)]:
                    return 1
    return 0

