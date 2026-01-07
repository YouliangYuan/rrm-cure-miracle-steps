import re
import re
import signal
from . import prime_math
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
import requests
import json

def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string.
    
    Args:
        string: Input string containing LaTeX code
        
    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx:right_brace_idx + 1] if right_brace_idx is not None else None


def format_score(solution_str, good_score=0., bad_score=-1.):
    # check if </think> is in the solution str, and only appear once
    if solution_str.count("</think>") != 1:
        score = bad_score
    else:
        score = good_score

    # cot is the text before first </think>
    cot = solution_str[:solution_str.find("</think>")]
    cot = cot.strip()
    # answer is the text after last </think>
    answer = solution_str[solution_str.rfind("</think>") + len("</think>"):]
    answer = answer.strip()
    return score, cot, answer

template = """You are a grading teacher. Based on the following information, please evaluate the student's submitted answer. The total score is out of 10 points. Carefully read the question, the scoring criteria, and the submitted answer. Then give a fair and reasonable score according to the grading criteria.
Here is the information:


Question:
{{QUESTION}}



Scoring Criteria:
{{CRITERIA}}



Student's Answer:
{{MODEL_ANSWER}}

"""


# get the tinyv config for the given model
def get_tinyv_config(model_name: str):
    ans =     {
        "max_completion_tokens": 2048,
        "temperature": 0,
        "top_p": 1,
        "is_think_model": False,
        "template": template
    }

    return ans

    # with open('/a_project/CodeReasoner/TinyV-main/verl/verl/tinyv_config.json', 'r') as f:
    #     tinyv_config_all = json.load(f)

    # if model_name not in tinyv_config_all:
    #     raise KeyError(f"Model '{model_name}' not found in tinyv_config. Available models: {list(tinyv_config_all.keys())}")
    # return tinyv_config_all[model_name]


client = OpenAI(api_key='token-abc123', base_url='http://localhost:8000/v1')
# get the model name from the response
response = requests.get('http://localhost:8000/v1/models')
try:
    models_data = response.json()
    VERIFIER_MODEL_NAME = models_data['data'][0]['id']
    VERIFIER_MODEL_CONFIG = get_tinyv_config(VERIFIER_MODEL_NAME)
    TINYV_PROMPT = VERIFIER_MODEL_CONFIG['template']
    MAX_COMPLETION_TOKENS = VERIFIER_MODEL_CONFIG['max_completion_tokens']
    TEMPERATURE = VERIFIER_MODEL_CONFIG['temperature']
    TOP_P = VERIFIER_MODEL_CONFIG['top_p']
    IS_THINK_MODEL = VERIFIER_MODEL_CONFIG['is_think_model']
except Exception as e:
    print(e)
    raise Exception(f"Verifier LLM is not running. Please run the verifier first.")


def model_infer(msg, model, retry=3, temperature=0, top_p=1):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=msg,
            temperature=temperature,
            top_p=top_p,
            max_completion_tokens=MAX_COMPLETION_TOKENS
        )
        parsed_resp = resp.choices[0].message.content.strip()

        # print(parsed_resp,'\n\n👹\n\n👹\n\n👹\n\n👹\n\n👹\n\n👹\n\n👹\n\n👹')
        if 'criteria' in TINYV_PROMPT.lower():
            # print("HERE IS THE grade: ", parsed_resp)

            ans = parsed_resp.split('Total Score')[-1].replace(':', '').replace('[', '').replace('points', '').replace(
                'point', '').replace(
                '*', '').replace('|', '').replace('\'', '')
            ans = ans.split('/')[0].strip()
            score = min(10.0, float(ans))/10
        else:
            # For think verifiers, only consider the answer part
            if IS_THINK_MODEL:
                if "Answer:" in parsed_resp: # Qwen2.5
                    parsed_resp = parsed_resp.split("Answer:")[1].strip()
                elif "</think>" in parsed_resp: # Qwen3
                    parsed_resp = parsed_resp.split("</think>")[1].strip()
                else:
                    print(f"Invalid response: {parsed_resp}")
                    score = 0
                    return score

            if 'true' in parsed_resp.lower():
                score = 1
            elif 'false' in parsed_resp.lower():
                score = 0
            else:
                print(f"Invalid response: {parsed_resp}")
                score = 0

        return score

    except Exception as e:
        # print(f"LLM Verifier InferenceError: {e}")
        if retry > 0:
            # In case the model is not working, we try again with a higher temperature
            return model_infer(msg, model, retry=retry-1, temperature=TEMPERATURE+0.3, top_p=1)
        else:
            return 0



#   "You will receive a question and its response, and you need to score the response according to the given criteria. \\n## Question\\n{{QUESTION}}\\n \\n## Response\\n{{{MODEL_ANSWER}}\\n \\n## Criteria\\n{{CRITERIA}}"
def tinyv_score(question_str: str, ground_truth: str, model_answer: str, criteria=None, debug=False):
    global client

    if criteria is None:
        msg = [
            {"role": "user",
             "content": TINYV_PROMPT.replace("{{QUESTION}}", question_str).replace("{{GROUND_TRUTH_ANSWER}}",
                                                                                   ground_truth).replace(
                 "{{MODEL_ANSWER}}", model_answer)}
        ]
    else:
        msg = [
            {"role": "user",
             "content": TINYV_PROMPT.replace("{{QUESTION}}", question_str).replace("{{MODEL_ANSWER}}", model_answer).replace("{{CRITERIA}}", criteria)}
        ]
    if debug:
        print(f"TinyV Prompt: {msg}")

    tinyv_score = model_infer(msg, VERIFIER_MODEL_NAME, retry=3, temperature=TEMPERATURE, top_p=TOP_P)

    return tinyv_score

def _compute_score(solution_str:str, ground_truth:str, question_str:str, criteria=None, tinyv_setup=None, tinyv_weight=None, debug=False):

    if  tinyv_setup == 'tinyv_only':
        tinyv_reward = tinyv_score(question_str, ground_truth, solution_str, criteria)
        score = tinyv_reward

    # Zhangchen: We don't consider format reward for now for consistency with prime
    elif tinyv_setup == 'addon':
        prime_is_correct, prime_format_correctness, answer = prime_math.compute_score(solution_str, ground_truth)
        if prime_is_correct == False:
            tinyv_reward = tinyv_score(question_str, ground_truth, answer,criteria) * tinyv_weight
        else:
            tinyv_reward = 1
        score = tinyv_reward
    elif tinyv_setup == 'mixed':
        # print('use mix mode')
        prime_is_correct, prime_format_correctness, answer = prime_math.compute_score(solution_str, ground_truth)
        tinyv_reward = tinyv_score(question_str, ground_truth, answer, criteria)
        score = tinyv_reward * tinyv_weight + prime_is_correct * (1 - tinyv_weight)
    else:
        raise ValueError(f"Invalid tinyv_setup: {tinyv_setup}")

    if debug:
        print(f"Question: {question_str}")
        print(f"TinyV Setup:             : {tinyv_setup}")
        print(f"Prime Correctness        : {prime_is_correct}")
        print(f"Prime Format Correctness : {prime_format_correctness}")
        print(f"Model Answer             : {answer}")
        print(f"Ground Truth Answer      : {ground_truth}")
        print(f"Assigned tinyv Reward    : {tinyv_reward}")
        print(f"Assigned score           : {score}")
        print("-"*80)

    return score

def find_repeated_substrings(text, min_length=300):
    seen = dict()
    duplicates = set()

    # 使用滑动窗口的方法遍历所有子串
    text_length = len(text)
    for start in range(0, text_length - min_length + 1):
        substr = text[start:start + min_length]
        if substr in seen:
            duplicates.add(substr)
        else:
            seen[substr] = start

    return len(list(duplicates))!=0


import re


def is_complete_response(response: str, verbose: bool = False) -> bool:

    if not response or not isinstance(response, str):
        # 空或非字符串输入，我们视其为“完整”的空响应
        return True

    # --- 1. 预处理 ---
    stripped_response = response.strip()
    if not stripped_response:
        return True

    last_char = stripped_response[-1]
    if last_char.isalpha() or last_char.isdigit():
        if verbose:
            print(f"异常：响应以字母或数字 '{last_char}' 结尾，可能被截断。")
        return False


    return True


def validate_response_structure(processed_str: str) -> bool:

    pattern = re.compile(r'<reasoning>.*</reasoning>.*<solution>.*</solution>.*$', re.DOTALL)
    return bool(pattern.match(processed_str.strip()))



def compute_score(solution_str, ground_truth, extra_info=None, tinyv_setup=None, tinyv_weight=None, format_reward_max=0., tinyv_reward_max=1.):
    question_str = extra_info['question']
    criteria = extra_info['criteria']

    # if '</think>' not in solution_str:
    #     return -1
    # solution_str = solution_str.split('</think>')[-1]

    # is_think_format = validate_response_structure(solution_str)
    # if not is_think_format:
    #     return 0
    # # print('BEFORE', solution_str)
    # solution_str = solution_str.split('</reasoning>')[-1]
    # if last_boxed_only_string(solution_str) is None:
    #     return 0
    # print('AFTER', solution_str)
    # if 'assistantassistantassistant' in solution_str.lower() or len(solution_str.lower().split('assistant')) >= 5:
    #     return -1.0
    # if len(solution_str.lower().split()) <= 10:
    #     return 0.0

    # if not is_complete_response(solution_str, verbose=False) and find_repeated_substrings(solution_str, min_length=300):
    #     return -1.0

    # if last_boxed_only_string(solution_str) is None:
    #     return -1.0


    score = _compute_score(solution_str, ground_truth, question_str, criteria, tinyv_setup, tinyv_weight)

    return score


