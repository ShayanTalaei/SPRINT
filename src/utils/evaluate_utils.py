import sympy
from sympy.parsing.latex import parse_latex
from math_verify import parse, verify
import re
import pandas as pd
import sys, os
sys.path.append("src/utils/")
import text_processing_utils
import trajectory_decomposition_utils
from transformers import AutoTokenizer
import numpy as np

planner_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
executor_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

def remove_punctuations_and_latex(text):
    # Define a regular expression pattern to match punctuations and LaTeX symbols
    pattern = r'[^\w\s]'
    # Use re.sub() to replace matched characters with an empty string
    cleaned_text = re.sub(pattern, '', text)
    cleaned_text = cleaned_text.replace(" ", "")
    return cleaned_text

def is_equiv(x1: str, x2: str) -> bool:
    """
    x1 and x2 are normalized latex string
    """
    if x1 == x2:
        return True
    try:
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (
                sympy.parsing.latex.errors.LaTeXParsingError,
                sympy.SympifyError,
                TypeError,
            ):
                # eval_logger.debug(f"couldn't parse one of {x1} or {x2}")
                return False

            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                # eval_logger.debug(f"couldn't subtract {x1} and {x2}")
                return False

            try:
                if sympy.simplify(diff) == 0:
                    return True
                else:
                    return False
            except ValueError:
                pass
                # eval_logger.debug(
                #     f"Had some trouble simplifying when comparing {x1} and {x2}"
                # )
    except Exception as e:
        # eval_logger.debug(f"Failed comparing {x1} and {x2} with {e}")
        return False

def use_math_verify(gold, answer):
    gold = str(gold)
    answer = str(answer)
    if gold=="" or answer=="":
        return False
    
    # Text responses may be in the form abc or \text{xyz}
    gold = re.sub(r'\\text\{([^}]*)\}', r'\1', gold)
    answer = re.sub(r'\\text\{([^}]*)\}', r'\1', answer)

    # Brackets may be in the form "(" or "\left(" AND ")" or "\right)"
    gold = re.sub(r'\\left\(', r'(', gold)
    gold = re.sub(r'\\right\)', r')', gold)
    answer = re.sub(r'\\left\(', r'(', answer)
    answer = re.sub(r'\\right\)', r')', answer)

    # Replace "_" or "," between 2 digits with "". This ensures equality between 10_000 and 10,000 and 10000
    gold = re.sub(r'(?<=\d)[_,](?=\d)', '', gold)
    answer = re.sub(r'(?<=\d)[_,](?=\d)', '', answer)
    
    if gold==answer:
        return True
    if gold.replace(" ", "")==answer.replace(" ", "") and gold.replace(" ", "")!="":
        return True
    if verify(parse(gold), parse(answer)):
        return True
    if is_equiv(gold, answer):
        return True
    gold_clean = remove_punctuations_and_latex(gold)
    answer_clean = remove_punctuations_and_latex(answer)
    if gold_clean!="" and gold_clean==answer_clean:
        return True
    return False

def get_eval_metrics(res_df):
    res_df_complete = res_df[res_df["PredictedFinalAnswer"].notna()]
    res_df_complete["IsCorrect"] = res_df_complete.apply(lambda x: use_math_verify(x["TrueFinalAnswer"], x["PredictedFinalAnswer"]), axis=1)
    accuracy = res_df_complete["IsCorrect"].sum()/len(res_df)
    num_problems_attempted = len(res_df)
    num_problems_incomplete = len(res_df) - len(res_df_complete)
    return accuracy, num_problems_attempted, num_problems_incomplete, res_df_complete

def extract_boxed_math(text):
    matches = re.findall(r"\[\n\\boxed{(.*?)}\n", text)
    if matches:
        return "Thus, the final answer is \\boxed{" + matches[-1] + "}."
    else:
        return None

def get_final_answer_line(solution):
    # solution_lines = extract_boxed_math(solution)
    # if solution_lines is not None:
    #     return solution_lines
    solution_lines = text_processing_utils.split_into_sentences(solution)
    solution_lines = [line for line in solution_lines if line.strip()!=""]
    solution_line_with_boxed = [(line_number, line) for line_number, line in enumerate(solution_lines) if "boxed" in line]
    if len(solution_line_with_boxed)>0:
        line_number, line = solution_line_with_boxed[-1]
        return "\n".join(solution_lines[line_number:]).strip()
    else:
        return "\n".join(solution_lines[-2:]).strip()
    
def last_boxed_only_string(string: str):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
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

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval

def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]

def get_max_token_len(trajectory, file_name, planner_multiplier=1):
    decomposed_parts = trajectory_decomposition_utils.decompose_trajectory(trajectory, split_plans=True, split_executions=True, include_token_count=True, 
                                            planner_tokenizer=planner_tokenizer, executor_tokenizer=executor_tokenizer)
    max_token_len = 0
    planner_contrib = 0
    exec_contrib = 0
    try:
        for i in range(0, len(decomposed_parts), 2):
            max_token_len_per_phase = 0
            
            if i!=len(decomposed_parts)-1:
                assert len(decomposed_parts[i]) == len(decomposed_parts[i+1])+1
                plan_len = 0
                for prompt_count in range(len(decomposed_parts[i])):
                    plan_len+=decomposed_parts[i][prompt_count][1]
                    exec_len = 0
                    if prompt_count<len(decomposed_parts[i+1]):
                        exec_len = decomposed_parts[i+1][prompt_count][1]
                    if (plan_len*planner_multiplier)+exec_len > max_token_len_per_phase:
                        max_token_len_per_phase = max(max_token_len_per_phase, (plan_len*planner_multiplier)+exec_len)
                        planner_contrib_per_phase = plan_len*planner_multiplier
                        exec_contrib_per_phase = exec_len
            else:
                max_token_len_per_phase = sum([decomposed_parts[i][k][1] for k in range(len(decomposed_parts[i]))])*planner_multiplier
                planner_contrib_per_phase = max_token_len_per_phase
                exec_contrib_per_phase = 0
            
            max_token_len+=max_token_len_per_phase
            planner_contrib+=planner_contrib_per_phase
            exec_contrib+=exec_contrib_per_phase
        return max_token_len, planner_contrib, exec_contrib

    except Exception as e:
        print("Invalid trajectory", file_name, e)
        return None, None, None

def get_seq_token_counts(folder_path):
    agent_orchestration_dir = folder_path
    valid_trajectory_count, invalid_trajectory_count = 0,0
    files = [i for i in os.listdir(agent_orchestration_dir) if ".txt" in i]
    files = sorted(files)
    seq_token_count_dict = dict()
    for file in files:
        trajectory = text_processing_utils.read_file_as_string(os.path.join(agent_orchestration_dir, file))
        max_token_len, planner_contrib, exec_contrib = get_max_token_len(trajectory, file, planner_multiplier=1)
        if max_token_len is not None:
            seq_token_count_dict[file] = [max_token_len, planner_contrib, exec_contrib]
            valid_trajectory_count+=1
        else:
            invalid_trajectory_count+=1
    return np.mean([seq_token_count_dict[i][0] for i in seq_token_count_dict])

def count_tokens(text, tokenizer=executor_tokenizer):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)