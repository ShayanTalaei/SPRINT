import sys
sys.path.append("src/utils/")
import text_processing_utils
import evaluate_utils
import os
import re, ast
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, Value
from concurrent.futures import ThreadPoolExecutor, as_completed
from engine import get_engine, invoke_engine
import csv
import time
import langchain_core
from openai import OpenAI

def process_question(i, engine):
    for retry in range(3):
        try:
            question = ds[question_col][i]
            problem_idx = ds[problem_idx_col][i]
            true_final_answer = ds[final_answer_col][i]

            if os.path.exists(os.path.join(base_model_output_dir, f"base_model_trajectory_{problem_idx}.txt")):
                print(f"Skipping question {problem_idx} as it already exists.")
                break
            
            start_time = time.time()
            if base_model_type=="think_ft_trajectory":
                response = invoke_engine(base_model_engine, prompt = base_model_prompt+f"\n\n<Question>\n{question.strip()}\n</Question>\n\n<think>\n")
            else:
                raise Exception(f"Unknown input format for base_model_type: {base_model_type}")
            
            if type(response)==langchain_core.messages.ai.AIMessage:
                response = response.content
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            try:
                final_answer_line = re.search(r"<Final_answer>(.*?)</Final_answer>", response, re.DOTALL)
                final_answer_line = final_answer_line.group(1).strip() if final_answer_line else ""
                predicted_final_answer = evaluate_utils.remove_boxed(evaluate_utils.last_boxed_only_string(final_answer_line))
                if predicted_final_answer is not None and predicted_final_answer.strip() == "":
                    predicted_final_answer = None
            except:
                predicted_final_answer = None

            # if predicted_final_answer is None:
            #     raise Exception(f"Final answer has not been identified for problem {i}")
            
            with open(os.path.join(base_model_output_dir, "base_model_output.csv"), mode='a', newline='') as file:
                writer = csv.writer(file)
                if file.tell() == 0:  # Check if file is empty to write the header
                    writer.writerow(["ProblemIdx", "PredictedFinalAnswer", "TrueFinalAnswer", "ElapsedTime", "Question", "ModelResponse"])
                writer.writerow([problem_idx, predicted_final_answer, true_final_answer, elapsed_time, question, response])

            with open(os.path.join(base_model_output_dir, f"base_model_trajectory_{problem_idx}.txt"), "w") as f:
                f.write(response)

            print(f"Completed question {problem_idx}.")
            break
        except Exception as e:
            import traceback
            print(f"Failed to process question with problem_idx={problem_idx} with error: {e}")
            traceback.print_exc()
            print(f"Using retry {retry+1}/3")
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data in required format for training")
    parser.add_argument("--test_range", type=str, required=False, default=":", help="Number of questions to evaluate")
    parser.add_argument("--test_hub_path", type=str, required=True, help="Path to the dataset on hub")
    parser.add_argument("--test_subset", type=str, required=True, help="Subset name")
    parser.add_argument("--orchestration_split", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--base_model_server_url", type=str, required=False, default=None)
    parser.add_argument("--base_model_temp", type=float, required=True, default=0.1)
    parser.add_argument("--base_model_type", type=str, required=True, default=None)
    
    args = parser.parse_args()

    test_range = args.test_range
    test_hub_path = args.test_hub_path
    test_subset = args.test_subset
    orchestration_split = args.orchestration_split
    base_model = args.base_model
    base_model_server_url = args.base_model_server_url
    base_model_temp = args.base_model_temp
    base_model_type = args.base_model_type

    if base_model_server_url=="None":
        base_model_server_url = None
    
    ds = load_dataset(test_hub_path, test_subset)
    ds = ds[orchestration_split]

    dataset_name = test_hub_path.split("/")[-1]
    
    # Create directories for logging outputs
    logs_dir = os.getenv("LOGS_DIR")
    base_model_output_dir = os.path.join(logs_dir+"_base_model", f"base_model_output_{base_model.replace('/', '_')}/{os.path.basename(dataset_name)}_{os.path.basename(test_subset)}")
    os.makedirs(base_model_output_dir, exist_ok=True)

    if base_model_type=="think_ft_trajectory":
        base_model_prompt = text_processing_utils.read_file_as_string("src/prompts/global_agent_prompt.txt")
    else:
        raise Exception(f"Base model prompt file not specified for base_model_type: {base_model_type}.")

    print(f"Running base model evaluation with base model: {base_model} at server {base_model_server_url}")

    if test_range == ":":
        st_idx = 0
        end_idx = len(ds)
    else:
        st_idx, end_idx = test_range.split(":")
        if st_idx == "":
            st_idx = 0
        else:
            st_idx = int(st_idx)
        if end_idx == "":
            end_idx = len(ds)
        else:
            end_idx = int(end_idx)

    ds_cols = ds.column_names
    question_col = [col for col in ds_cols if col in ["Question", "question", "Problem", "problem"]]
    assert len(question_col) == 1, f"Found multiple question columns: {question_col}"
    question_col = question_col[0]

    final_answer_col = [col for col in ds_cols if col in ["FinalAnswer", "final_answer", "Answer", "answer", "result"]]
    assert len(final_answer_col) == 1, f"Found multiple final answer columns: {final_answer_col}"
    final_answer_col = final_answer_col[0]

    problem_idx_col = [col for col in ds_cols if col in ["ProblemIdx", "problemidx", "unique_id", "id"]]
    assert len(problem_idx_col) == 1, f"Found multiple problem index columns: {problem_idx_col}"
    problem_idx_col = problem_idx_col[0]
    ds = ds.cast_column(problem_idx_col, Value("string"))
    if "/" in ds[problem_idx_col][0]:
        ds = ds.map(lambda x: {problem_idx_col : os.path.splitext(x[problem_idx_col].replace("/", "_"))[0]})

    base_model_engine = get_engine(model_name=base_model, temperature=base_model_temp, inference_server_url=base_model_server_url)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_question, j, base_model_engine) for j in range(st_idx, end_idx)]
        try:
            for future in as_completed(futures, timeout=300):
                future.result()
        except Exception as e:
            print(f"Failed to process with error: {e}")
    
    print("Completed orchestration.")