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
import traceback
from transformers import AutoTokenizer

planner_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
executor_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

def remove_prompt_numbering(text):
    # Replace opening tags
    text = re.sub(r'<prompt_\d+\.\d+>', '<prompt>', text)
    # Replace closing tags
    text = re.sub(r'</prompt_\d+\.\d+>', '</prompt>', text)
    return text

def remove_prompt_from_text(plan):
    plan = re.sub(r'<prompt.*?>.*?</prompt.*?>', '', plan, flags=re.DOTALL)
    return plan.strip()

def enforce_phase_tags_on_planner_output(plan, phase, last_retry):
    # Ensure plans are numbered correctly
    plan = plan.replace("<think>", "").replace("</think>", "")
    if re.search(r'<Final_answer>', plan) or re.search(r'</Final_answer>', plan):
        plan = re.sub(r'<Final_answer.*?>', '', plan, flags=re.IGNORECASE)
        plan = re.sub(r'</Final_answer.*?>', '', plan, flags=re.IGNORECASE)
        plan = plan.strip()
        plan = f"<Final_answer>\n{plan}\n</Final_answer>"
    elif re.search(r'<Plan_\d+>', plan) or re.search(r'</Plan_\d+>', plan):
        plan = re.sub(r'<Plan_.*?>', '', plan, flags=re.IGNORECASE)
        plan = re.sub(r'</Plan_.*?>', '', plan, flags=re.IGNORECASE)
        plan = plan.strip()
        plan = f"<Plan_{phase}>\n{plan}\n</Plan_{phase}>"
    elif last_retry:
        plan = re.sub(r'<Execution.*?>', '', plan, flags=re.IGNORECASE)
        plan = re.sub(r'</Execution.*?>', '', plan, flags=re.IGNORECASE)
        plan = re.sub(r'<execution.*?>', '', plan, flags=re.IGNORECASE)
        plan = re.sub(r'</execution.*?>', '', plan, flags=re.IGNORECASE)
        plan = plan.strip()
        if plan.rfind("\\boxed")>=0:
            plan = f"<Final_answer>\n{plan}\n</Final_answer>"
        else:
            plan = f"<Plan_{phase}>\n{plan}\n</Plan_{phase}>"
    else:
        # print(plan)
        print("----------------")
        raise Exception(f"No valid planner tag found at phase {phase}")

    # Replace <prompt...> with prompt
    plan = re.sub(r'<prompt.*?>', '<prompt>', plan, flags=re.IGNORECASE)
    plan = re.sub(r'</prompt.*?>', '</prompt>', plan, flags=re.IGNORECASE)
    prompt_counter = 1

    # Replace <prompt> with numbered prompt
    while re.search(r'<prompt>', plan):
        plan = re.sub(r'<prompt>', f'<prompt_{phase}.{prompt_counter}>', plan, 1, flags=re.IGNORECASE)
        plan = re.sub(r'</prompt>', f'</prompt_{phase}.{prompt_counter}>', plan, 1, flags=re.IGNORECASE)
        prompt_counter += 1
    
    plan = plan.strip()
    return plan

def enforce_phase_tags_on_executor_output(execution, prompt_name):
    # Remove <execution...> tags if they exist
    execution = execution.replace("<think>", "").replace("</think>", "")
    execution = re.sub(r'<Execution.*?>', '', execution, flags=re.IGNORECASE)
    execution = re.sub(r'</Execution.*?>', '', execution, flags=re.IGNORECASE)
    execution = re.sub(r'<execution.*?>', '', execution, flags=re.IGNORECASE)
    execution = re.sub(r'</execution.*?>', '', execution, flags=re.IGNORECASE)

    # Add <execution_x.y> tags
    execution = execution.strip()
    execution = f"<execution_{prompt_name}>\n{execution}\n</execution_{prompt_name}>"
    return execution

def get_executor_result(past_trajectory, phase, prompt_name, executor_plan, engine, executor_model, global_prompt, problem_idx):
    for retry in range(3):
        try:
            trajectory_for_executor = past_trajectory + f"<Plan_{phase}>\n" + executor_plan + f"\n</Plan_{phase}>\n\n<Execution_{phase}>\n<execution_{phase}.1>\n"
            # print(f"Input:{prompt_name}\n"+global_prompt + "\n\n" + trajectory_for_executor+"-------------------------\n")
            execution_result = invoke_engine(engine, global_prompt + "\n\n" + trajectory_for_executor)
            # print(f"Output:{prompt_name}\n"+execution_result+"-------------------------\n")
            if re.search(r'<prompt.*?>', execution_result, re.DOTALL):
                execution_result = remove_prompt_from_text(execution_result)
            tokens = executor_tokenizer.encode(execution_result, add_special_tokens=False)
            if len(tokens) > 2000 and retry<2:
                raise Exception(f"Executor output too long ({len(tokens)} tokens) for problem {problem_idx} at prompt {prompt_name}")
            return execution_result
        except Exception as e:
            print(f"Error in execution for problem {problem_idx}: {e}. Using execution retry {retry+1}/3")
    return None

def run_parallel_executions(past_trajectory, phase, plans_per_executor, executor_model, executor_server_url, global_prompt, problem_idx):
    if "MrezaPRZ" in executor_model:
        engine = get_engine(model_name=executor_model, temperature=executor_temp, inference_server_url=executor_server_url, stop=["</Execution_", "</Plan_", "</Final_answer>", "</think>", "</execution_"])
    else:
        engine = get_engine(model_name=executor_model, temperature=executor_temp, inference_server_url=executor_server_url, model_kwargs={"stop":["</Execution_", "</Plan_", "</Final_answer>", "</think>", "</execution_"]})
    execution_results = dict()
    with ThreadPoolExecutor(max_workers=max_execution_workers) as worker:
        futures = {prompt_name:worker.submit(get_executor_result, past_trajectory, phase, prompt_name, executor_plan, engine, executor_model, global_prompt, problem_idx) for prompt_name, executor_plan in plans_per_executor}
        for prompt_name, future in futures.items():
            execution_result = future.result()
            if execution_result:
                execution_results[prompt_name] = execution_result.strip()
            else:
                print("prompt_name for execution None", prompt_name, problem_idx)
                execution_results[prompt_name] = None
    return execution_results

def replace_prompt_numbering(subplan, phase, prompt_count):
    subplan = re.sub(r'<prompt_(\d+\.\d+)>', lambda m: f'<prompt_{phase}.{prompt_count}>', subplan)
    subplan = re.sub(r'</prompt_(\d+\.\d+)>', lambda m: f'</prompt_{phase}.{prompt_count}>', subplan)
    return subplan

# def get_plans_per_executor(plan, phase):
#     if not re.search(r'<prompt.*?>', plan):
#         return []
#     plan = re.sub("<Plan_\d+>", "", plan)
#     plan = re.sub("</Plan_\d+>", "", plan)
#     plan = plan.strip()
#     plan = re.sub("</prompt.*?>", lambda m: m.group(0) + '<stop>', plan)
#     split_pattern = r'<stop>'
#     plans_per_executor = re.split(split_pattern, plan)
#     plans_per_executor = [i.strip() for i in plans_per_executor if i not in [None, "", " "]]
#     plans_per_executor = [(re.search(r'<prompt_(\d+\.\d+)>', i).group(1), i) if re.search(r'<prompt_(\d+\.\d+)>', i) else (None, i) for i in plans_per_executor]
#     rearranged_plans_per_executor = []

#     for i in range(len(plans_per_executor)):
#         prompt_name, executor_plan = plans_per_executor[i]
#         if prompt_name is None:
#             continue
#         else:
#             prompt_count = 1
#             rearranged_plan_per_executor = [replace_prompt_numbering(executor_plan, phase, prompt_count)]
#             prompt_count += 1
#             for j in range(i):
#                 rearranged_plan_per_executor.append(replace_prompt_numbering(plans_per_executor[j][1], phase, prompt_count))
#                 prompt_count += 1
#             for j in range(i+1, len(plans_per_executor)):
#                 rearranged_plan_per_executor.append(replace_prompt_numbering(plans_per_executor[j][1], phase, prompt_count))
#                 prompt_count += 1
#             rearranged_plans_per_executor.append((prompt_name, "\n\n".join(rearranged_plan_per_executor)))
#     return rearranged_plans_per_executor

def get_plans_per_executor(plan, phase):
    if not re.search(r'<prompt.*?>', plan):
        return []
    plan = re.sub("<Plan_\d+>", "", plan)
    plan = re.sub("</Plan_\d+>", "", plan)
    plan = plan.strip()
    plan = re.sub("</prompt.*?>", lambda m: m.group(0) + '<stop>', plan)
    split_pattern = r'<stop>'
    plans_per_executor = re.split(split_pattern, plan)
    plans_per_executor = [i.strip() for i in plans_per_executor if i not in [None, "", " "]]
    plans_per_executor = [(re.search(r'<prompt_(\d+\.\d+)>', i).group(1), i) if re.search(r'<prompt_(\d+\.\d+)>', i) else (None, i) for i in plans_per_executor]
    rearranged_plans_per_executor = []

    for i in range(len(plans_per_executor)):
        prompt_name, executor_plan = plans_per_executor[i]
        if prompt_name is None:
            continue
        else:
            rearranged_plan_per_executor = [remove_prompt_from_text(plans_per_executor[j][1]) for j in range(i)]
            prompt_count = 1
            rearranged_plan_per_executor.append(replace_prompt_numbering(executor_plan, phase, prompt_count))
            prompt_count += 1
            for j in range(i+1, len(plans_per_executor)):
                rearranged_plan_per_executor.append(replace_prompt_numbering(plans_per_executor[j][1], phase, prompt_count))
                prompt_count += 1
            rearranged_plans_per_executor.append((prompt_name, "\n\n".join(rearranged_plan_per_executor)))
    return rearranged_plans_per_executor

def run_agent_orchestration(question, planner_engine, problem_idx, max_phases=12):
    past_planner_trajectory = f"<Question>\n{question}\n</Question>\n\n<think>\n"
    for phase in range(1, max_phases+1):
        max_retries = 3
        for retry in range(max_retries):
            try:
                if phase==max_phases:
                    current_planner_prompt = global_prompt + "\n\n" + past_planner_trajectory + "\n\n<Final_answer>\n" 
                    # print(f"Phase {phase}:\nInput:\n"+current_planner_prompt+"\n-------------------------\n")
                    plan = "<Final_answer>\n" + invoke_engine(planner_engine, prompt = current_planner_prompt)
                else:
                    current_planner_prompt = global_prompt + "\n\n" + past_planner_trajectory
                    # print(f"Phase {phase}:\nInput:\n"+current_planner_prompt+"\n-------------------------\n")
                    plan = invoke_engine(planner_engine, prompt = current_planner_prompt)
                
                tokens = planner_tokenizer.encode(plan, add_special_tokens=False)
                if len(tokens) > planner_output_token_limit and retry<max_retries-1:
                    raise Exception(f"Planner output too long ({len(tokens)} tokens) for problem {problem_idx} at phase {phase}")
                plan = enforce_phase_tags_on_planner_output(plan, phase, last_retry=(retry==max_retries-1))
                # print(f"Phase {phase}:\nOutput:\n"+plan+"\n-------------------------\n")
                break
            except Exception as e:
                print(f"Failed to create plan for phase {phase} in problem {problem_idx} with error: {e}. Using planner retry {retry+1}/3")
        
        if re.search("<Final_answer.*?>", plan) or (re.search("\*\*Final Answer.*?", plan) and plan.rfind("\\boxed")>=0):
            past_planner_trajectory += f"{plan}\n</think>"
            return past_planner_trajectory

        current_planner_trajectory = plan + f"\n\n<Execution_{phase}>\n"

        rearranged_plans_per_executor = get_plans_per_executor(plan, phase)
        if len(rearranged_plans_per_executor)==0:
            current_planner_trajectory += "No prompt to execute."
        else:
            execution_results = run_parallel_executions(past_planner_trajectory, phase, rearranged_plans_per_executor, executor_model, executor_server_url, global_prompt, problem_idx)
            for prompt_name, executor_plan in rearranged_plans_per_executor:
                execution = enforce_phase_tags_on_executor_output(execution_results[prompt_name], prompt_name)
                current_planner_trajectory += f"{execution}\n\n" 
        
        current_planner_trajectory = current_planner_trajectory.strip()
        past_planner_trajectory += f"{current_planner_trajectory}\n</Execution_{phase}>\n\n"
    return past_planner_trajectory

def update_metrics_df(output_csv_path):
    accuracy, num_problems_attempted, num_problems_incomplete = evaluate_utils.get_eval_metrics(pd.read_csv(output_csv_path))
    metrics_df = pd.read_csv("metrics.csv")
    row_filter = ((metrics_df["PlannerModel"] == planner_model) & (metrics_df["PlannerTemp"] == planner_temp) & (metrics_df["ExecutorModel"] == executor_model) & (metrics_df["ExecutorTemp"] == executor_temp) & (metrics_df["DatasetName"] == dataset_name) & (metrics_df["SubsetName"] == test_subset))

    if metrics_df[row_filter].empty:
        metrics_df = pd.concat([metrics_df, pd.DataFrame([{"PlannerModel": planner_model, "PlannerTemp": planner_temp, "ExecutorModel": executor_model, "ExecutorTemp": executor_temp,
            "DatasetName": dataset_name, "SubsetName": test_subset, "Accuracy": accuracy, "NumProblemsAttempted": num_problems_attempted, "NumProblemsIncomplete": num_problems_incomplete}])], ignore_index=True)
    else:
        metrics_df.loc[row_filter, ["Accuracy", "NumProblemsAttempted", "NumProblemsIncomplete"]] = [accuracy, num_problems_attempted, num_problems_incomplete]
    return

def process_question(i, planner_engine):
    for retry in range(3):
        try:
            question = ds[question_col][i]
            problem_idx = ds[problem_idx_col][i]
            solution = ds[solution_col][i] if solution_col is not None else None
            true_final_answer = ds[final_answer_col][i]
            if type(true_final_answer) == list:
                true_final_answer = true_final_answer[0]

            if os.path.exists(os.path.join(agent_orchestration_output_dir, f"planner_trajectory_{problem_idx}.txt")):
                print(f"Skipping question {problem_idx} as it already exists.")
                break

            start_time = time.time()
            past_planner_trajectory = run_agent_orchestration(question, planner_engine, problem_idx, max_phases=args.max_phases)
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # final_answer_match = re.search(r'<Final_answer.*?>(.*?)</Final_answer.*?>', past_planner_trajectory, re.DOTALL)
            # predicted_final_answer = final_answer_match.group(1).strip() if final_answer_match else ""
            try:
                predicted_final_answer = evaluate_utils.remove_boxed(evaluate_utils.last_boxed_only_string(past_planner_trajectory))
            except Exception as e:
                predicted_final_answer = None
            if predicted_final_answer is not None and predicted_final_answer.strip() == "":
                predicted_final_answer = None
            if predicted_final_answer is None:
                raise Exception(f"Final answer has not been reached for problem {i}")
            
            output_csv_path = os.path.join(agent_orchestration_output_dir, "agent_orchestration_output.csv")
            with open(output_csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                if file.tell() == 0:  # Check if file is empty to write the header
                    writer.writerow(["ProblemIdx", "PredictedFinalAnswer", "TrueFinalAnswer", "ElapsedTime", "Question", "PlannerTrajectory", "Solution"])
                writer.writerow([problem_idx, predicted_final_answer, true_final_answer, elapsed_time, question, past_planner_trajectory, solution])
                
            with open(os.path.join(agent_orchestration_output_dir, f"planner_trajectory_{problem_idx}.txt"), "w") as f:
                f.write(past_planner_trajectory)
            
            # update_metrics_df(output_csv_path)
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

    parser.add_argument("--planner_model", type=str, required=True)
    parser.add_argument("--planner_server_url", type=str, required=False, default=None)
    parser.add_argument("--planner_temp", type=float, required=True, default=0.1)
    parser.add_argument("--planner_output_token_limit", type=int, required=False, default=1000, help="Maximum number of tokens allowed in planner output")

    parser.add_argument("--executor_model", type=str, required=True)
    parser.add_argument("--executor_server_url", type=str, required=False, default=None)
    parser.add_argument("--executor_temp", type=float, required=True, default=0.1)
    parser.add_argument("--max_phases", type=int, required=False, default=10)
    parser.add_argument("--max_problem_workers", type=int, required=False, default=10, help="Maximum number of workers for processing multiple problems in parallel")
    parser.add_argument("--max_execution_workers", type=int, required=False, default=5, help="Maximum number of workers for processing multiple prompts within a problem in parallel")
    
    args = parser.parse_args()

    test_range = args.test_range
    test_hub_path = args.test_hub_path
    test_subset = args.test_subset
    orchestration_split = args.orchestration_split
    planner_model = args.planner_model
    planner_server_url = args.planner_server_url
    planner_temp = args.planner_temp
    executor_model = args.executor_model
    executor_server_url = args.executor_server_url
    executor_temp = args.executor_temp
    max_phases = args.max_phases
    max_problem_workers = args.max_problem_workers
    max_execution_workers = args.max_execution_workers
    planner_output_token_limit = args.planner_output_token_limit

    if executor_server_url=="None":
        executor_server_url = None
    if planner_server_url=="None":
        planner_server_url = None
    
    ds = load_dataset(test_hub_path, test_subset)
    ds = ds[orchestration_split]

    dataset_name = test_hub_path.split("/")[-1]
    
    # Create directories for logging outputs
    logs_dir = os.getenv("LOGS_DIR")+"_full_trajectory"
    agent_orchestration_output_dir = os.path.join(logs_dir, f"agent_orchestration_output_planner_{planner_model.replace('/', '_')}_temp{planner_temp}_executor_temp{executor_temp}/{os.path.basename(dataset_name)}_{os.path.basename(test_subset)}")
    trajectory_data_dir = os.path.join(logs_dir, f"trajectory_data/{os.path.basename(dataset_name)}_{os.path.basename(test_subset)}")
    os.makedirs(agent_orchestration_output_dir, exist_ok=True)

    global_prompt = text_processing_utils.read_file_as_string("src/prompts/global_agent_prompt.txt")

    print(f"Running agent orchestration with planner model: {planner_model} at server {planner_server_url} and executor model: {executor_model} at server {executor_server_url}")

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

    solution_col = [col for col in ds_cols if col in ["solution", "Solution", "chain"]]
    if len(solution_col)==0:
        solution_col=None
    elif len(solution_col)>1:
        raise Exception(f"Found multiple solution columns: {solution_col}")
    else:
        solution_col = solution_col[0]

    final_answer_col = [col for col in ds_cols if col in ["FinalAnswer", "final_answer", "Answer", "answer", "result"]]
    assert len(final_answer_col) == 1, f"Found multiple final answer columns: {final_answer_col}"
    final_answer_col = final_answer_col[0]

    problem_idx_col = [col for col in ds_cols if col in ["ProblemIdx", "problemidx", "unique_id", "id"]]
    assert len(problem_idx_col) == 1, f"Found multiple problem index columns: {problem_idx_col}"
    problem_idx_col = problem_idx_col[0]
    ds = ds.cast_column(problem_idx_col, Value("string"))
    if "/" in ds[problem_idx_col][0]:
        ds = ds.map(lambda x: {problem_idx_col : os.path.splitext(x[problem_idx_col].replace("/", "_"))[0]})

    if "MrezaPRZ" in planner_model:
        planner_engine = get_engine(model_name=planner_model, temperature=planner_temp, inference_server_url=planner_server_url, stop=["</Execution_", "</Plan_", "</Final_answer>", "</think>", "</execution_"])
    else:
        planner_engine = get_engine(model_name=planner_model, temperature=planner_temp, inference_server_url=planner_server_url, model_kwargs={"stop":["</Execution_", "</Plan_", "</Final_answer>", "</think>", "</execution_"]})

    with ThreadPoolExecutor(max_workers=max_problem_workers) as executor:
        futures = [executor.submit(process_question, j, planner_engine) for j in range(st_idx, end_idx)]
        try:
            for future in as_completed(futures, timeout=300):
                future.result()
        except Exception as e:
            print(f"Failed to process with error: {e}")
    
    print("Completed orchestration.")