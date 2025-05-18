import sys
import os
import argparse
import re
import ast
import pandas as pd
import numpy as np
from IPython.display import display, Markdown
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.append("src/utils/")
import text_processing_utils
import component_formatting_utils
from engine import get_engine, invoke_engine
from datasets import load_dataset


def separate_solution_components(engine, component_separation_prompt, formatted_math_sol):
    response = invoke_engine(engine, component_separation_prompt + "\n\n" + formatted_math_sol)
    return response

def process_thought(thought_tuple, component_separation_prompt, component_separation_model, dataset_name, subset, logs_dir):        
        engine = get_engine(model_name=component_separation_model, temperature=0.0)
        i, thought = thought_tuple
        try:
            # if os.path.exists(os.path.join(logs_dir, f"component_inline_{component_separation_model}/{dataset_name}_{subset}", f"component_inline_{i}.txt")):
            #     return True
            formatted_thought, thought_lines = (component_formatting_utils.format_solution(thought))
            component_separation_output = separate_solution_components(engine, component_separation_prompt, formatted_thought)

            solution_identifier = f"{i}"
            with open(os.path.join(logs_dir, f"component_separation_output_{component_separation_model}/{dataset_name}_{subset}", f"component_separation_{solution_identifier}.txt"), "w") as log_file:
                log_file.write(component_separation_output)

            components_list = component_formatting_utils.get_components_list(component_separation_output, thought_lines)
            component_inline_sol = (component_formatting_utils.format_solution_with_components_inline(components_list))  ## Note: Comments are not present in `formatted_sol_in_component_form`

            with open(os.path.join(logs_dir, f"component_inline_{component_separation_model}/{dataset_name}_{subset}", f"component_inline_{solution_identifier}.txt"), "w") as log_file:
                log_file.write(component_inline_sol)

            print(f"Completed component separation for {solution_identifier}")
            return True

        except Exception as e:
            print(f"Error occurred while processing thought {i}. Error: {str(e)}")
            return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run component separation on a given solution")
    parser.add_argument("--component_separation_model", type=str, required=True, help="Model to use for component separation")
    parser.add_argument("--range", type=str, required=False, default=":", help="Range of thoughts to process")
    parser.add_argument("--data_hub_path", type=str, required=True, help="Path to the dataset on hub")
    parser.add_argument("--subset", type=str, required=True, help="Subset name")
    parser.add_argument("--num_workers", type=int, required=False, help="Number of workers")
    args = parser.parse_args()

    component_separation_model = args.component_separation_model
    data_range = args.range
    data_hub_path = args.data_hub_path
    subset = args.subset
    num_workers = args.num_workers
    ds = load_dataset(data_hub_path, subset)
    ds = ds["train"]

    dataset_name = data_hub_path.split("/")[-1]

    # Create directories for logging outputs
    logs_dir = os.getenv("LOGS_DIR")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(os.path.join(logs_dir, f"component_separation_output_{component_separation_model}/{dataset_name}_{subset}"), exist_ok=True)
    os.makedirs(os.path.join(logs_dir, f"component_inline_{component_separation_model}/{dataset_name}_{subset}"), exist_ok=True)

    print(f"Using model {component_separation_model} for component separation...")
    component_separation_prompt = text_processing_utils.read_file_as_string("src/prompts/component_separation_prompt.txt")

    if "thought" not in ds.column_names:
        ds = ds.map(lambda x: {'thought':re.search(r'<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>', x["conversations"][1]["value"], re.DOTALL).group(1).strip()})
    thoughts = ds["thought"]
    
    # Extract thoughts based on the provided range
    if data_range != ":":
        thoughts = [(i, thoughts[i]) for i in 
                    range(int(data_range.split(":")[0]), int(data_range.split(":")[1]))]
        print(f"Processing thoughts in data_range: {data_range}")
    else:
        thoughts = [(i, thoughts[i]) for i in range(len(thoughts))]
        print(f"Processing all thoughts from subset {subset} of dataset {data_hub_path}")

    # for thought_tuple in thoughts: # Sequential processing
    #     process_thought(thought_tuple, engine, component_separation_model, dataset_name, subset, logs_dir)

    with ProcessPoolExecutor(max_workers=num_workers) as executor: # Parallel processing
        futures = [executor.submit(process_thought, thought_tuple, component_separation_prompt, component_separation_model, dataset_name, subset, logs_dir) for thought_tuple in thoughts]
        for future in as_completed(futures):
            future.result()

