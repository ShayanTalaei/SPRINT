import sys
import os
import argparse
from openai import OpenAI
import re
import ast
import json
from IPython.display import display, Markdown
import networkx as nx
import matplotlib.pyplot as plt
import graphviz
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.append("src/utils/")
import text_processing_utils
from engine import get_engine, invoke_engine


def extract_python_code(text):
    pattern = re.compile(r'```python(.*?)```', re.DOTALL)
    matches = pattern.findall(text)
    return matches[-1].strip() # String is expected to only contain one match. Otherwise, use the last match.

def create_dag(engine, dag_creation_prompt, component_inline_sol):
    response = invoke_engine(engine, dag_creation_prompt + "\n\n" + component_inline_sol)
    return response

# Verify that there are no cycles - for a key, the values must be smaller than it (since input solution is sequential)
def verify_dag_property(parent_dictionary):
    for node in parent_dictionary:
        for parent in parent_dictionary[node]:
            if parent >= node:
                return False
    return True

def process_thought_to_get_dag(file_names_tuple, component_inline_dir, dag_creation_prompt, model, logs_dir, dataset_name, subset):
        i, component_inline_file_name = file_names_tuple
        
        try:
            engine = get_engine(model_name=model, temperature=0.0)
            solution_identifier = f"{i}"
            component_inline_full_path = os.path.join(component_inline_dir, component_inline_file_name)            
            component_inline_sol = text_processing_utils.read_file_as_string(component_inline_full_path)
            dag_creation_output = create_dag(engine, dag_creation_prompt, component_inline_sol)

            with open(os.path.join(logs_dir, f"dag_creation_output_{model}/{dataset_name}_{subset}", f"dag_creation_{solution_identifier}.txt"), "w") as log_file:
                log_file.write(dag_creation_output)

            parent_dictionary = extract_python_code(dag_creation_output)
            parent_dictionary = ast.literal_eval(parent_dictionary.split("=", 1)[1].strip())
            parent_dictionary = {int(k): [int(i) for i in v] for k, v in parent_dictionary.items()}
            assert verify_dag_property(parent_dictionary)

            with open(os.path.join(logs_dir, f"created_dags_{model}/{dataset_name}_{subset}", f"created_dag_{solution_identifier}.json"), "w") as log_file:
                json.dump(parent_dictionary, log_file)

            dot = graphviz.Digraph()

            for node, parents in parent_dictionary.items():
                dot.node(str(node))
                for parent in parents:
                    dot.edge(str(parent), str(node))

            # dot.render(os.path.join(logs_dir, f"dags_viz_{model}/{dataset_name}_{subset}",  f"dag_{solution_identifier}"), format='png', cleanup=True)

            print(f"Completed DAG creation for {os.path.basename(solution_identifier)}")
            return True
        
        except Exception as e:
            print(f"Error processing {component_inline_file_name}: {e}")
            return False

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description='Run DAG creation on a thought formatted with component breakdown')
    parser.add_argument('--dag_creation_model', type=str, required=False, help='Model to use for DAG creation')
    parser.add_argument('--component_separation_model', type=str, required=False, help='Model to use for component separation')
    parser.add_argument(
        "--range",
        type=str,
        required=False,
        default=":",
        help="Range of thoughts to process",
    )
    parser.add_argument("--num_workers", type=int, required=False, help="Number of workers")
    parser.add_argument("--data_hub_path", type=str, required=False, help="Path to the dataset on hub")
    parser.add_argument("--subset", type=str, required=False, help="Subset name")
    args = parser.parse_args()

    data_hub_path = args.data_hub_path
    subset = args.subset
    component_separation_model = args.component_separation_model
    dag_creation_model = args.dag_creation_model
    data_range = args.range
    num_workers = args.num_workers

    dataset_name = data_hub_path.split("/")[-1]

    # Create directories for logging outputs
    logs_dir = os.getenv("LOGS_DIR")
    component_inline_dir = f"{logs_dir}/component_inline_{component_separation_model}/{dataset_name}_{subset}"
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(os.path.join(logs_dir, f"dag_creation_output_{dag_creation_model}/{dataset_name}_{subset}"), exist_ok=True)
    os.makedirs(os.path.join(logs_dir, f"created_dags_{dag_creation_model}/{dataset_name}_{subset}"), exist_ok=True)
    # os.makedirs(os.path.join(logs_dir, f"dags_viz_{dag_creation_model}/{dataset_name}_{subset}"), exist_ok=True)

    print(f"Using model {dag_creation_model} for DAG creation...")
    dag_creation_prompt = text_processing_utils.read_file_as_string("src/prompts/dag_creation_prompt.txt")

    if data_range == ":":
        component_inline_file_names = [(int(os.path.splitext(component_inline_filename)[0].split("_")[-1]), component_inline_filename) for component_inline_filename in sorted(os.listdir(component_inline_dir))]
        print(f"Creating DAGs from all inline components in {component_inline_dir}")
    else:
        component_inline_file_names = [(i, f"component_inline_{i}.txt") for i in range(int(data_range.split(":")[0]), int(data_range.split(":")[1]))]
        print(f"Creating DAGs from inline components in data_range: {data_range}")
    
    solution_identifiers = [i[0] for i in component_inline_file_names]
    if len(set(solution_identifiers)) != len(solution_identifiers):
        raise ValueError("Solution identifiers are not unique!")
    print(f"Creating DAG's for thoughts from index {min(solution_identifiers)} to {max(solution_identifiers)}")

    # for file_names_tuple in component_inline_file_names: # Sequential processing
    #     process_thought_to_get_dag(file_names_tuple, component_inline_dir, dag_creation_prompt, model, logs_dir, dataset_subset_name) 

    with ProcessPoolExecutor(max_workers=num_workers) as executor: # Parallel processing
        futures = [executor.submit(process_thought_to_get_dag, file_names_tuple, component_inline_dir, dag_creation_prompt, dag_creation_model, logs_dir, dataset_name, subset) 
                  for file_names_tuple in component_inline_file_names]
        for future in as_completed(futures):
            future.result()