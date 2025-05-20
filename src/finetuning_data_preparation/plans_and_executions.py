import sys
import os
import shutil
import argparse
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import networkx as nx
import pandas as pd
import graphviz
import pickle
from datasets import load_dataset, DatasetDict, Dataset
import numpy as np
sys.path.append("src/utils/")
import text_processing_utils
import component_formatting_utils
import evaluate_utils
from dotenv import load_dotenv
np.random.seed(42)

def merge_data(data1, data2):
    if data1 is None:
        return data2
    if data2 is None:
        return data1
    if type(data1) == list and type(data2) == list:
        return data1 + data2
    return data1 + "\n" + data2

def merge_nodes(G, node1, node2):
    # Combine the component data
    data1 = G.nodes[node1]['component_data']
    data2 = G.nodes[node2]['component_data']
    
    merged_data = {
        'Component_source': data1['Component_source'] + data2['Component_source'],
        'Description': merge_data(data1['Description'], data2['Description']),
        'Plan_lines': merge_data(data1['Plan_lines'], data2['Plan_lines']),
        'Prompt': data2['Prompt'],
        'Execution_lines': merge_data(data1['Execution_lines'], data2['Execution_lines']),
        'Comment_lines': merge_data(data1['Comment_lines'], data2['Comment_lines']),
        'Plan': merge_data(data1['Plan'], data2['Plan']),
        'Execution': merge_data(data1['Execution'], data2['Execution']),
        'Comment': merge_data(data1['Comment'], data2['Comment'])
    }
    
    # Add edges from node1's predecessors to node2
    for pred in G.predecessors(node1):
        G.add_edge(pred, node2)
    
    # Update node2's component data
    G.nodes[node2]['component_data'] = merged_data
    
    # Remove node1
    G.remove_node(node1)

def merge_plan_with_execution_for_short_executions(G, exec_len_threshold):
    for node in G.nodes():
        if G.nodes[node]['component_data']['Execution'] in ['', None]:
            G.nodes[node]['component_data']['Execution_lines'] = None
            continue
        if G.nodes[node]['component_data']['Execution_lines'][1]-G.nodes[node]['component_data']['Execution_lines'][0]<exec_len_threshold:
            G.nodes[node]['component_data']['Plan'] = merge_data(G.nodes[node]['component_data']['Plan'], G.nodes[node]['component_data']['Execution'])
            G.nodes[node]['component_data']['Plan_lines'] = merge_data(G.nodes[node]['component_data']['Plan_lines'], G.nodes[node]['component_data']['Execution_lines'])
            G.nodes[node]['component_data']['Execution'] = None
            G.nodes[node]['component_data']['Execution_lines'] = None

def is_node_with_empty_execution(G, node):
    # Check if node has empty execution
    node_data = G.nodes[node]['component_data']
    return node_data['Execution'] == '' or node_data['Execution'] == None

def get_node_depth(G, node, phase_of_node):
    parents = list(G.predecessors(node))
    if len(parents) == 0:
        return 1
    if any(parent not in phase_of_node.keys() for parent in parents):
        return -1
    node_depth = 1
    for parent in parents:
        if is_node_with_empty_execution(G, parent): # If the node has a plan but no execution, assign it to the same phase as the parent. If there are multiple parents, assign it to the phase of the parent with the highest phase.
            node_depth = max(node_depth, phase_of_node[parent])
        else:
            node_depth = max(node_depth, phase_of_node[parent] + 1)
    return node_depth

def get_node_parents(G, node, phase_of_node):
    # Get parents of node that are at an earlier phase (not same phase) and which have execution
    parents = list(G.predecessors(node))
    phase_parents_with_execution = []
    phase_parents_without_execution = []
    same_phase_parents = []
    for parent in parents:
        if phase_of_node[parent]<phase_of_node[node] and not is_node_with_empty_execution(G, parent):
            phase_parents_with_execution.append(parent)
        elif phase_of_node[parent]<phase_of_node[node] and is_node_with_empty_execution(G, parent):
            phase_parents_without_execution.append(parent)
        else:
            same_phase_parents.append(parent)
    return phase_parents_with_execution, phase_parents_without_execution, same_phase_parents

def get_phase_dict(G):
    phase_dict = defaultdict(list)
    visited = set()
    phase_of_node = dict()
    queue = [(node, 1) for node in G.nodes() if len(list(G.predecessors(node))) == 0]  # (node, depth) for all nodes with no parents
    queue.sort(key=lambda x: x[0])  # Sort by node number
    
    while queue:
        node, current_depth = queue.pop(0)
        if node in visited:
            continue
        phase_dict[current_depth].append(node)
        phase_of_node[node] = current_depth
        visited.add(node)

        successors = sorted(G.successors(node))
        for successor in successors:
            if successor not in visited:
                successor_depth = get_node_depth(G, successor, phase_of_node)
                if successor_depth != -1:
                    queue.append((successor, successor_depth))
    
    # Sort each list in phase_dict
    for phase in phase_dict:
        phase_dict[phase].sort()
    return dict(phase_dict), phase_of_node

def assign_prompt_names(G, phase_dict):
    for phase in sorted(phase_dict.keys()):
        nodes = phase_dict[phase]
        prompt_idx = 1
        for node in nodes:
            if is_node_with_empty_execution(G, node):
                continue
            prompt_name = f"{phase}.{prompt_idx}"
            G.nodes[node]['component_data']['prompt_name'] = prompt_name
            prompt_idx += 1

def create_trajectory(question, phase_dict, phase_of_node, components_list, dag, final_answer_line):
    trajectory = f"""<Question>\n{question}\n</Question>\n"""
    for phase in sorted(phase_dict.keys()):
        nodes = phase_dict[phase]
        plans = ""
        executions = "" 
        prev_phase_parents_with_execution_names = None
        for i in range(len(nodes)):
            component_dict = dag.nodes[nodes[i]]['component_data']
            phase_parents_with_execution, _, _ = get_node_parents(dag, nodes[i], phase_of_node)
            phase_parents_with_execution_names = [dag.nodes[parent]['component_data']['prompt_name'] for parent in phase_parents_with_execution]
            
            plan = component_dict["Plan"]
            prompt = component_dict["Prompt"]
            comment = component_dict["Comment"]

            phase_parent_with_execution_comments = {dag.nodes[parent]['component_data']['prompt_name']:components_list[parent]["Comment"] for parent 
                                                    in phase_parents_with_execution if components_list[parent]["Comment"] is not None}
            comments = ""
            for parent_name, parent_comment in phase_parent_with_execution_comments.items():
                comments += f"Observation about execution_{parent_name}: {parent_comment}\n"

            plans+=comments

            if i>0 and nodes[i]!=nodes[i-1]+1:
                plans += "-----\n"
            plans += f"\n"
            if (prev_phase_parents_with_execution_names is None or prev_phase_parents_with_execution_names!=phase_parents_with_execution_names) and len(phase_parents_with_execution_names)>0:
                plans += f"Based on "+', '.join([f'execution_{parent_name}' for parent_name in phase_parents_with_execution_names]) + ":\n"
            plans += f"{plan}\n"
            
            if is_node_with_empty_execution(dag, nodes[i]):
                if comment is not None:
                    plans += f"{comment}\n"
            else:
                execution = component_dict["Execution"]
                prompt_name = component_dict["prompt_name"]
                plans += f"\n<prompt_{prompt_name}> {prompt} </prompt_{prompt_name}>\n"
                executions += f"\n<execution_{prompt_name}>\n{execution}\n</execution_{prompt_name}>\n"

            if len(phase_parents_with_execution_names)>0:
                prev_phase_parents_with_execution_names = phase_parents_with_execution_names

        if executions.strip()=="":
            executions = "No prompt to execute."       
        phase_text = f"""\n<Plan_{phase}>
{plans.strip()}
</Plan_{phase}>

<Execution_{phase}>
{executions.strip()}
</Execution_{phase}>
"""
        trajectory+=phase_text
        
    trajectory += f"\n<Final_answer>\n{final_answer_line}\n</Final_answer>"  
    return trajectory

def visualize_phase_dag(dag, phase_of_node, show_all_edges=False):
    dot = graphviz.Digraph(format="png")
    dot.attr(rankdir="TB")  # Top to Bottom layout

    # Create subgraphs for each depth level
    depth_groups = {}
    for node, depth in phase_of_node.items():
        if depth not in depth_groups:
            depth_groups[depth] = graphviz.Digraph()
            depth_groups[depth].attr(rank="same")  # Force same depth
        # Color nodes with execution in blue
        if not is_node_with_empty_execution(dag, node):
            depth_groups[depth].node(str(node), style='filled', fillcolor='lightblue')
        else:
            depth_groups[depth].node(str(node))

    # Add all subgraphs to main graph
    for depth in sorted(depth_groups.keys()):
        dot.subgraph(depth_groups[depth])

    for node in dag.nodes():
        phase_parents_with_execution, parents_without_execution, same_phase_parents = get_node_parents(dag, node, phase_of_node)
        for parent in phase_parents_with_execution:
            dot.edge(str(parent), str(node)) 
        if show_all_edges:
            for parent in parents_without_execution + same_phase_parents:
                dot.edge(str(parent), str(node), style='dashed')
    return dot

def remove_prompt_numbering(text):
    # Replace opening tags
    text = re.sub(r'<prompt_\d+\.\d+>', '<prompt>', text)
    # Replace closing tags
    text = re.sub(r'</prompt_\d+\.\d+>', '</prompt>', text)
    return text

def get_plans_per_executor(plan):
    if not re.search(r'<prompt.*?>', plan):
        return []
    plan = re.sub("<Plan_\d+>", "", plan)
    plan = re.sub("</Plan_\d+>", "", plan)
    plan = plan.strip()
    plan = re.sub("</prompt.*?>", lambda m: m.group(0) + '<stop>', plan)
    split_pattern = r'<stop>'
    plans_per_executor = re.split(split_pattern, plan)
    plans_per_executor = [i.strip() for i in plans_per_executor if i not in [None, "", " "]]
    plans_per_executor = [(re.search(r'<prompt_(\d+\.\d+)>', i).group(1), i) for i in plans_per_executor if re.search(r'<prompt_(\d+\.\d+)>', i)]
    return plans_per_executor

def get_executions_per_executor(execution):
    execution = re.sub("<Execution_\d+>", "", execution)
    execution = re.sub("</Execution_\d+>", "", execution)
    execution = execution.strip()
    execution = re.sub("</execution.*?>", lambda m: m.group(0) + '<stop>', execution)
    split_pattern = r'<stop>'
    executions_per_executor = re.split(split_pattern, execution)
    executions_per_executor = [i.strip() for i in executions_per_executor if i not in [None, "", " "]]
    executions_per_executor = [(re.search(r'<execution_(\d+\.\d+)>', i).group(1), i) for i in executions_per_executor if re.search(r'<execution_(\d+\.\d+)>', i)]
    return executions_per_executor

def remove_prompt_numbering(text):
    # Replace opening tags
    text = re.sub(r'<prompt_\d+\.\d+>', '<prompt>', text)
    # Replace closing tags
    text = re.sub(r'</prompt_\d+\.\d+>', '</prompt>', text)
    return text

def create_finetuning_df(trajectory, problem_idx, final_answer, planner_prompt, executor_prompt):
    trajectory = trajectory.strip()
    patterns = [
        r'</Question>',
        r'</Plan_\d+>',
        r'</Execution_\d+>'
    ]
    # Replace <thought_k> with empty string
    trajectory = re.sub(r'<thought_\d+> ', '', trajectory)
    # Add <stop> after each pattern
    for pattern in patterns:
        trajectory = re.sub(pattern, lambda m: m.group(0) + '<stop>', trajectory)

    # Break at <stop> using regex
    split_pattern = r'<stop>'
    parts = re.split(split_pattern, trajectory)

    # Remove empty strings and strip whitespace
    parts = [p.strip() for p in parts if p.strip()]
    
    finetuning_data = []
    phase = 1
    for i in range(len(parts)):
        part = parts[i]
        if part.startswith("<Plan") or part.startswith("<Final_answer"):
            planner_prompt_append = ""
            if phase<max_phases and phase>=max_phases-3:
                planner_prompt_append = f" You have {max_phases-phase+1} phases left to complete the solution and generate the final answer."
            elif phase==max_phases:
                planner_prompt_append = f" This is the final phase. Using the plans and execution results provided to you from the previous phases, you *MUST* provide the accurate final answer in this phase. Start your response with <Final_answer>."
            if parts[i].count("<prompt")<=8: # Remove planner outputs with too many prompts
                finetuning_data.append(("\n\n".join(parts[:i]), remove_prompt_numbering(parts[i]), phase, planner_prompt+planner_prompt_append))
            phase += 1
        elif part.startswith("<Execution"):
            plans_per_executor = get_plans_per_executor(parts[i-1])
            executions_per_executor = get_executions_per_executor(parts[i])
            if len(plans_per_executor)!=len(executions_per_executor):
                print("Mismatch in plans and executions")
                continue
            for j in range(len(plans_per_executor)):
                finetuning_data.append(("\n\n".join(parts[:i-1]+[plans_per_executor[j][1]]), executions_per_executor[j][1], float(plans_per_executor[j][0]), executor_prompt))

    finetuning_df = pd.DataFrame(finetuning_data, columns=["UserPrompt", "ExpectedOutput", "Phase", "SystemPrompt"])
    finetuning_df["ProblemIdx"] = problem_idx
    finetuning_df["FinalAnswer"] = final_answer
    return finetuning_df

def add_messages(row):
    row["messages"] = [
        {"role": "system", "content": row["SystemPrompt"]},
        {"role": "user", "content": row["UserPrompt"]},
        {"role": "assistant", "content": row["ExpectedOutput"]}
    ]
    return row

def visualize_dag(dag):
    # Visualize dag using graphviz with component_source as label
    dot = graphviz.Digraph()
    for node in dag.nodes():
        dot.node(str(node), label=','.join([str(x) for x in dag.nodes[node]['component_data']['Component_source']]))
    for edge in dag.edges():
        dot.edge(str(edge[0]), str(edge[1]))
    return dot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare data in required format for training"
    )
    parser.add_argument(
        "--range",
        type=str,
        required=False,
        default=":",
        help="Range of thoughts to process",
    )
    parser.add_argument("--data_hub_path", type=str, required=True, help="Path to the dataset on hub")
    parser.add_argument("--subset", type=str, required=True, help="Subset name")
    parser.add_argument("--component_separation_model", type=str, required=True)
    parser.add_argument("--dag_creation_model", type=str, required=True)
    parser.add_argument("--parallelization_threshold", type=float, required=True, default=1.5, help="Minimum ratio of number of execution nodes to number of phases to include in finetuning data")
    parser.add_argument("--max_phases", type=int, required=True, default=12, help="Maximum number of phases allowed")
    parser.add_argument("--test_frac", type=float, required=True, default=0.2, help="Fraction of data to use for testing")
    parser.add_argument("--push_to_hub", action="store_true", help="Push the finetuned dataset to hub")
    parser.add_argument("--output_hub_path", type=str, required=False, help="Path to save the finetuned dataset to hub")
    args = parser.parse_args()
    load_dotenv()  

    data_range = args.range
    data_hub_path = args.data_hub_path
    subset = args.subset
    component_separation_model = args.component_separation_model
    dag_creation_model = args.dag_creation_model
    parallelization_threshold = args.parallelization_threshold
    max_phases = args.max_phases
    test_frac = args.test_frac
    push_to_hub = args.push_to_hub
    output_hub_path = args.output_hub_path
    ds = load_dataset(data_hub_path, subset)
    # Use train split for default
    ds = ds["train"]

    dataset_name = data_hub_path.split("/")[-1]
    
    # Create directories for logging outputs
    logs_dir = os.getenv("LOGS_DIR")
    finetuning_data_dir = os.path.join(logs_dir, f"finetuning_data/{os.path.basename(dataset_name)}_{os.path.basename(subset)}")
    # dags_networkx_dir = os.path.join(logs_dir, f"dags_networkx/{os.path.basename(dataset_name)}_{os.path.basename(subset)}")
    trajectory_data_dir = os.path.join(logs_dir, f"trajectory_data/{os.path.basename(dataset_name)}_{os.path.basename(subset)}")
    phase_dags_viz_dir = os.path.join(logs_dir, f"phase_dags_viz/{os.path.basename(dataset_name)}_{os.path.basename(subset)}")

    os.makedirs(finetuning_data_dir, exist_ok=True)
    # os.makedirs(dags_networkx_dir, exist_ok=True)
    if os.path.exists(trajectory_data_dir):
        shutil.rmtree(trajectory_data_dir)
    os.makedirs(trajectory_data_dir, exist_ok=True)
    os.makedirs(phase_dags_viz_dir, exist_ok=True)

    component_separation_prompt = text_processing_utils.read_file_as_string("src/prompts/component_separation_prompt.txt")
    component_separation_output_dir = f"{logs_dir}/component_separation_output_{component_separation_model}/{dataset_name}_{subset}"
    created_dags_dir = f"{logs_dir}/created_dags_{dag_creation_model}/{dataset_name}_{subset}"

    if "thought" not in ds.column_names:
        ds = ds.map(lambda x: {'thought':re.search(r'<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>', x["conversations"][1]["value"], re.DOTALL).group(1).strip()})

    # Extract thoughts based on the provided range
    if data_range == ":":
        max_problem_idx = max([int(os.path.splitext(filename)[0].split("_")[-1]) for filename in os.listdir(component_separation_output_dir)])
        data_range = f"0:{max_problem_idx+1}"
        print(f"Creating finetuning data for problems in data_range: {data_range}")
    else:
        print(f"Creating finetuning data for problems in data_range: {data_range}")
   
    num_dags_removed_due_to_low_parallelization = 0

    planner_prompt = text_processing_utils.read_file_as_string("src/prompts/global_agent_prompt.txt")
    executor_prompt = text_processing_utils.read_file_as_string("src/prompts/global_agent_prompt.txt")

    for problem_idx in tqdm(range(int(data_range.split(":")[0]), int(data_range.split(":")[1]))):
        try:
            thought = ds["thought"][problem_idx]
            question = ds["problem"][problem_idx]
            final_answer_line = evaluate_utils.get_final_answer_line(ds["solution"][problem_idx])
            if "boxed" in final_answer_line:
                final_answer = evaluate_utils.remove_boxed(evaluate_utils.last_boxed_only_string(final_answer_line))
            else:
                final_answer = final_answer_line

            solution_identifier = f"{problem_idx}"
            component_separation_output = text_processing_utils.read_file_as_string(os.path.join(component_separation_output_dir, f"component_separation_{problem_idx}.txt"))
            formatted_thought, thought_lines = component_formatting_utils.format_solution(thought)
            components_list = component_formatting_utils.get_components_list(component_separation_output, thought_lines)
            
            # Load dag as Networkx graph and merge nodes in chains that do not have execution
            dag = text_processing_utils.read_json_as_dag(os.path.join(created_dags_dir, f"created_dag_{problem_idx}.json"))
            for node in dag.nodes():
                dag.nodes[node]['question'] = question
                dag.nodes[node]['component_data'] = components_list[node]
            
            merge_plan_with_execution_for_short_executions(dag, 3)
            # merge_nodes_in_chains_with_empty_execution(dag)
            phase_dict, phase_of_node = get_phase_dict(dag)
            assign_prompt_names(dag, phase_dict)

            # Skip if DAG has too many phases. phase_dict does not include <Final_answer>. Hence, use max_phases-1.
            if len(phase_dict.keys())>max_phases-1:
                continue

            # # Skip if DAG is too sequential - only use DAGs that show good parallelization
            num_execution_nodes = len([node for node in dag.nodes() if not is_node_with_empty_execution(dag, node)])
            if num_execution_nodes/len(phase_dict.keys())<parallelization_threshold:
                num_dags_removed_due_to_low_parallelization += 1
                continue

            # Save processed DAG to file
            # pickle.dump(dag, open(os.path.join(dags_networkx_dir, f"dag_{solution_identifier}.pickle"), 'wb'))

            dot = visualize_phase_dag(dag, phase_of_node, show_all_edges=False)
            # dot.render(os.path.join(phase_dags_viz_dir,  f"dag_{solution_identifier}"), format='png', cleanup=True)

            # Create trajectory and prepare finetuning data
            trajectory = create_trajectory(question, phase_dict, phase_of_node, components_list, dag, final_answer_line)
            with open(os.path.join(trajectory_data_dir, f"trajectory_data_{solution_identifier}.txt"), "w") as log_file:
                log_file.write(trajectory)

        except Exception as e:
            print(f"Error occurred while processing thought {problem_idx} at line {sys.exc_info()[2].tb_lineno}. Error: {str(e)}")
            continue

    print(f"{num_dags_removed_due_to_low_parallelization} out of {int(data_range.split(':')[1]) - int(data_range.split(':')[0])} DAGs were removed due to low parallelization")