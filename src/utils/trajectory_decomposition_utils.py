import re

def split_trajectory_into_phases(trajectory):
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
    return parts

def count_tokens(text, tokenizer):
    tokens = tokenizer(text, return_tensors="pt")
    token_count = len(tokens.input_ids[0])
    return token_count

def get_plans_per_executor(plan):
    if not re.search(r'<prompt.*?>', plan):
        return [plan]
    plan = re.sub("</prompt.*?>", lambda m: m.group(0) + '<stop>', plan)
    split_pattern = r'<stop>'
    plans_per_executor = re.split(split_pattern, plan)
    plans_per_executor = [i.strip() for i in plans_per_executor if i not in [None, "", " "]]
    return plans_per_executor

def get_executions_per_executor(execution):
    execution = re.sub(r'<Execution.*?>', '', execution)
    execution = re.sub(r'</Execution.*?>', '', execution)
    execution = execution.strip()
    if "No prompt to execute." in execution:
        return []
    if not re.search(r'<execution.*?>', execution):
        return [execution]
    execution = re.sub("</execution.*?>", lambda m: m.group(0) + '<stop>', execution)
    split_pattern = r'<stop>'
    executions_per_executor = re.split(split_pattern, execution)
    executions_per_executor = [i.strip() for i in executions_per_executor if i not in [None, "", " "]]
    return executions_per_executor

def remove_text_between_tags(text):
    text = re.sub(r'<.*?>\n\n', '', text)
    text = re.sub(r'<.*?>', '', text)
    return text

def decompose_trajectory(trajectory, split_plans=True, split_executions=True, include_token_count=True, planner_tokenizer=None, executor_tokenizer=None):
    parts = split_trajectory_into_phases(trajectory)
    decomposed_parts = []
    for part in parts:
        if re.search(r'<Question>', part):
            continue
        if split_plans and re.search(r'<Plan_\d+>', part):
            plans = get_plans_per_executor(part)
            if include_token_count:
                plans = [(plan, count_tokens(remove_text_between_tags(plan), planner_tokenizer)) for plan in plans]
            decomposed_parts.append(plans)
        elif split_executions and re.search(r'<Execution_\d+>', part):
            executions = get_executions_per_executor(part)
            if include_token_count:
                executions = [(execution, count_tokens(remove_text_between_tags(execution), executor_tokenizer)) for execution in executions]
            decomposed_parts.append(executions)
        else:
            if include_token_count:
                part = [(part, count_tokens(remove_text_between_tags(part), planner_tokenizer))]
            decomposed_parts.append(part)
    return decomposed_parts

def combine_plan_with_executions_within_a_phase(plan, executions):
    # Input: Plan separated based on prompts, Executions per executor
    # Output: Plan1 \n\n Exec1 \n\n Plan2 \n\n Exec2... (prompts are removed) -- all within a single phase
    assert len(plan)==len(executions)+1
    for i in range(len(plan)-1):
        executions[i] = re.sub(r'<execution.*?>', '', executions[i], flags=re.IGNORECASE)
        executions[i] = re.sub(r'</execution.*?>', '', executions[i], flags=re.IGNORECASE)
        executions[i] = executions[i].strip()
        plan[i] = re.sub(r'<prompt_\d+\.\d+>', '<prompt>', plan[i], flags=re.DOTALL)
        plan[i] = re.sub(r'</prompt_\d+\.\d+>', '</prompt>', plan[i], flags=re.DOTALL)
        plan[i] = re.sub(r'<prompt>.*?</prompt>', '', plan[i], flags=re.DOTALL)
        plan[i]+=executions[i]

    plan = "\n\n".join(plan)
    plan = re.sub(r'Based on execution_.*?\n', '', plan)
    plan = re.sub(r'<Plan.*?>', '', plan, flags=re.IGNORECASE)
    plan = re.sub(r'</Plan.*?>', '', plan, flags=re.IGNORECASE)
    plan = plan.strip()
    return plan    

def reconstruct_trajectory_from_decomposition(decomposed_parts):
    # Reconstructions from start of the solution to the end of each phase will be returned. 
    # Last element in the returned list will be the full reconstruction.
    phase_wise_reconstructions = []
    phase_plans = []
    for i in range(0, len(decomposed_parts), 2):
        if i==len(decomposed_parts)-1:
            ## Uncomment if final answer line needs to be included in trajectory.
            # part = decomposed_parts[i]
            # part = re.sub(r'<Final_answer>', '', part, flags=re.IGNORECASE)
            # part = re.sub(r'</Final_answer>', '', part, flags=re.IGNORECASE)
            # phase_plans.append(part.strip())
            continue
        phase_plans.append(combine_plan_with_executions_within_a_phase(decomposed_parts[i], decomposed_parts[i+1]))
        phase_wise_reconstructions.append("\n\n".join(phase_plans))
    return phase_wise_reconstructions

def count_tokens(text, tokenizer):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    return len(tokens)