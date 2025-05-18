import re
import sys
sys.path.append("src/utils/")
import text_processing_utils

def format_solution(math_sol):
    lines = text_processing_utils.split_into_sentences(math_sol)
    lines = [i for i in lines if i!="" and i!=None and i!=" "]

    math_sol = ""
    for i in range(len(lines)):
        math_sol += "<Line {}> {} </Line {}>\n".format(str(i+1), lines[i].rstrip('\n'), str(i+1))
    return math_sol, lines

def parse_line_range(line_range):
    line_range = line_range.replace("Lines", "").replace("Line", "").strip()
    try:
        if '-' not in line_range: # For cases like "Lines 14" or "Line 14"
            return [int(line_range), int(line_range)]
        else:
            match = re.match(r'(\d+).*?(\d+)', line_range, re.DOTALL) # Matches with "Lines 24-25", "Lines 25 -- 28", etc.
            return [int(match.group(1)), int(match.group(2))]
    except:
        return None

def join_sol_lines(math_sol_lines, line_range_tuple):
    if line_range_tuple is None:
        return None
    if not isinstance(line_range_tuple, list) or len(line_range_tuple) != 2 or not all(isinstance(i, int) for i in line_range_tuple):
        raise ValueError("line_range_tuple must be a tuple of 2 integers. Instead found {line_range_tuple}")
    start_idx, end_idx = line_range_tuple
    start_idx-=1 # Since line numbering in formatted_math_sol starts with 1.
    end_idx-=1
    if start_idx>end_idx:
        raise ValueError(f"start_idx must be less than or equal to end_idx Instead found {line_range_tuple}")
    if not (start_idx>=0 and start_idx<len(math_sol_lines) and end_idx>=0 and end_idx<len(math_sol_lines)):
        raise ValueError(f"start_idx or end_idx is not in the required range. (start_idx, end_idx)=({start_idx}, {end_idx}) but len(math_sol_lines)={str(len(math_sol_lines))}")
    required_lines = math_sol_lines[start_idx:end_idx+1]
    required_lines = [line if line.endswith("\n") else line+" " for line in required_lines]
    return "".join(required_lines).strip()

def get_components_list(solution_separation_output, math_sol_lines):
    components = solution_separation_output.split("### Component")
    if not solution_separation_output.startswith("### Component"):
        components = components[1:]
    components = ["### Component" + component for component in components if component]

    components_list = dict()

    for component in components:
        match = re.match(r'### Component (\d+)(.*)\n+- \*\*Description:\*\*(.*?)\n+- \*\*Plan:\*\*(.*?)\n+- \*\*Prompt:\*\*(.*?)\n+- \*\*Execution:\*\*(.*?)\n+- \*\*Comment:\*\* (.*?)\n*', component, re.DOTALL)
        if match:
            component_number = int(match.group(1))
            component_dict = {
                "Component_source": [component_number],
                "Description": match.group(3).strip(),
                "Plan_lines": match.group(4).strip(),
                "Prompt": match.group(5).strip(),
                "Execution_lines": match.group(6).strip(),
                "Comment_lines": match.group(7).strip()
            }
            component_dict["Plan_lines"] = parse_line_range(component_dict["Plan_lines"])
            component_dict["Execution_lines"] = parse_line_range(component_dict["Execution_lines"])
            component_dict["Comment_lines"] = parse_line_range(component_dict["Comment_lines"])

            component_dict["Plan"] = join_sol_lines(math_sol_lines, component_dict["Plan_lines"])
            component_dict["Execution"] = join_sol_lines(math_sol_lines, component_dict["Execution_lines"])
            component_dict["Comment"] = join_sol_lines(math_sol_lines, component_dict["Comment_lines"])

            components_list[component_number] = component_dict
    return components_list

def format_solution_with_components_inline(components_list):
    component_inline_sol = []
    for component_number, component_dict in components_list.items():
        ComponentNumber, Description, Plan, Prompt, Execution, Comment = component_number, component_dict["Description"] ,component_dict["Plan"], component_dict["Prompt"], component_dict["Execution"], component_dict["Comment"]
        formatted_component = f"""### Component {ComponentNumber}
<description>
{Description}
</description>

<plan>
{Plan}
</plan>

<prompt>
{Prompt}
</prompt>

<execution>
{Execution}
</execution>
"""
        component_inline_sol.append(formatted_component)
    return "\n\n".join(component_inline_sol)