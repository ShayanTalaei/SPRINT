# Define variables directly in the script
PLANNER_MODEL="xyz/model_name"
PLANNER_SERVER_URL="http://34.147.114.179:8000/v1/"
PLANNER_TEMP="0.6"
PLANNER_OUTPUT_TOKEN_LIMIT=1000

EXECUTOR_MODEL="xyz/model_name"
EXECUTOR_SERVER_URL="http://34.147.114.179:8000/v1/"
EXECUTOR_TEMP="0.6"

TEST_HUB_PATH="HuggingFaceH4/MATH-500"
ORCHESTRATION_SPLIT="test"
TEST_SUBSET="default"
ORCHESTRATION_RANGE=":"
MAX_PHASES=12
MAX_PROBLEM_WORKERS=5
MAX_EXECUTION_WORKERS=5

# Run the script with defined variables
python src/agent_orchestration/main_full_trajectory.py --test_range $ORCHESTRATION_RANGE --test_hub_path $TEST_HUB_PATH --test_subset $TEST_SUBSET --orchestration_split $ORCHESTRATION_SPLIT\
 --planner_model $PLANNER_MODEL --planner_server_url $PLANNER_SERVER_URL --planner_temp $PLANNER_TEMP\
 --executor_model $EXECUTOR_MODEL --executor_server_url $EXECUTOR_SERVER_URL --executor_temp $EXECUTOR_TEMP\
 --max_phases $MAX_PHASES --max_problem_workers $MAX_PROBLEM_WORKERS --max_execution_workers $MAX_EXECUTION_WORKERS\
 --planner_output_token_limit $PLANNER_OUTPUT_TOKEN_LIMIT