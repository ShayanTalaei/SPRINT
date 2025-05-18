# Define input variables
FINETUNING_RANGE=":"
DATA_HUB_PATH="xyz/OpenThoughts-correct"
SUBSET="math"
COMPONENT_SEPARATION_MODEL="gpt-4o"
DAG_CREATION_MODEL="gpt-4o"
PARALLELIZATION_THRESHOLD=1.5
MAX_PHASES=12
TEST_FRAC=0.1
OUTPUT_HUB_PATH="xyz/Plan-Execution-Data-Math-Full-Soln"

# Run the script with defined variables
python src/finetuning_data_preparation/plans_and_executions_remove_prompt_merge_chains.py \
    --range $FINETUNING_RANGE \
    --data_hub_path $DATA_HUB_PATH \
    --subset $SUBSET \
    --component_separation_model $COMPONENT_SEPARATION_MODEL \
    --dag_creation_model $DAG_CREATION_MODEL \
    --parallelization_threshold $PARALLELIZATION_THRESHOLD \
    --max_phases $MAX_PHASES \
    --test_frac $TEST_FRAC \
    --output_hub_path $OUTPUT_HUB_PATH \
    --push_to_hub