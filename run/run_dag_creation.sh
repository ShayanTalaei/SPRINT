# Define input variables
DAG_CREATION_MODEL="gpt-4o"
COMPONENT_SEPARATION_MODEL="gpt-4o"
DATA_HUB_PATH="xyz/dataset_name"
SUBSET="math"
RANGE="1:5"
NUM_WORKERS=10

# Run the script with defined variables
python src/dag_creation/main.py --dag_creation_model $DAG_CREATION_MODEL --component_separation_model $COMPONENT_SEPARATION_MODEL --range $RANGE --num_workers $NUM_WORKERS --data_hub_path $DATA_HUB_PATH --subset $SUBSET