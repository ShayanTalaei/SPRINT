# Define input variables
COMPONENT_SEPARATION_MODEL="gpt-4o"
RANGE=":"
DATA_HUB_PATH="anonym-submit-paper/OpenThoughts-correct"
SUBSET="math"
NUM_WORKERS=10
# Run the script with defined variables

python src/component_separation/main.py --component_separation_model $COMPONENT_SEPARATION_MODEL --range $RANGE --data_hub_path $DATA_HUB_PATH --subset $SUBSET --num_workers $NUM_WORKERS