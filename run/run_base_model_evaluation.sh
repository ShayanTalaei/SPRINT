BASE_MODEL="anonym-submit-paper/RFT-DeepSeek-R1-Distill-Qwen-7B"
BASE_MODEL_SERVER_URL="http://127.0.0.1:8000/v1/"
BASE_MODEL_TEMP="0.6"
BASE_MODEL_TYPE="think_ft_trajectory"

TEST_HUB_PATH="HuggingFaceH4/MATH-500"
ORCHESTRATION_SPLIT="train"
TEST_SUBSET="default"
ORCHESTRATION_RANGE=":500"


# Run the script with defined variables
python src/base_model_evaluation/main.py --test_range $ORCHESTRATION_RANGE --test_hub_path $TEST_HUB_PATH --test_subset $TEST_SUBSET --orchestration_split $ORCHESTRATION_SPLIT\
 --base_model $BASE_MODEL --base_model_server_url $BASE_MODEL_SERVER_URL --base_model_temp $BASE_MODEL_TEMP\
 --base_model_type $BASE_MODEL_TYPE