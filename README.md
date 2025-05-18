# SPRINT: Enabling Interleaved Planning and Parallelized Execution in Reasoning Models

## Overview
Large reasoning models (LRMs) excel at complex reasoning tasks but typically generate lengthy sequential chains-of-thought, resulting in long inference times before arriving at the final answer. To address this challenge, we introduce SPRINT, a novel post-training and inference-time framework designed to enable LRMs to dynamically identify and exploit opportunities for parallelization during their reasoning process. SPRINT incorporates an innovative data curation pipeline that reorganizes natural language reasoning trajectories into structured rounds of long-horizon planning and parallel execution. By fine-tuning LRMs on a small amount of such curated data, the models *learn* to dynamically identify independent subtasks within extended reasoning processes and effectively execute them in parallel. Through extensive evaluations, we show that the models fine-tuned with the SPRINT framework match the performance of reasoning models on complex domains such as mathematics while generating up to 39% fewer sequential tokens on problems requiring more than 8000 output tokens. Finally, we observe consistent results transferred to two out-of-distribution tasks of GPQA and Countdown with up to 45% and 65% reduction in average sequential tokens for longer reasoning trajectories, while achieving the performance of the fine-tuned reasoning model.

## Data Curation

![Data curation pipeline](figures/SPRINT_Training_overview.png)

### 1) Component separation

Set the following variables in `run_component_separation.sh`
- `COMPONENT_SEPARATION_MODEL`: Name of the model to use for component separation
- `RANGE`: Range of problems to run component separation on. Set it to ":" to run on all problems.
- `DATA_HUB_PATH`: HuggingFace hub path of the dataset containing reasoning trajectories to run component separation on. The reasoning traces must be in a column labeled `thought`.
- `SUBSET`: Subset of the above HuggingFace dataset to use.
- `NUM_WORKERS`: Number of workers to use for component separation.

```
$ sh run_component_separation.sh
```

### 2) DAG creation
- `DAG_CREATION_MODEL`: Name of the model to use for DAG creation.

```
$ sh run_dag_creation.sh
```

### 3) Packing and Finetuning data preparation

```
$ sh run_finetuning_data_preparation.sh
```

## Running SPRINT

To run SPRINT over a dataset, serve the model using vLLM and use the following variables in `run_agent_orchestration.sh` to configure the run:

- `PLANNER_MODEL` and `EXECUTOR_MODEL`: Name of the served model. Typically, the same model is used to generate both plans and executions. 
- `PLANNER_SERVER_URL` and `EXECUTOR_SERVER_URL`: URL of the vLLM server where the model is served.
- `TEST_HUB_PATH`: HuggingFace hub path of the test set to run evaluation on
- `ORCHESTRATION_SPLIT`: Split of the HuggingFace dataset used for evaluation. For HuggingFace datasets that do not specify a split, set it to `default`.
- `TEST_SUBSET`: Subset of the above HuggingFace dataset to use.
- `ORCHESTRATION_RANGE`: Range of problems to run evaluation on. Set it to ":" to run on all problems. 
- `MAX_PHASES`: Maximum number of iterative plan and execution stages the model is allowed to iterate over before arriving at the final answer. 

Finally, run the following command
```
$ sh run_agent_orchestration.sh
```





