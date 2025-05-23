nproc_per_node=8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=$nproc_per_node \
swift sft \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --train_type full \
    --dataset anonym-submit-paper/Finetuning-Trajectories-Math \
    --use_hf true \
    --torch_dtype bfloat16 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_checkpointing true \
    --weight_decay 0.1 \
    --target_modules all-linear \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 7168 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero3

