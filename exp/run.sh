export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MASTER_PORT=$((20000 + RANDOM % 10000))
export OMP_NUM_THREADS=1

JOB_NAME='mogo'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"

GPUS=$(expr $(echo "${CUDA_VISIBLE_DEVICES}" | grep -o "," | wc -l) + 1)

torchrun \
    --master_port=${MASTER_PORT} \
    --nproc_per_node=${GPUS} \
    run_class_finetuning.py \
    --model videomamba_middle \
    --finetune /path/to/videomamba_m16_k400_mask_ft_f8_res224.pth \
    --data_set 'Kinetics_sparse' \
    --split ',' \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 30 \
    --num_sample 1 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 100 \
    --nb_classes 22 \
    --num_workers 16 \
    --warmup_epochs 5 \
    --tubelet_size 1 \
    --epochs 50 \
    --lr 1e-4 \
    --layer_decay 0.8 \
    --drop_path 0.4 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --test_num_segment 4 \
    --test_num_crop 3 \
    --dist_eval \
    --test_best \
    --bf16
