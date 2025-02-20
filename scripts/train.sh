module load libjpeg-turbo
OUTPUT_DIR='outs/mae_small_patch16_224_1101'
export HUGGINGFACE_HUB_CACHE=./pretrained_ckpts

DATA_LIST='data_lists/co3d_train.txt'
DATA_VAL_PATH='data_lists/co3d_val.txt'
DATA_ROOT='/oscar/data/tserre/Shared/'

NCCL_DEBUG=INFO OMP_NUM_THREADS=1  python3 -m torch.distributed.run \
     --standalone \
     --nnodes=1 \
     --nproc_per_node=2 \
      main_pretrain.py \
        --num_workers 16 \
        --data_path /oscar/data/tserre/Shared/Co3D/binocular_trajectory/ \
        --data_root ${DATA_ROOT} \
        --data_list ${DATA_LIST} \
        --model mae_vit_small_patch16 \
        --batch_size 64 \
        --blr 1.5e-4 \
        --weight_decay 0.01 \
        --warmup_epochs 10 \
        --epochs 50 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --mask_ratio 0.75