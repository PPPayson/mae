module load libjpeg-turbo
OUTPUT_DIR='outs/mae/mae_small_patch16_224_timm_0115'
export HUGGINGFACE_HUB_CACHE=./pretrained_ckpts

DATA_PATH='data_lists/co3d_train.txt'
DATA_VAL_PATH='data_lists/co3d_val.txt'
DATA_ROOT='/oscar/data/tserre/Shared/'
IMAGE_DATA='/oscar/data/tserre/Shared/Co3D/binocular_trajectory/'

NCCL_DEBUG=INFO OMP_NUM_THREADS=1  python3 -m torch.distributed.run \
     --standalone \
     --nnodes=1 \
     --nproc_per_node=2 \
      main_pretrain.py \
        --num_workers 16 \
        --data_root ${DATA_ROOT} \
        --data_path ${DATA_PATH} \
        --data_val_path ${DATA_VAL_PATH} \
        --img_data_path ${IMAGE_DATA} \
        --model mae_vit_small_patch16_timm \
        --batch_size 512 \
        --blr 1.5e-5 \
        --weight_decay 0.01 \
        --warmup_epochs 40 \
        --epochs 200 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --mask_ratio 0.75 \
        --not_pretrained