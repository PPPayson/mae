module load libjpeg-turbo
export HUGGINGFACE_HUB_CACHE=./pretrained_ckpts

OUTPUT_DIR='outs/finetune/imgnet/full'
DATA_PATH='data_lists/co3d_train.txt'
DATA_ROOT='/oscar/data/tserre/data/ImageNet/ILSVRC/Data/CLS-LOC'
DATA_VAL_PATH='data_lists/co3d_val.txt'
#DATA_ROOT='/oscar/data/tserre/Shared/'

# NCCL_DEBUG=INFO OMP_NUM_THREADS=1  python -m torch.distributed.run \
#      --standalone \
#      --nnodes=1 \
#      --nproc_per_node=2 \
python  main_finetune_classification.py \
    --model vit_small_patch16_224.augreg_in21k_ft_in1k \
    --num_workers 12 \
    --batch_size 128 \
    --data_root ${DATA_ROOT} \
    --data_path ${DATA_PATH} \
    --data_val_path ${DATA_VAL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --lr 1e-5 \
    --opt_betas 0.9 0.999 \
    --warmup_epochs 5 \
    --weight_decay 0 \
    --opt_eps 1e-8 \
    --wandb \
    --drop_path 0 \
    --epochs 200 \
    --num_classes 1000 \
    --dataset imgnet \
    --drop_rate 0.7 \
    --clickmaps_human_path ./assets/human_ceiling_split_half_jay_imagenet_for_co3d_val_0.1.npz \
    --clickmaps_path ./assets/jay_imagenet_for_co3d_val_0.1_processed.npz \
    --imgnet_clickmaps_path assets/jay_imagenet_val_0.1_processed.npz \
    --imgnet_clickmaps_human_path assets/human_ceiling_split_half_imagenet_val_0.1.npz \
    --ckpt_path outs/finetune/imgnet/full/vit_small_patch16_224.augreg_in21k_ft_in1k/best_val.ckpt