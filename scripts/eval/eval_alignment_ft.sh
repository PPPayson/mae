module load libjpeg-turbo
export HUGGINGFACE_HUB_CACHE=./pretrained_ckpts

OUTPUT_DIR='outs/finetune/co3d_lp/vit_small_patch16_224.augreg_in21k_ft_in1k/02'
DATA_PATH='data_lists/co3d_train.txt'
DATA_VAL_PATH='data_lists/co3d_val.txt'
DATA_ROOT='/oscar/data/tserre/Shared/'

python -m tools.eval_alignment \
    --model vit_small_patch16_224.augreg_in21k_ft_in1k\
    --num_workers 8 \
    --batch_size 128 \
    --data_root ${DATA_ROOT} \
    --data_path ${DATA_PATH} \
    --data_val_path ${DATA_VAL_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${OUTPUT_DIR} \
    --lr 1e-5 \
    --opt_betas 0.9 0.999 \
    --warmup_epochs 5 \
    --weight_decay 0 \
    --opt_eps 1e-8 \
    --drop_path 0 \
    --epochs 50 \
    --num_classes 51 \
    --dataset co3d \
    --drop_rate 0.0 \
    --timm_model \
    --eval_co3d \
    --eval_co3d_every 1 \
    --eval_co3d_batch_size 128 \
    --eval_co3d_epochs 10\
    --clickmaps_human_path ./assets/human_ceiling_split_half_co3d_val.npz \
    --clickmaps_path ./assets/co3d_val_processed.npz \
    --imgnet_clickmaps_path ./assets/jay_imagenet_for_co3d_val_0.1_processed.npz \
    --imgnet_clickmaps_human_path ./assets/human_ceiling_split_half_jay_imagenet_for_co3d_val_0.1.npz \
    --ckpt_path outs/finetune/co3d/vit_small_patch16_224.augreg_in21k_ft_in1k/best_val.ckpt