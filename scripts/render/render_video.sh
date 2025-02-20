module load libjpeg-turbo
OUTPUT_DIR='outs/autoreg_beit/renders'
export HUGGINGFACE_HUB_CACHE=./pretrained_ckpts

DATA_PATH='data_lists/co3d_train.txt'
DATA_VAL_PATH='data_lists/co3d_val.txt'
DATA_ROOT='/oscar/data/tserre/Shared/'

python -m tools.render_video \
        --num_workers 16 \
        --data_root ${DATA_ROOT} \
        --data_path ${DATA_PATH} \
        --data_val_path ${DATA_VAL_PATH} \
        --mask_type autoregressive \
        --mask_ratio 0.75 \
        --model autoreg_beitv2_large_patch16_dec512d8 \
        --batch_size 64 \
        --lr 0.015 \
        --no-binocular \
        --num_frames 8 \
        --weight_decay 0.01 \
        --sampling_rate 4 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 10 \
        --seed 84 \
        --save_ckpt_freq 2 \
        --epochs 50 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --opt_eps 1e-7 \
        --schedule_free \
        --no-normalize_target \
        --eval_co3d \
        --eval_co3d_every 1 \
        --eval_co3d_batch_size 128 \
        --eval_co3d_epochs 50\
        --use_cce \
        --num_classes 1000 \
        --attn_drop_rate 0 \
        --drop_path 0 \
        --categorical_camera \
        --feature_loss \
        --decoder_pos_embed 3d \
        --clickmaps_human_path ./assets/human_ceiling_split_half_co3d_val.npz \
        --clickmaps_path ./assets/co3d_val_processed.npz \
        --imgnet_clickmaps_path ./assets/jay_imagenet_for_co3d_val_0.1_processed.npz \
        --imgnet_clickmaps_human_path ./assets/human_ceiling_split_half_jay_imagenet_for_co3d_val_0.1.npz \
        --ckpt_path outs/results/autoreg_beit_large_patch16_no_cam_lr0005_0212/checkpoint-0.pth \
        --linear_probe

