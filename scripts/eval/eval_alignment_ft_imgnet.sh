module load libjpeg-turbo
export HUGGINGFACE_HUB_CACHE=./pretrained_ckpts

OUTPUT_DIR='outs/finetune/imgnet/eval'
DATA_PATH='data_lists/co3d_train.txt'
DATA_VAL_PATH='data_lists/co3d_val.txt'
DATA_ROOT='/oscar/data/tserre/data/ImageNet/ILSVRC/Data/CLS-LOC'

python eval_alignment.py \
    --model vit_small_patch16_224.augreg_in21k_ft_in1k \
    --num_workers 12 \
    --batch_size 128 \
    --dataset imgnet \
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
    --epochs 398 \
    --num_classes 1000 \
    --drop_rate 0.7 \
    --timm_model \
    --eval_co3d \
    --eval_co3d_every 1 \
    --eval_co3d_batch_size 128 \
    --eval_co3d_epochs 50\
    --clickmaps_human_path ./assets/human_ceiling_split_half_co3d_val.npz \
    --clickmaps_path ./assets/co3d_val_processed.npz \
    --imgnet_clickmaps_path ./assets/imagenet_val_random_50_0.1_processed.npz \
    --imgnet_clickmaps_human_path ./assets/human_ceiling_split_half_imagenet_val_random_50_0.1.npz \
    --ckpt_path outs/finetune/imgnet/full_new/vit_small_patch16_224.augreg_in21k_ft_in1k/best_val.ckpt
