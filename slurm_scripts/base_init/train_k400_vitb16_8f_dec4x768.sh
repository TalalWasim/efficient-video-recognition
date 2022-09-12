#!/bin/bash
#SBATCH --job-name=eff_base
#SBATCH --partition=multigpu
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:16
#SBATCH --exclude=p4-r67-b.g42cloud.net

exp_dir=runs/base_init/k400_vitb16_8f_dec4x768

mkdir -p "${exp_dir}"
python -u -m torch.distributed.run --nproc_per_node 16 \
  main.py \
    --num_steps 50000 \
    --backbone "ViT-B/16-dino-mae" \
    --backbone_type dino \
    --backbone_path ../pretrained/backbones_eff/converted_base.pth \
    --decoder_num_layers 4 \
    --decoder_qkv_dim 768 \
    --decoder_num_heads 12 \
    --num_classes 400 \
    --checkpoint_dir "${exp_dir}" \
    --auto_resume \
    --train_list_path ../datasets/kinetics-dataset/k400_resized_1/annotations_svt/train.csv \
    --val_list_path ../datasets/kinetics-dataset/k400_resized_1/annotations_svt/val.csv \
    --batch_size 256 \
    --batch_split 1 \
    --auto_augment rand-m7-n4-mstd0.5-inc1 \
    --mean 0.48145466 0.4578275 0.40821073 \
    --std 0.26862954 0.26130258 0.27577711 \
    --num_workers 6 \
    --num_frames 8 \
    --sampling_rate 16 \
    --num_spatial_views 1 \
    --num_temporal_views 3 \
  2>&1 | tee "${exp_dir}/train-$(date +"%Y%m%d_%H%M%S").log"