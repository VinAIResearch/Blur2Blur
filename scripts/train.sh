set -ex
python train.py --dataroot datasets/GoPro/b2b_exp/RB2V_GOPRO_filter \
        --name test_b2b_official \
        --model blur2blur --netG mimounet \
        --batch_size 1 \
        --dataset_mode unaligned \
        --norm instance --pool_size 0 \
        --display_id -1 \
        # --direction AtoB \
        # --preprocessA resize_crop \
        # --preprocessB resize_crop \
        # --load_size 512 \
        # --lambda_Perc 100 \
        # --use_wandb
