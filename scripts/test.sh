set -ex
python test.py --dataroot /home/dangpb1/Research/CamSpecDeblurring/blur2blur/datasets/datasets/PhoneCraft2/b2b/FakeBlur_opencv \
                --name b2bbase_PhoneCraft2OpenCV_2Dnet_512crop256_100L1_0.8Perc_TopK_augcolorreal_instancenorm \
                --eval \
                --model b2bbase --netG unet_256 \
                --checkpoints_dir ckpts/new1_ckpts \
                --direction AtoB \
                --preprocessA padding \
                --preprocessB none \
                --dataset_mode unaligned \
                --norm instance \
                --phase train \
                --delta -1 \
                --max_dataset_size 100 \
                --epoch 150 \
                # --load_iter 56 \
                # --max_dataset_size 100 \
                # --epoch 26 \