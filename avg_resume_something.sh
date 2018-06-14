CUDA_VISIBLE_DEVICES=0,3 python main.py something RGB \
    ../TRN-pytorch/video_datasets/something/something_rgb_train_split_1.txt \
    ../TRN-pytorch/video_datasets/something/something_rgb_val_split_1.txt \
    --arch resnet34 --num_segments 3 --gd 20  --lr 0.001 --lr_steps 15 18 20 \
    --epochs 24 -b 128 -j 4 --dropout 0.8 --consensus_type avg \
    --result_path Something_results/avg_results/results_34_n_n \
    --resume Something_results/avg_results/results_34_n_n/_rgb_checkpoint.pth.tar
