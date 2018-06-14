CUDA_VISIBLE_DEVICES=0,1 python main.py  ucf101 RGB \
    ../temporal-segment-networks/data/ucf101_rgb_train_split_1.txt \
    ../temporal-segment-networks/data/ucf101_rgb_val_split_1.txt \
    --arch resnet34 --num_segments 3 --gd 20  --lr 0.001 --lr_steps 30 60 \
    --epochs 80 -b 64 -j 4 --dropout 0.8 --consensus_type ele_multi \
    --result_path UCF101_results/ele_multi_results/results_34_n_n

CUDA_VISIBLE_DEVICES=0,1 python main.py  ucf101 RGB \
    ../temporal-segment-networks/data/ucf101_rgb_train_split_1.txt \
    ../temporal-segment-networks/data/ucf101_rgb_val_split_1.txt \
    --arch resnet34 --num_segments 3 --gd 20  --lr 0.001 --lr_steps 30 60 \
    --epochs 80 -b 64 -j 4 --dropout 0.8 --consensus_type ele_multi \
    --result_path UCF101_results/ele_multi_results/results_34_s_n \
    --train_shuffle
