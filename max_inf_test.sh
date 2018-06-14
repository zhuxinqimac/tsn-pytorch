CUDA_VISIBLE_DEVICES=2 python score_inf.py \
    UCF101_results/max_results/results_34_n_n/_rgb_model_best.pth.tar \
    UCF101_results/max_results/results_34_s_n/_rgb_model_best.pth.tar \
    max ucf101 \
    ../temporal-segment-networks/data/ucf101_rgb_val_split_1.txt \
    UCF101_results/avg_results/inf --num_segments 3 --modality RGB \
    --arch resnet34 
