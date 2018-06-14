CUDA_VISIBLE_DEVICES=1 python score_sens.py \
    UCF101_results/avg_results/results_34_n_n/_rgb_model_best.pth.tar \
    avg ucf101 \
    ../temporal-segment-networks/data/ucf101_rgb_val_split_1.txt \
    UCF101_results/avg_results/sens \
    /home/xinqizhu/ucfTrainTestlist/classInd.txt \
    --num_segments 3 --modality RGB \
    --arch resnet34 

