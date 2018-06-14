#CUDA_VISIBLE_DEVICES=1 python test_models.py ucf101 Flow \
    #../temporal-segment-networks/data/ucf101_flow_val_split_1.txt \
    #UCF101_results/optical_results/results_34_n_n/_flow_model_best.pth.tar \
    #--arch resnet34 --workers 1 \
    #--save_scores UCF101_results/optical_results/results_34_n_s/standard_test_scores.txt \
    #--flow_pref flow_ --test_shuffle
CUDA_VISIBLE_DEVICES=1 python test_models.py something Flow \
    ../TRN-pytorch/video_datasets/something/something_flow_nozero_val_split_1.txt \
    Something_results/optical_results/results_34_n_n/_flow_model_best.pth.tar \
    --arch resnet34 --workers 1 \
    --save_scores Something_results/optical_results/results_34_n_n/standard_test_scores.txt \
    --flow_pref flow_
