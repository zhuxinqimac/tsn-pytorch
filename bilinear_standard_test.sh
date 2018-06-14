CUDA_VISIBLE_DEVICES=1 python test_models.py ucf101 RGB \
    ../temporal-segment-networks/data/ucf101_rgb_val_split_1.txt \
    UCF101_results/bilinear_results/results_34_n_n/_rgb_model_best.pth.tar \
    --test_segments 8 \
    --consensus_type bilinear_att\
    --arch resnet34 --workers 1 \
    --save_scores UCF101_results/bilinear_results/results_34_n_n/standard_test_scores.txt \
    --bi_out_dims 256 \
    --bi_rank 256 \
    --bi_add_clf \
    --bi_filter_size 1 \
    --bi_conv_dropout 0 \
    --bi_dropout 0 \
    --dropout 0.5
    #--flow_pref flow_ --test_shuffle
#CUDA_VISIBLE_DEVICES=1 python test_models.py something Flow \
    #../TRN-pytorch/video_datasets/something/something_flow_nozero_val_split_1.txt \
    #Something_results/optical_results/results_34_n_n/_flow_model_best.pth.tar \
    #--arch resnet34 --workers 1 \
    #--save_scores Something_results/optical_results/results_34_n_n/standard_test_scores.txt \
    #--flow_pref flow_
