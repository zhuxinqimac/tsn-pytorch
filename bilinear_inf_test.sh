CUDA_VISIBLE_DEVICES=1 python score_inf.py \
    UCF101_results/bilinear_results/results_34_n_n/_rgb_model_best.pth.tar \
    UCF101_results/bilinear_results/results_34_s_s/_rgb_model_best.pth.tar \
    bilinear_att ucf101 \
    ../temporal-segment-networks/data/ucf101_rgb_val_split_1.txt \
    UCF101_results/conv_lstm_results/inf \
    /home/xinqizhu/ucfTrainTestlist/classInd.txt \
    --num_segments 8 --modality RGB \
    --arch resnet34 \
    --dropout 0.5 \
    --bi_out_dims 256 \
    --bi_rank 256 \
    --bi_add_clf \
    --bi_filter_size 1 \
    --bi_conv_dropout 0 \
    --bi_dropout 0

#CUDA_VISIBLE_DEVICES=1 python score_inf.py \
    #Something_results/bilinear_results/results_34_n_n/_rgb_model_best.pth.tar \
    #Something_results/bilinear_results/results_34_s_s/_rgb_model_best.pth.tar \
    #bilinear_att something \
    #../TRN-pytorch/video_datasets/something/something_rgb_val_split_1.txt \
    #Something_results/conv_lstm_results/inf \
    #../TRN-pytorch/video_datasets/something/category.txt \
    #--num_segments 8 --modality RGB \
    #--arch resnet34 \
    #--dropout 0.5 \
    #--bi_out_dims 256 \
    #--bi_rank 256 \
    #--bi_add_clf \
    #--bi_filter_size 1 \
    #--bi_conv_dropout 0 \
    #--bi_dropout 0
