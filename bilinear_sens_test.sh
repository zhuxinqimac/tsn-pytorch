CUDA_VISIBLE_DEVICES=1 python score_sens.py \
    UCF101_results/bilinear_results/results_34_n_n/_rgb_model_best.pth.tar \
    bilinear_att ucf101 \
    ../temporal-segment-networks/data/ucf101_rgb_val_split_1.txt \
    UCF101_results/bilinear_results/sens \
    /home/xinqizhu/ucfTrainTestlist/classInd.txt \
    --arch resnet34 --num_segments 8 --modality RGB \
    --dropout 0.5 \
    --bi_out_dims 256 \
    --bi_rank 256 \
    --bi_add_clf \
    --bi_filter_size 1 \
    --bi_conv_dropout 0 \
    --bi_dropout 0

#CUDA_VISIBLE_DEVICES=1 python score_sens.py \
    #Something_results/bilinear_results/results_34_n_n/_rgb_model_best.pth.tar \
    #bilinear_att something \
    #../TRN-pytorch/video_datasets/something/something_rgb_val_split_1.txt \
    #Something_results/bilinear_results/sens \
    #../TRN-pytorch/video_datasets/something/category.txt \
    #--arch resnet34 --num_segments 8 --modality RGB \
    #--dropout 0.5 \
    #--bi_out_dims 256 \
    #--bi_rank 256 \
    #--bi_add_clf \
    #--bi_filter_size 1 \
    #--bi_conv_dropout 0 \
    #--bi_dropout 0
