CUDA_VISIBLE_DEVICES=1 python visual_bi_feature.py ucf101 RGB \
    ../temporal-segment-networks/data/ucf101_rgb_train_split_1.txt \
    ../temporal-segment-networks/data/ucf101_rgb_val_split_1.txt \
    --arch resnet34 --num_segments 8 --gd 20  --lr 0.001 --lr_steps 15 20 25 \
    --epochs 30 -b 6 -j 4 --dropout 0.5 \
    --consensus_type bilinear_att \
    --class_index /home/xinqizhu/ucfTrainTestlist/classInd.txt \
    --result_path UCF101_results/bilinear_results/results_34_n_n \
    --resume UCF101_results/bilinear_results/results_34_n_n/_rgb_model_best.pth.tar \
    --evaluate \
    --bi_out_dims 256 \
    --bi_rank 256 \
    --bi_add_clf \
    --bi_att_softmax \
    --bi_filter_size 1 \
    --bi_conv_dropout 0 \
    --bi_dropout 0 
#CUDA_VISIBLE_DEVICES=1 python visual_bi_feature.py something RGB \
    #../TRN-pytorch/video_datasets/something/something_rgb_train_split_1.txt \
    #../TRN-pytorch/video_datasets/something/something_rgb_val_split_1.txt \
    #--arch resnet34 --num_segments 8 --gd 20  --lr 0.001 --lr_steps 15 20 25 \
    #--epochs 30 -b 6 -j 4 --dropout 0.5 \
    #--consensus_type bilinear_att \
    #--class_index ../TRN-pytorch/video_datasets/something/category.txt \
    #--result_path Something_results/bilinear_results/results_34_n_n \
    #--resume Something_results/bilinear_results/results_34_n_n/_rgb_model_best.pth.tar \
    #--evaluate \
    #--bi_out_dims 256 \
    #--bi_rank 256 \
    #--bi_add_clf \
    #--bi_att_softmax \
    #--bi_filter_size 1 \
    #--bi_conv_dropout 0 \
    #--bi_dropout 0 
