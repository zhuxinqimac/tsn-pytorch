#CUDA_VISIBLE_DEVICES=0,3 python main.py ucf101 RGB \
    #../temporal-segment-networks/data/ucf101_rgb_train_split_1.txt \
    #../temporal-segment-networks/data/ucf101_rgb_val_split_1.txt \
    #--arch resnet34 --num_segments 8 --gd 20  --lr 0.001 --lr_steps 20 25 30 \
    #--epochs 35 -b 64 -j 4 --dropout 0.5 \
    #--consensus_type bilinear_att \
    #--result_path UCF101_results/bilinear_results/results_34_s_s \
    #--bi_out_dims 256 \
    #--bi_rank 256 \
    #--bi_add_clf \
    #--bi_filter_size 1 \
    #--bi_conv_dropout 0 \
    #--bi_dropout 0 \
    #--train_shuffle --val_shuffle

    #--arch resnet34 --num_segments 8 --gd 20 --lr 0.001 --lr_steps 30 60 70 \
    #--arch resnet34 --num_segments 8 --gd 20 --lr 0.001 --lr_steps 16 20 24 26\
    #--bi_add_clf \
    #--result_path Something_results/bilinear_results/para_search/scatter_seg_8_rank_256_no_relu_filter_1_addclf256_clfdrop05_lrelu_fast_initnorm \

CUDA_VISIBLE_DEVICES=0 python main.py something RGB \
    ../TRN-pytorch/video_datasets/something/something_rgb_train_split_1.txt \
    ../TRN-pytorch/video_datasets/something/something_rgb_val_split_1.txt \
    --arch resnet34 --num_segments 32 --gd 20 --lr 0.001 --lr_steps 15 20 25 \
    --epochs 30 -b 16 -j 4 --dropout 0.5 \
    --consensus_type bilinear_multi_top \
    --result_path Something_results/bi_multi_top_results/para_search/layer2_r64_f2_scv64_r64_f2_scv64
