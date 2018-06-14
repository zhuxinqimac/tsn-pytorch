#CUDA_VISIBLE_DEVICES=1 python score_sens.py \
    #UCF101_results/lstm_results/results_34_n_n/_rgb_model_best.pth.tar \
    #lstm ucf101 \
    #--lstm_out_type avg --lstm_hidden_dims 256 \
    #../temporal-segment-networks/data/ucf101_rgb_val_split_1.txt \
    #UCF101_results/lstm_results/sens \
    #/home/xinqizhu/ucfTrainTestlist/classInd.txt \
    #--num_segments 16 --modality RGB \
    #--arch resnet34 

CUDA_VISIBLE_DEVICES=1 python score_sens.py \
    Something_results/lstm_results/results_34_n_n/_rgb_model_best.pth.tar \
    lstm something --lstm_out_type avg --lstm_hidden_dims 512 \
    ../TRN-pytorch/video_datasets/something/something_rgb_val_split_1.txt \
    Something_results/lstm_results/sens \
    ../TRN-pytorch/video_datasets/something/category.txt \
    --num_segments 16 --modality RGB \
    --arch resnet34 

