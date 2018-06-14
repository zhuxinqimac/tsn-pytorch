#CUDA_VISIBLE_DEVICES=1 python score_inf.py \
    #UCF101_results/conv_lstm_results/results_34_n_n/_rgb_model_best.pth.tar \
    #UCF101_results/conv_lstm_results/results_34_s_n/_rgb_model_best.pth.tar \
    #conv_lstm ucf101 \
    #--lstm_out_type avg --conv_lstm_kernel 3 --lstm_hidden_dims 512 \
    #../temporal-segment-networks/data/ucf101_rgb_val_split_1.txt \
    #UCF101_results/conv_lstm_results/inf \
    #/home/xinqizhu/ucfTrainTestlist/classInd.txt \
    #--num_segments 16 --modality RGB \
    #--arch resnet34 

CUDA_VISIBLE_DEVICES=1 python score_inf.py \
    Something_results/conv_lstm_results/results_34_n_n/_rgb_model_best.pth.tar \
    Something_results/conv_lstm_results/results_34_s_s/_rgb_model_best.pth.tar \
    conv_lstm something \
    --lstm_out_type avg --conv_lstm_kernel 3 --lstm_hidden_dims 512 \
    ../TRN-pytorch/video_datasets/something/something_rgb_val_split_1.txt \
    Something_results/conv_lstm_results/inf \
    ../TRN-pytorch/video_datasets/something/category.txt \
    --num_segments 16 --modality RGB \
    --arch resnet34 
