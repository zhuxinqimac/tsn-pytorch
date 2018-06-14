#CUDA_VISIBLE_DEVICES=1 python score_inf.py \
    #UCF101_results/lstm_results/results_34_n_n/_rgb_model_best.pth.tar \
    #UCF101_results/lstm_results/results_34_s_n/_rgb_model_best_bk.pth.tar \
    #lstm ucf101 --lstm_hidden_dims 256 --lstm_out_type avg \
    #../temporal-segment-networks/data/ucf101_rgb_val_split_1.txt \
    #UCF101_results/lstm_results/inf \
    #/home/xinqizhu/ucfTrainTestlist/classInd.txt \
    #--num_segments 16 --modality RGB \
    #--arch resnet34 

CUDA_VISIBLE_DEVICES=1 python score_inf.py \
    Something_results/lstm_results/results_34_n_n/_rgb_model_best.pth.tar \
    Something_results/lstm_results/results_34_s_s/_rgb_model_best.pth.tar \
    lstm something --lstm_hidden_dims 512 --lstm_out_type=avg \
    ../TRN-pytorch/video_datasets/something/something_rgb_val_split_1.txt \
    Something_results/lstm_results/inf \
    ../TRN-pytorch/video_datasets/something/category.txt \
    --num_segments 16 --modality RGB \
    --arch resnet34 
