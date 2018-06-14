#CUDA_VISIBLE_DEVICES=1 python main.py  ucf101 RGB \
    #../temporal-segment-networks/data/ucf101_rgb_train_split_1.txt \
    #../temporal-segment-networks/data/ucf101_rgb_val_split_1.txt \
    #--arch resnet34 --num_segments 16 --gd 20  --lr 0.001 --lr_steps 30 60 70 \
    #--epochs 80 -b 32 -j 2 --dropout 0.8 --consensus_type conv_lstm \
    #--result_path UCF101_results/conv_lstm_results/results_34_n_n \
    #--lstm_out_type avg --conv_lstm_kernel 3 \
    #--resume UCF101_results/conv_lstm_results/results_34_n_n/_rgb_model_best.pth.tar \
    #--evaluate --val_reverse
CUDA_VISIBLE_DEVICES=1 python main.py  something RGB \
    ../TRN-pytorch/video_datasets/something/something_rgb_train_split_1.txt \
    ../TRN-pytorch/video_datasets/something/something_rgb_val_split_1.txt \
    --arch resnet34 --num_segments 16 --gd 20  --lr 0.001 --lr_steps 30 60 70 \
    --epochs 80 -b 32 -j 1 --dropout 0.8 \
    --consensus_type conv_lstm \
    --result_path Something_results/conv_lstm_results/results_34_s_r \
    --lstm_out_type avg --conv_lstm_kernel 3 --lstm_hidden_dims 512 \
    --resume Something_results/conv_lstm_results/results_34_s_s/_rgb_model_best.pth.tar \
    --evaluate --val_reverse
