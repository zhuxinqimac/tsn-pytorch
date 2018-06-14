CUDA_VISIBLE_DEVICES=1 python main.py  ucf101 RGB \
    ../temporal-segment-networks/data/ucf101_rgb_train_split_1.txt \
    ../temporal-segment-networks/data/ucf101_rgb_val_split_1.txt \
    --arch resnet34 --num_segments 16 --gd 20  --lr 0.001 --lr_steps 30 60 70 \
    --epochs 80 -b 4 -j 2 --dropout 0.8 --consensus_type lstm \
    --result_path UCF101_results/contras_lstm_results/results_34_n_n \
    --lstm_hidden_dims 512 --lstm_out_type avg \
    --contrastive_mode --contras_m1 0.02 --contras_m2 0.02
    #--train_shuffle

#CUDA_VISIBLE_DEVICES=0,3 python main.py  something RGB \
    #../TRN-pytorch/video_datasets/something/something_rgb_train_split_1.txt \
    #../TRN-pytorch/video_datasets/something/something_rgb_val_split_1.txt \
    #--arch resnet34 --num_segments 16 --gd 20  --lr 0.001 --lr_steps 30 60 70 \
    #--epochs 80 -b 32 -j 4 --dropout 0.8 --consensus_type lstm \
    #--result_path Something_results/lstm_results/results_34_s_s \
    #--train_shuffle --val_shuffle
    ##--lstm_hidden_dims 256 
