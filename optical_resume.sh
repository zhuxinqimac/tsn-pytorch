CUDA_VISIBLE_DEVICES=0,3 python main.py something Flow \
    ../TRN-pytorch/video_datasets/something/something_flow_nozero_train_split_1.txt \
    ../TRN-pytorch/video_datasets/something/something_flow_nozero_val_split_1.txt \
    --arch resnet34 --num_segments 1 --gd 20  --lr 0.001 --lr_steps 190 300 \
    --epochs 340 -b 128 -j 4 --dropout 0.7 --consensus_type avg \
    --result_path Something_results/optical_results/results_34_s_s \
    --resume Something_results/optical_results/results_34_s_s/_flow_model_best.pth.tar \
    --val_shuffle --train_shuffle
