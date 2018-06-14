CUDA_VISIBLE_DEVICES=1 python score_inf.py \
    Something_results/optical_results/results_34_n_n/_flow_model_best.pth.tar \
    Something_results/optical_results/results_34_s_s/_flow_model_best.pth.tar \
    avg something \
    ../TRN-pytorch/video_datasets/something/something_flow_nozero_val_split_1.txt \
    Something_results/optical_results/inf \
    ../TRN-pytorch/video_datasets/something/category.txt \
    --num_segments 1 --modality Flow \
    --arch resnet34 --flow_prefix flow_

#CUDA_VISIBLE_DEVICES=1 python score_inf.py \
    #UCF101_results/optical_results/results_34_n_n/_flow_model_best.pth.tar \
    #UCF101_results/optical_results/results_34_s_s/_flow_model_best.pth.tar \
    #avg ucf101 \
    #../temporal-segment-networks/data/ucf101_flow_val_split_1.txt \
    #UCF101_results/optical_results/inf \
    #/home/xinqizhu/ucfTrainTestlist/classInd.txt \
    #--num_segments 1 --modality Flow \
    #--arch resnet34 --flow_prefix flow_
