CUDA_VISIBLE_DEVICES=1 python score_inf.py \
    Something_results/avg_results/results_34_n_n/_rgb_model_best.pth.tar \
    Something_results/avg_results/results_34_s_s/_rgb_model_best.pth.tar \
    avg something \
    ../TRN-pytorch/video_datasets/something/something_rgb_val_split_1.txt \
    Something_results/avg_results/inf \
    ../TRN-pytorch/video_datasets/something/category.txt \
    --num_segments 3 --modality RGB \
    --arch resnet34 
