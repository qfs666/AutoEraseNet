CUDA_VISIBLE_DEVICES=1,3 python train.py \
    --numOfWorkers 4 \
    --modelsSavePath "./runs2/20250618" \
    --logPath "./runs2/20250618/logs" \
    --batchSize 6 \
    --num_epochs 800 \
    --resume "/root/qfs/project/remove_watermark/EraseNet/runs2/20250618/last.pth" \
    --dataRoot "/root/qfs/project/remove_watermark/EraseNet/data/train_data"
