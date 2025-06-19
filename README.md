# 项目名称（Auto-EraseNet）

> 基于 GAN 的图像水印擦除模型，支持训练和推理。

---

## 🔧 项目简介 / Features

- 使用基于 STRNet 和 EraseNet 改进的图像擦除网络；
- 支持批量水印生成、mask 同步生成；
- 提供完整的训练代码、验证脚本与评估指标（PSNR / SSIM）；
- 支持分块训练与整图拼接恢复；
- 使用 [DINOv2](https://github.com/facebookresearch/dinov2) / VGG16 提取特征感知损失。

---

## 📂 数据格式说明

```bash
原始图像目录结构如下：
train_data/
├──train
  ├── image1.jpg
  ├── image2.jpg
  ├── ...
├──val
  ├── images
    ├── image1.jpg
    ├── image2.jpg
    ├── ...
  ├── masks
    ├── image1.jpg
    ├── image2.jpg
    ├── ...
  ├── gts
    ├── image1.jpg
    ├── image2.jpg
    ├── ...
```

## 训练脚本

```bash
CUDA_VISIBLE_DEVICES=1,3 python train.py \
    --numOfWorkers 4 \
    --modelsSavePath "./runs2/" \
    --logPath "./runs2/logs" \
    --batchSize 6 \
    --num_epochs 800 \
    --resume "./runs2/last.pth" \
    --dataRoot "./data/train_data"
```

## 推理脚本

```bash
CUDA_VISIBLE_DEVICES=1,3 python inference.py
```

## 参考
https://github.com/lcy0604/EraseNet  
https://github.com/facebookresearch/dinov2