import torch
import torch.nn.functional as F
from pytorch_msssim import ssim

def compute_batch_psnr(generated, gt, max_val=1.0):
    mse = F.mse_loss(generated, gt, reduction='none')
    mse = mse.view(mse.size(0), -1).mean(dim=1)  # 每张图的 MSE
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.mean().item()

def compute_batch_ssim(generated, gt):
    return ssim(generated, gt, data_range=1.0, size_average=True).item()


def evaluate_dataset_psnr_ssim(model, dataloader, device):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            imgs, gts, *_ = batch  # 或根据你的 dataset 格式解包
            imgs = imgs.to(device)
            gts = gts.to(device)

            *_, g_images, _ = model(imgs)

            # 若模型输出是 [-1, 1]，请先映射到 [0, 1]
            g_images = torch.clamp(g_images, 0, 1)
            gts = torch.clamp(gts, 0, 1)

            batch_size = gts.size(0)
            psnr = compute_batch_psnr(g_images, gts)
            ssim_val = compute_batch_ssim(g_images, gts)

            total_psnr += psnr * batch_size
            total_ssim += ssim_val * batch_size
            total_samples += batch_size

    avg_psnr = total_psnr / total_samples
    avg_ssim = total_ssim / total_samples
    return avg_psnr, avg_ssim
