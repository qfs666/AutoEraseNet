import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from models.sa_gan import STRnet2
from models.discriminator import Discriminator_STE
from models.Model import DINOv2FeatureExtractor
from loss.Loss import LossWithGAN_STE
from data.dataloader import ErasingData, custom_collate_fn
from calculalte_psnr_ssim import compute_batch_psnr, compute_batch_ssim

# ------------------------
# 解析参数
# ------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--modelsSavePath', type=str, default='./checkpoints', help='path to save models')
parser.add_argument('--dataRoot', type=str, default='', help='path to dataset')
parser.add_argument('--logPath', type=str, default='./logs')
parser.add_argument('--batchSize', type=int, default=16)
parser.add_argument('--loadSize', type=int, default=512)
parser.add_argument('--pretrained', type=str, default='', help='path to pretrained netG weights (only netG)')
parser.add_argument('--resume', type=str, default='', help='path to checkpoint to resume training')
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--numOfWorkers', type=int, default=4)
args = parser.parse_args()

# ------------------------
# 初始化
# ------------------------
torch.set_num_threads(5)
cudnn.benchmark = True
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
os.makedirs(args.modelsSavePath, exist_ok=True)

# ------------------------
# 数据加载
# ------------------------
Erase_data = ErasingData(args.dataRoot, args.loadSize, num_crops=4, is_train=True, params_path=os.path.join(args.modelsSavePath, "params.json"))
Erase_loader = DataLoader(Erase_data, batch_size=args.batchSize, shuffle=True, num_workers=args.numOfWorkers, collate_fn=custom_collate_fn)

val_data = ErasingData(args.dataRoot, args.loadSize, num_crops=4, is_train=False)
Val_loader = DataLoader(val_data, batch_size=args.batchSize * 5, shuffle=False, num_workers=args.numOfWorkers, collate_fn=custom_collate_fn)

# ------------------------
# 模型定义
# ------------------------
netG = STRnet2(3).to(device)
netD = Discriminator_STE(3).to(device)


# ------------------------
# 优化器 & LR Scheduler
# ------------------------
G_optimizer = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
D_optimizer = optim.Adam(netD.parameters(), lr=1e-5, betas=(0.0, 0.9))

def get_lr_lambda(start_decay_epoch, total_epochs):
    return lambda epoch: 1.0 if epoch < start_decay_epoch else 1.0 - float(epoch - start_decay_epoch) / (total_epochs - start_decay_epoch)

lr_scheduler_G = optim.lr_scheduler.LambdaLR(G_optimizer, lr_lambda=get_lr_lambda(100, args.num_epochs))

# ------------------------
# 损失函数
# ------------------------
criterion = LossWithGAN_STE(args.logPath, DINOv2FeatureExtractor(), Lamda=10.0, netD=netD, D_optimizer=D_optimizer).to(device)

# ------------------------
# 恢复训练
# ------------------------
start_epoch = 1
max_psnr = 0
if args.resume!="":
    checkpoint = torch.load(args.resume)
    netG.load_state_dict(checkpoint['netG'])
    netD.load_state_dict(checkpoint['netD'])
    G_optimizer.load_state_dict(checkpoint['G_optimizer'])
    D_optimizer.load_state_dict(checkpoint['D_optimizer'])
    lr_scheduler_G.load_state_dict(checkpoint['lr_scheduler_G'])
    start_epoch = checkpoint['epoch'] + 1
    # max_psnr = checkpoint.get('psnr', 0)
    max_psnr = 0
    print(f"Resumed from {args.resume} (epoch {start_epoch-1})")

if args.pretrained!="":
    checkpoint = torch.load(args.pretrained)
    netG.load_state_dict(checkpoint["netG"])
    netD.load_state_dict(checkpoint['netD'])
    print(f"Loaded pretrained netG from {args.pretrained}")


if torch.cuda.device_count() > 1:
    netG = nn.DataParallel(netG)
    netD = nn.DataParallel(netD)
# ------------------------
# 训练循环
# ------------------------
for epoch in range(start_epoch, args.num_epochs + 1):
    netG.train()
    for step, label_info in enumerate(Erase_loader):
        imgs, gt, masks = label_info["input_crops"].to(device), label_info["gt_crops"].to(device), label_info["mask_crops"].to(device)
        
        netG.zero_grad()
        x_o1, x_o2, x_o3, fake_images, mm = netG(imgs)
        G_loss = criterion(imgs, masks, x_o1, x_o2, x_o3, fake_images, mm, gt, step, epoch).sum()
        G_loss.backward()
        G_optimizer.step()

        print(f"[Epoch {epoch} Step {step} / {len(Erase_loader)}] G_loss: {G_loss.item():.4f}")

    if epoch%2==0:
        # ------------------------
        # 验证
        # ------------------------
        netG.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        total_samples = 0
        with torch.no_grad():
            for val_info in Val_loader:
                imgs = val_info["input_crops"].to(device)
                gts = val_info["gt_crops"].to(device)
                masks = val_info["mask_crops"].to(device)
                _, _, _, fake_images, _ = netG(imgs)
                fake_images = torch.clamp(fake_images, 0, 1)
                gts = torch.clamp(gts, 0, 1)
                psnr = compute_batch_psnr(fake_images, gts)
                ssim = compute_batch_ssim(fake_images, gts)
                batch_size = imgs.size(0)
                total_psnr += psnr * batch_size
                total_ssim += ssim * batch_size
                total_samples += batch_size

        avg_psnr = total_psnr / total_samples
        avg_ssim = total_ssim / total_samples
        print(f"==> Val Epoch {epoch}: PSNR={avg_psnr:.2f}, SSIM={avg_ssim:.4f}")
        lr_scheduler_G.step()

        # ------------------------
        # 保存模型
        # ------------------------
        def save_checkpoint(name):
            torch.save({
                'epoch': epoch,
                'netG': netG.module.state_dict() if isinstance(netG, nn.DataParallel) else netG.state_dict(),
                'netD': netD.module.state_dict() if isinstance(netD, nn.DataParallel) else netD.state_dict(),
                'G_optimizer': G_optimizer.state_dict(),
                'D_optimizer': D_optimizer.state_dict(),
                'lr_scheduler_G': lr_scheduler_G.state_dict(),
                'psnr': avg_psnr,
                'ssim': avg_ssim,
            }, os.path.join(args.modelsSavePath, f'{name}.pth'))

        save_checkpoint("last")
        if avg_psnr > max_psnr:
            max_psnr = avg_psnr
            save_checkpoint("best")
            with open(os.path.join(args.modelsSavePath, 'best.txt'), 'w') as f:
                f.write(f'Best Epoch: {epoch}, PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}')
