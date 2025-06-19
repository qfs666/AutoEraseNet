import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from PIL import Image
import numpy as np

def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram

def visual(image):
    im = image.transpose(1,2).transpose(2,3).detach().cpu().numpy()
    Image.fromarray(im[0].astype(np.uint8)).show()

def dice_loss(input, target):
    input = torch.sigmoid(input)

    input = input.contiguous().view(input.size()[0], -1)
    target = target.contiguous().view(target.size()[0], -1)
    
    input = input 
    target = target

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    d = (2 * a) / (b + c)
    dice_loss = torch.mean(d)
    return 1 - dice_loss

class LossWithGAN_STE(nn.Module):
    def __init__(self, logPath, extractor, Lamda, netD, D_optimizer):
        super(LossWithGAN_STE, self).__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor
        self.discriminator = netD    ## local_global sn patch gan
        self.D_optimizer = D_optimizer
        self.lamda = Lamda
        self.writer = SummaryWriter(logPath)

    def forward(self, input, mask, x_o1, x_o2, x_o3, output, mm, gt, count, epoch):
        ##############################
        #       Train Discriminator
        ##############################
        self.discriminator.zero_grad()

        # 判别器：真实图像 vs 伪造图像（detach 防止梯度回传到 G）
        D_real = self.discriminator(gt.detach(), mask.detach())
        D_fake_for_D = self.discriminator(output.detach(), mask.detach())

        D_real_loss = torch.mean(F.relu(1. - D_real))
        D_fake_loss = torch.mean(F.relu(1. + D_fake_for_D))
        D_loss = D_real_loss + D_fake_loss

        self.D_optimizer.zero_grad()
        D_loss.backward()
        self.D_optimizer.step()

        self.writer.add_scalar('LossD/Discriminator loss', D_loss.item(), count)

        ##############################
        #       Train Generator
        ##############################
        D_fake_for_G = self.discriminator(output, mask)  # 用于 G 的 adversarial loss
        adv_loss = -torch.mean(D_fake_for_G)

        # Inpainting losses
        output_comp = mask * input + (1 - mask) * output
        hole_loss = 10 * self.l1((1 - mask) * output, (1 - mask) * gt)
        valid_loss = 2 * self.l1(mask * output, mask * gt)

        # mask loss (dice)
        mask_loss = dice_loss(mm, 1 - mask)

        # Multi-scale reconstruction loss
        masks_a = F.interpolate(mask, scale_factor=0.25)
        masks_b = F.interpolate(mask, scale_factor=0.5)
        imgs1 = F.interpolate(gt, scale_factor=0.25)
        imgs2 = F.interpolate(gt, scale_factor=0.5)
        msr_loss = (
            8 * self.l1((1 - mask) * x_o3, (1 - mask) * gt) +
            0.8 * self.l1(mask * x_o3, mask * gt) +
            6 * self.l1((1 - masks_b) * x_o2, (1 - masks_b) * imgs2) +
            1 * self.l1(masks_b * x_o2, masks_b * imgs2) +
            5 * self.l1((1 - masks_a) * x_o1, (1 - masks_a) * imgs1) +
            0.8 * self.l1(masks_a * x_o1, masks_a * imgs1)
        )

        # Perceptual & Style losses
        feat_output_comp = self.extractor(output_comp)
        feat_output = self.extractor(output)
        feat_gt = self.extractor(gt)

        perceptual_loss = 0.0
        style_loss = 0.0
        for i in range(3):
            perceptual_loss += 0.01 * self.l1(feat_output[i], feat_gt[i])
            perceptual_loss += 0.01 * self.l1(feat_output_comp[i], feat_gt[i])

            style_loss += 120 * self.l1(gram_matrix(feat_output[i]), gram_matrix(feat_gt[i]))
            style_loss += 120 * self.l1(gram_matrix(feat_output_comp[i]), gram_matrix(feat_gt[i]))

        # 总损失
        total_G_loss = (
            msr_loss +
            hole_loss +
            valid_loss +
            perceptual_loss +
            style_loss +
            0.1 * adv_loss +
            1.0 * mask_loss
        )

        # 日志记录
        self.writer.add_scalar('LossG/Hole loss', hole_loss.item(), count)
        self.writer.add_scalar('LossG/Valid loss', valid_loss.item(), count)
        self.writer.add_scalar('LossG/MSR loss', msr_loss.item(), count)
        self.writer.add_scalar('LossPrc/Perceptual loss', perceptual_loss.item(), count)
        self.writer.add_scalar('LossStyle/Style loss', style_loss.item(), count)
        self.writer.add_scalar('LossMask/Dice mask loss', mask_loss.item(), count)
        self.writer.add_scalar('LossGAN/Adversarial loss', adv_loss.item(), count)
        self.writer.add_scalar('Generator/Total loss', total_G_loss.item(), count)

        return total_G_loss

    
