import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoImageProcessor, Dinov2Model

#VGG16 feature extract
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
      #  vgg16.load_state_dict(torch.load('./vgg16-397923af.pth'))
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


# DINOv2 特征提取器
class DINOv2FeatureExtractor(nn.Module):
    def __init__(self, model_name="facebook/dinov2-base"):
        super(DINOv2FeatureExtractor, self).__init__()
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = Dinov2Model.from_pretrained(self.model_name)

        # 冻结参数
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, image):
        """
        image: torch.Tensor of shape [B, C, H, W] in range [0, 1] or PIL image list
        returns: [feat_stage1, feat_stage2, feat_stage3]
        """
        # 如果是 tensor，需转为 [0, 255] 且转 int 类型，因为 DINOv2 processor 期望原始图像格式
        if isinstance(image, torch.Tensor):
            image = (image * 255).byte()  # to [0, 255]
            image = [img.permute(1, 2, 0).cpu().numpy() for img in image]  # to HWC for each image

        inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # 获取中间层特征（DINOv2 一般有 12/24 层，取其中三层）
        # hidden_states 是一个 list: [embedding_output, layer1, ..., layerN]
        hidden_states = outputs.hidden_states  # shape: (L+1, B, num_tokens, C)
        # 选择第4层、第8层、第12层（或更深）作为中间特征
        feat1 = hidden_states[4][:, 1:,:]   # 第4层Patch token
        feat2 = hidden_states[8][:, 1:,:]   # 第8层Patch token
        feat3 = hidden_states[12][:, 1:,:]  # 第12层Patch token

        # 根据需要可以reshape成图片空间特征，比如：
        # B, N, C -> B, C, H_patch, W_patch
        # 假设 patch_size=16，图像大小是512x512，则 H_patch = W_patch = 512/16 = 32
        B, N, C = feat1.shape
        H = W = int(N ** 0.5)  # 例如32
        feat1 = feat1.permute(0, 2, 1).reshape(B, C, H, W)
        feat2 = feat2.permute(0, 2, 1).reshape(B, C, H, W)
        feat3 = feat3.permute(0, 2, 1).reshape(B, C, H, W)
        return [feat1, feat2, feat3]



if __name__ == "__main__":
    net = DINOv2FeatureExtractor()
    x = torch.randn(1,3,512,512)
    out = net(x) 
    for i in out:
        print(i.shape)