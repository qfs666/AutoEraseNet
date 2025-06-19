import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import cv2
from os import walk
from os.path import join
from torchvision.transforms import Compose, ToTensor, Resize,RandomApply
from torchvision import transforms
from torchvision.transforms.functional import crop
from filelock import FileLock
import json
import os
from data.generate_watermark_data import generate_tiled_watermark
# from generate_watermark_data import generate_tiled_watermark
# 文件路径
# ======== 替换成你真实的水印生成函数 =========
# ======== 工具函数 =========
def CheckImageFile(filename):
    return any(filename.endswith(ext) for ext in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP'])


    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
def ImageTransform(is_train=True):
    return Compose([
        ToTensor(),
    ])
# ======== 主数据集类（已改造） =========
class ErasingData(Dataset):
    def __init__(self, dataRoot, crop_size=512, num_crops=4,is_train=True,params_path = "params.json"):
        super(ErasingData, self).__init__()
        if is_train:
            dataRoot = os.path.join(dataRoot,"train")
        else:
            dataRoot = os.path.join(dataRoot,"val","images")
        self.imageFiles = [join(dataRootK, f) for dataRootK, _, fs in walk(dataRoot)
                           for f in fs if CheckImageFile(f)]
        if not is_train:
            self.maskFiles = [f.replace("images", "masks") for f in self.imageFiles]
            self.gtFiles = [f.replace("images", "gts") for f in self.imageFiles]
        self.crop_size = crop_size
        self.num_crops = num_crops
        self.ImgTrans = ImageTransform(True)

        self.is_train = is_train
        self.params_path = params_path
        self.params_lock_path = params_path+".lock"

    def pad_to_multiple(self,img, multiple=512, fill=255):
        W, H = img.size
        pad_w = (multiple - W % multiple) % multiple
        pad_h = (multiple - H % multiple) % multiple
        padded_img = Image.new("RGB", (W + pad_w, H + pad_h), (fill, fill, fill))
        padded_img.paste(img, (0, 0))
        return padded_img, (W, H), (W + pad_w, H + pad_h)

    def split_into_patches(self,img, patch_size=512):
        W, H = img.size
        patches = []
        positions = []
        for top in range(0, H, patch_size):
            for left in range(0, W, patch_size):
                crop_patch = crop(img, top, left, patch_size, patch_size)
                patches.append(crop_patch)
                positions.append((top, left))
        return patches, positions

    def reconstruct_from_patches(self,patches, positions, padded_size, orig_size):
        """
        patches: list of PIL.Image or Tensor (assume same size, 512×512)
        positions: list of (top, left) positions
        padded_size: (W, H) after padding
        orig_size: (W, H) original image size
        """
        import torchvision.transforms.functional as TF
        from PIL import Image

        W_pad, H_pad = padded_size
        full_img = Image.new("RGB", (W_pad, H_pad))

        for patch, (top, left) in zip(patches, positions):
            if isinstance(patch, torch.Tensor):
                patch = TF.to_pil_image(patch.cpu().clamp(0, 1))  # assume 0~1 float
            full_img.paste(patch, (left, top))

        # Crop back to original size
        return full_img.crop((0, 0, *orig_size))

    def __getitem__(self, index):
        img_path = self.imageFiles[index]
        img = Image.open(img_path).convert('RGB')
        W,H = img.size
        if self.is_train:
            if W < self.crop_size or H < self.crop_size:
                tmp_img = Image.new("RGB", (max(W, self.crop_size), max(H, self.crop_size)), (255, 255, 255))
                tmp_img.paste(img, (0, 0))
                img = tmp_img
                W,H = img.size
            # 生成整张水印图及 mask
            watermarked_img_full, mask_full,params = generate_tiled_watermark(img)
            params["img_path"] = img_path
            params["crop_position"] = []
            crops = []

            for _ in range(self.num_crops):
                top = random.randint(0, H - self.crop_size)
                left = random.randint(0, W - self.crop_size)
                input_crop = crop(watermarked_img_full, top, left, self.crop_size, self.crop_size)
                gt_crop = crop(img, top, left, self.crop_size, self.crop_size)
                mask_crop = crop(mask_full, top, left, self.crop_size, self.crop_size)
                crops.append((input_crop, gt_crop, mask_crop))
                params["crop_position"].append([top,left,self.crop_size,self.crop_size])
            
            top = random.randint(0, H - self.crop_size)
            left = random.randint(0, W - self.crop_size)
            input_crop = crop(watermarked_img_full, top, left, self.crop_size, self.crop_size)
            gt_crop = crop(img, top, left, self.crop_size, self.crop_size)
            mask_crop = crop(mask_full, top, left, self.crop_size, self.crop_size)
            maskx,masky = 0,0
            # 从0，1，2选出一个数字随机
            choice_id = random.choice([0,1,2])
            if choice_id==0:
                maskx,masky = random.randint(self.crop_size//2,self.crop_size),self.crop_size
            elif choice_id==1:
                maskx,masky = self.crop_size,random.randint(self.crop_size//2,self.crop_size)
            else:
                maskx,masky = random.randint(self.crop_size//2,self.crop_size),random.randint(self.crop_size//2,self.crop_size)
            input_crop = np.array(input_crop)
            gt_crop = np.array(gt_crop)
            mask_crop = np.array(mask_crop)
            input_crop[masky:,:,:] = 255
            input_crop[:,maskx:,:] = 255
            gt_crop[masky:,:,:] = 255
            gt_crop[:,maskx:,:] = 255
            mask_crop[masky:,:,:] = 255
            mask_crop[:,maskx:,:] = 255
            input_crop = Image.fromarray(input_crop)
            gt_crop = Image.fromarray(gt_crop)
            mask_crop = Image.fromarray(mask_crop)
            crops.append((input_crop, gt_crop, mask_crop))
            params["mask_crop_position"] = [[top,left,self.crop_size,self.crop_size],maskx,masky]

            # 随机补充一张负样本
            top = random.randint(0, H - self.crop_size)
            left = random.randint(0, W - self.crop_size)
            input_crop = crop(img, top, left, self.crop_size, self.crop_size)
            mask_full = Image.new("RGB",(W,H),(255,255,255))
            mask_crop = crop(mask_full, top, left, self.crop_size, self.crop_size)
            crops.append((input_crop, input_crop, mask_crop))
            params["crop_nativate_position"]=[top,left,self.crop_size,self.crop_size]
            # 添加写入文件锁，避免出现错误

            # 加锁写入
            with FileLock(self.params_lock_path):
                with open(self.params_path, "a") as f:
                    f.write(json.dumps(params) + "\n")

            # for i,(im,gt,mask) in enumerate(crops):
            #     save_transformed_image(self.train_ImgTrans(im),f"{i}_input.jpg")
            #     save_transformed_image(self.val_ImgTrans(gt),f"{i}_gt.jpg")
            #     save_transformed_image(self.mask_ImgTrans(mask),f"{i}_mask.jpg")
            
            return [(self.ImgTrans(input_crop),self.ImgTrans(gt_crop),self.ImgTrans(mask_crop)) for input_crop, gt_crop, mask_crop in crops]

        else:
            mask_img = Image.open(self.maskFiles[index]).convert('RGB')
            gt_img = Image.open(self.gtFiles[index]).convert('RGB')
            return [(self.ImgTrans(img),self.ImgTrans(gt_img),self.ImgTrans(mask_img))]


    def __len__(self):
        return len(self.imageFiles)



def custom_collate_fn(batch):
    input_crops = []
    gt_crops = []
    mask_crops = []
    paths = []

    for crops in batch:
        for input_crop, gt_crop, mask_crop in crops:
            input_crops.append(input_crop)
            gt_crops.append(gt_crop)
            mask_crops.append(mask_crop)

    return {
        'input_crops': torch.stack(input_crops),   # [B*num_crops, C, H, W]
        'gt_crops': torch.stack(gt_crops),
        'mask_crops': torch.stack(mask_crops),
    }


if __name__=="__main__":
    from torch.utils.data import DataLoader
    datas = ErasingData("/root/qfs/project/remove_watermark/EraseNet/data/train_data",512,4,True)
    loader = DataLoader(datas,2,False,collate_fn=custom_collate_fn)
    for i,out in enumerate(loader):
        print(out["input_crops"].shape)
        print(out["gt_crops"].shape)
        print(out["mask_crops"].shape)
