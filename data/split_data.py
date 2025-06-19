import os
from PIL import Image
import random
import glob
from generate_watermark_data import generate_tiled_watermark

def pad_to_multiple(img, size=512):
    """将图像填充为 size 的整数倍，填充为白色"""
    w, h = img.size
    new_w = ((w + size - 1) // size) * size
    new_h = ((h + size - 1) // size) * size

    if new_w == w and new_h == h:
        return img

    padded = Image.new("RGB", (new_w, new_h), (255, 255, 255))  # 白色背景
    padded.paste(img, (0, 0))
    return padded

def split_image(img, size=512):
    """将图像分割为多个 size×size 的块"""
    w, h = img.size
    tiles = []
    for top in range(0, h, size):
        for left in range(0, w, size):
            box = (left, top, left + size, top + size)
            tile = img.crop(box)
            tiles.append(tile)
    return tiles

def process_directory(input_dir, output_dir, size=512):
    os.makedirs(output_dir, exist_ok=True)
    img_extensions = ['.png', '.jpg', '.jpeg', '.bmp']

    for filename in os.listdir(input_dir):
        if not any(filename.lower().endswith(ext) for ext in img_extensions):
            continue

        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path).convert("RGB")
        padded = pad_to_multiple(img, size)
        tiles = split_image(padded, size)

        base_name = os.path.splitext(filename)[0]
        for idx, tile in enumerate(tiles):
            tile.save(os.path.join(output_dir, f"{base_name}_{idx:03d}.jpg"))


if __name__=="__main__":
    # import subprocess
    # datas = glob.glob("/root/qfs/project/remove_watermark/EraseNet/data/train_data/train2/images/*.jpg")
    # save_path = "/root/qfs/project/remove_watermark/EraseNet/data/train_data/train2/val/source_images"
    # os.makedirs(save_path,exist_ok=True)
    # datas = random.sample(datas,int(2))
    # for data in datas:
    #     cmd = f"mv {data} {save_path}"
    #     subprocess.run(cmd,shell=True)
    # # 使用示例
    input_folder = "/root/qfs/project/remove_watermark/EraseNet/data/train_data/train2/val/source_images"
    output_folder = "/root/qfs/project/remove_watermark/EraseNet/data/train_data/train2/val/gts"
    # process_directory(input_folder, output_folder)
    datas = os.listdir(output_folder)
    no_process = random.sample(datas,int(0.2*len(datas)))
    output_watermark_path = "/root/qfs/project/remove_watermark/EraseNet/data/train_data/val/images"
    output_mask_path = "/root/qfs/project/remove_watermark/EraseNet/data/train_data/val/masks"
    os.makedirs(output_watermark_path,exist_ok=True)
    os.makedirs(output_mask_path,exist_ok=True)
    for data in datas:
        img = Image.open(os.path.join(output_folder,data))
        if data in no_process:
            mask = Image.new("RGB", img.size, (255,255,255))
            mask.save(os.path.join(output_mask_path,data))
            img.save(os.path.join(output_watermark_path,data))
        else:
            watermarked, mask_layer,watermark_params = generate_tiled_watermark(img)
            watermarked.save(os.path.join(output_watermark_path,data))
            mask_layer.save(os.path.join(output_mask_path,data))