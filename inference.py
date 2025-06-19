from PIL import Image
import torch
import numpy as np
import os
from models.sa_gan import STRnet2

def pad_to_multiple(img, size=512):
    w, h = img.size
    new_w = ((w + size - 1) // size) * size
    new_h = ((h + size - 1) // size) * size
    padded = Image.new("RGB", (new_w, new_h), (255, 255, 255))
    padded.paste(img, (0, 0))
    print(padded.size)
    return padded, (w, h)

def split_image_with_overlap(img, tile_size=512, overlap=128):
    w, h = img.size
    step = tile_size - overlap
    tiles = []
    positions = []
    
    for top in range(0, h, step):
        if top + tile_size > h:
            top = h - tile_size
        for left in range(0, w, step):
            if left + tile_size > w:
                left = w - tile_size
            box = (left, top, left + tile_size, top + tile_size)
            tile = img.crop(box)
            if (left, top) not in positions:  # 避免重复添加
                tiles.append(tile)
                positions.append((left, top))
    
    return tiles, positions

def get_2d_weight(tile_size, overlap):
    w = np.ones(tile_size, dtype=np.float32)
    ramp = np.linspace(0, 1, overlap, endpoint=False)
    w[:overlap] = ramp
    w[-overlap:] = ramp[::-1]
    weight_2d = np.outer(w, w)
    return np.stack([weight_2d]*3, axis=-1)  # (tile_size, tile_size, 3)

def merge_tiles_with_weighted_blending(tiles, positions, original_size, tile_size=512, overlap=128):
    orig_w, orig_h = original_size
    padded_w = ((orig_w + tile_size - 1) // tile_size) * tile_size
    padded_h = ((orig_h + tile_size - 1) // tile_size) * tile_size

    result = np.zeros((padded_h, padded_w, 3), dtype=np.float32)
    weight_sum = np.zeros_like(result)

    weight_2d = get_2d_weight(tile_size, overlap)

    for tile, (x, y) in zip(tiles, positions):
        tile_np = np.array(tile).astype(np.float32) / 255.0
        h, w = tile_np.shape[:2]
        weight = weight_2d[:h, :w, :]
        result[y:y+h, x:x+w, :] += tile_np * weight
        weight_sum[y:y+h, x:x+w, :] += weight

    weight_sum = np.maximum(weight_sum, 1e-6)
    result = result / weight_sum
    result = (result[:orig_h, :orig_w, :] * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(result)

def infer_image(image_path, netG, device, batch_size=16, tile_size=512, overlap=128, save_dir='./inferred'):
    os.makedirs(save_dir, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    padded_img, original_size = pad_to_multiple(image, tile_size)
    tiles, positions = split_image_with_overlap(padded_img, tile_size, overlap)
    # for i,im in enumerate(tiles):
    #     im.save(f'tile_{i}.png')
    tensor_tiles = [torch.tensor(np.array(tile)).permute(2, 0, 1).float() / 255.0 for tile in tiles]
    tensor_tiles = torch.stack(tensor_tiles)

    all_outputs = []
    netG.eval()
    with torch.no_grad():
        for i in range(0, len(tensor_tiles), batch_size):
            batch = tensor_tiles[i:i+batch_size].to(device)
            _, _, _, g_images, _ = netG(batch)
            out_images = g_images.cpu().clamp(0, 1)
            all_outputs.extend(out_images)

    result_tiles = [Image.fromarray((tile.permute(1, 2, 0).numpy() * 255).astype(np.uint8)) for tile in all_outputs]
    merged_result = merge_tiles_with_weighted_blending(result_tiles, positions, original_size, tile_size, overlap)
    merged_result.save(os.path.join(save_dir, os.path.basename(image_path)))

if __name__ == "__main__":
    netG = STRnet2(3)
    weights = torch.load('/root/qfs/project/remove_watermark/EraseNet/runs2/20250618/last.pth')
    print(weights.keys())
    netG.load_state_dict(weights["netG"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG.to(device)
    while True:
        image_path = input("输入图片路径：")
        infer_image(image_path, netG, device,overlap=256)
