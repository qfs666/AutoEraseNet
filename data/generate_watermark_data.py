from PIL import Image, ImageDraw, ImageFont,ImageOps
import random
import numpy as np
import os
import glob

all_data = open("/root/qfs/project/remove_watermark/EraseNet/data/dictionary.txt",'r').readlines()
CHINESE_DICT,NUM_DICT,ENGLISH_DICT = all_data
CHINESE_DICT,NUM_DICT,ENGLISH_DICT = str(CHINESE_DICT),str(NUM_DICT),str(ENGLISH_DICT)
FONT_LIST = glob.glob("/root/qfs/project/remove_watermark/EraseNet/data/data_font/chinese/*.ttf")


# 用于生成随机字符串（中英文数字混合）
def random_string(char_sources=[CHINESE_DICT,NUM_DICT,ENGLISH_DICT],length=8):
    # 所有字符集合放在一个列表中
    char_sources = char_sources
    # 总共要选多少个字符（1~5）
    total_chars = random.randint(2, length+1)

    # 所有可选字符整合到一个列表中（保持分类）
    available_chars = []
    result = []
    for _ in range(total_chars):
        selected_pool = random.choice(char_sources)
        result.append(random.choice(selected_pool))
    random.shuffle(result)  # 打乱顺序
    return ''.join(result)

# 主函数：生成平铺水印
def generate_tiled_watermark(image: Image.Image,
                             font_path=None,
                             font_size=None,
                             color=None,
                             spacing=None,
                             rotation=True,
                             text=None,
                             angle=None):
    """
    生成平铺水印 + 可记录参数
    """
    font_path = font_path or random.choice(FONT_LIST)
    font_size = font_size or random.randint(40,100)
    width, height = image.size
    image = image.convert("RGBA")
    watermark_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    mask_layer = Image.new("L", image.size, 0)
    spacing = spacing or random.randint(50,200)
    color = color or (random.randint(0,255), random.randint(0,255), random.randint(0,255), random.randint(45,180))
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        raise ValueError("字体路径无效，请提供有效的 .ttf 字体文件", e)

    # 使用给定的 text，否则随机生成
    if text is None:
        text = random_string()
    text_layer = Image.new("RGBA", (font_size * len(text), font_size * 2), (0, 0, 0, 0))
    text_draw = ImageDraw.Draw(text_layer)
    text_draw.text((0, 0), text, font=font, fill=color)

    text_mask = Image.new("L", text_layer.size, 0)
    mask_draw_t = ImageDraw.Draw(text_mask)
    mask_draw_t.text((0, 0), text, font=font, fill=255)

    # 控制 angle（手动传入或随机）
    if rotation:
        if angle is None:
            angle_choices = list(range(0, 360, 5))
            angle = random.choice(angle_choices)
        text_layer = text_layer.rotate(angle, expand=1)
        text_mask = text_mask.rotate(angle, expand=1)
    else:
        angle = None

    tw, th = text_layer.size

    for x in range(0, width + tw, spacing):
        for y in range(0, height + th, spacing):
            watermark_layer.paste(text_layer, (x, y), text_layer)
            mask_layer.paste(text_mask, (x, y), text_mask)

    watermarked = Image.alpha_composite(image, watermark_layer).convert("RGB")
    mask_layer = ImageOps.invert(mask_layer)  # 反转白黑区域
    mask_layer = mask_layer.convert("RGB")

    # 保存参数
    watermark_params = {
        "font_path": font_path,
        "font_size": font_size,
        "color": color,
        "spacing": spacing,
        "rotation": rotation,
        "text": text,
        "angle": angle,
        "width": width,
        "height": height
    }
    return watermarked, mask_layer,watermark_params

if __name__ == "__main__":
    import subprocess
    pass
    # path = "/root/qfs/project/remove_watermark/EraseNet/data/data_font/chinese"
    # for i in os.listdir(path):
    #     img = Image.open("/root/qfs/project/remove_watermark/EraseNet/data/20250613-151102.jpg")
    #     watermarked, mask,params = generate_tiled_watermark(img, font_path=os.path.join(path,i), font_size=36,color=(0,0,0,180))
    #     mask.save(f"./out/{i.replace('.ttf','.jpg')}")
    #     watermarked.save(f"./out/{i.replace('.ttf','.png')}")
    #     print(params)
    #     break
        
    # dict_ = {'font_path': '/root/qfs/project/remove_watermark/EraseNet/data/data_font/chinese/ARDCCaiShenXuanXingShuGB.ttf', 'font_size': 36, 'color': (0, 0, 0, 180), 'spacing': 150, 'rotation': True, 'text': '拙茵\n拍苗拽O', 'angle': 310}
    # img = Image.open("/root/qfs/project/remove_watermark/EraseNet/data/20250613-151102.jpg")
    # watermarked, mask,params = generate_tiled_watermark(img, **dict_)
    # mask.save(f"./zzz.jpg")
    # watermarked.save(f"./zzz.png")
