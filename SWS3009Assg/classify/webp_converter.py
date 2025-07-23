import os
from PIL import Image

def convert_and_delete_webp(folder_path):
    if not os.path.exists(folder_path):
        print(f"❌ 文件夹不存在：{folder_path}")
        return

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.webp'):
            webp_path = os.path.join(folder_path, filename)
            jpg_path = os.path.splitext(webp_path)[0] + '.jpg'

            try:
                with Image.open(webp_path) as img:
                    rgb_img = img.convert('RGB')  # 去除透明通道
                    rgb_img.save(jpg_path, 'JPEG')
                os.remove(webp_path)  # 删除原始webp文件
                print(f"✅ 转换并删除：{filename} → {os.path.basename(jpg_path)}")
            except Exception as e:
                print(f"❌ 处理失败：{filename}，错误：{e}")

# 🧪 使用示例
folder = "/Users/ruiyuhan/Desktop/NUS Deep Learning/SWS3009Assg/cat_datasets/sphynx"  # 替换为你的实际路径
convert_and_delete_webp(folder)
