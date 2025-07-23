import os
from pathlib import Path

# ======= 你只需要改下面这两项 =======
FOLDER_PATH = "/Users/ruiyuhan/Desktop/new"  # 改成你的文件夹路径
DIGITS = 5  # 重命名编号的位数，例如 3 表示 001, 002...
# ====================================

def clean_and_rename(folder: str, digits: int = 3):
    folder_path = Path(folder).expanduser().resolve()
    if not folder_path.is_dir():
        print(f"❌ 文件夹不存在: {folder_path}")
        return

    allowed_ext = {".jpg", ".jpeg", ".png"}
    images = []

    # 删除非图片文件
    for f in folder_path.iterdir():
        if f.is_file():
            if f.suffix.lower() in allowed_ext:
                images.append(f)
            else:
                print(f"🗑 删除非图片文件: {f.name}")
                f.unlink()

    # 重命名剩下的图片
    images.sort()
    for idx, img in enumerate(images, 1):
        new_name = f"{idx:0{digits}d}{img.suffix.lower()}"
        new_path = folder_path / new_name
        print(f"🔄 重命名: {img.name} → {new_name}")
        img.rename(new_path)

    print(f"✅ 完成！共保留 {len(images)} 张图片。")

# 执行函数
clean_and_rename(FOLDER_PATH, DIGITS)
