import os

def rename_images_in_folder(folder_path, prefix="Pallas_cats"):
    """
    将文件夹中的所有图片按格式重命名：prefix_1.jpg, prefix_2.jpg ...
    :param folder_path: 图片所在文件夹路径
    :param prefix: 重命名的前缀
    """
    if not os.path.exists(folder_path):
        print(f"❌ 文件夹不存在: {folder_path}")
        return

    # 只处理图片文件（jpg、jpeg、png）
    files = [f for f in os.listdir(folder_path)
             if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    files.sort()  # 保持顺序

    for idx, filename in enumerate(files, start=1):
        ext = os.path.splitext(filename)[1].lower()
        new_name = f"{prefix}_{idx}{ext}"
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)
        os.rename(src, dst)
        print(f"✅ 已重命名: {filename} → {new_name}")

# 示例调用：
if __name__ == "__main__":
    rename_images_in_folder("/Users/ruiyuhan/Downloads/Pallas", prefix="Pallas_cats")
