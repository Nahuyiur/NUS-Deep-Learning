import os

def delete_non_image_files(folder_path):
    if not os.path.exists(folder_path):
        print(f"❌ 文件夹不存在：{folder_path}")
        return

    allowed_extensions = {'.jpg', '.jpeg', '.png'}  # 可保留的格式

    deleted_files = 0
    total_files = 0

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            total_files += 1
            ext = os.path.splitext(filename)[1].lower()
            if ext not in allowed_extensions:
                try:
                    os.remove(file_path)
                    deleted_files += 1
                    print(f"🗑️ 删除文件：{filename}")
                except Exception as e:
                    print(f"⚠️ 删除失败：{filename}，错误：{e}")

    print(f"\n✅ 处理完成：共处理 {total_files} 个文件，删除 {deleted_files} 个非图片文件。")

# 🧪 使用示例（替换为你的路径）
folder = "/Users/ruiyuhan/Desktop/NUS Deep Learning/SWS3009Assg/cat_datasets/sphynx"
delete_non_image_files(folder)
