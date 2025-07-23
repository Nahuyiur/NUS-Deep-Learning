import os

# 设置你的数据集根路径
data_root = "mix_dataset"  # ← 修改为你本地的数据集路径

# 猫的分类列表（对应的文件夹名称必须一致）
categories = ['Pallas cats', 'Persian cats','Ragdolls','Singapura cats',  'Sphynx cats']

# 初始化计数
count_dict = {cat: 0 for cat in categories}

# 遍历 train 和 validation 文件夹
for subset in ['train', 'validation']:
    subset_path = os.path.join(data_root, subset)
    for cat in categories:
        cat_path = os.path.join(subset_path, cat)
        if os.path.isdir(cat_path):
            images = [f for f in os.listdir(cat_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            count_dict[cat] += len(images)

# 打印结果
print("Image Counts by Category:")
for cat in categories:
    print(f"{cat}: {count_dict[cat]} images")
