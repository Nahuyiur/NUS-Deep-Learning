from pathlib import Path
import os
import re
import cv2
from image_process.detection import extract_bbox
from classify.predict_breed import predict_cat

YOLO_WEIGHTS = 'SWS3009Assg/image_process/yolov8n-seg.pt'
MODEL_PATH   = 'SWS3009Assg/classify/mix_dataset_model.h5'
TMP_DIR      = Path('SWS3009Assg/tmp/judge_outputs')
CLASS_NAMES  = ['Pallas cats', 'Persian cats', 'Ragdolls', 'Singapura cats', 'Sphynx cats']


def extract_true_label_from_path(img_path):
    filename = os.path.basename(img_path)
    match = re.match(r'([A-Za-z]+)', filename)
    if not match:
        return None
    short_name = match.group(1).lower()
    name_map = {
        'pallas': 'Pallas cats',
        'persian': 'Persian cats',
        'ragdoll': 'Ragdolls',
        'singapura': 'Singapura cats',
        'sphynx': 'Sphynx cats'
    }
    for key in name_map:
        if key in short_name:
            return name_map[key]
    return None

# 主评估流程
def judge_pipeline(img_dir):
    total = 0
    correct = 0
    errors = []

    for img_path in Path(img_dir).glob('*'):
        if not img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            continue

        total += 1
        padded_img, save_path = extract_bbox(str(img_path), YOLO_WEIGHTS, out_size=300, save_dir=TMP_DIR)

        if save_path is None:
            true_label = extract_true_label_from_path(str(img_path))
            errors.append({
                'file': img_path.name,
                'true': true_label,
                'pred': 'None (no detection)',
                'conf': 0.0
            })
            print(f'❌ {img_path.name} | No cat detected')
            continue

        pred_label, conf = predict_cat(MODEL_PATH, str(save_path), CLASS_NAMES)
        true_label = extract_true_label_from_path(str(img_path))

        if pred_label == true_label:
            correct += 1
            print(f'✅ {img_path.name} | Correct ({pred_label}, {conf:.2f})')
        else:
            errors.append({
                'file': img_path.name,
                'true': true_label,
                'pred': pred_label,
                'conf': conf
            })
            print(f'❌ {img_path.name} | Wrong | True: {true_label} | Pred: {pred_label} ({conf:.2f})')

    # 输出结果统计
    print('\n🎯 Result Summary')
    print(f'Total: {total}')
    print(f'Correct: {correct}')
    print(f'Accuracy: {(correct / total * 100):.2f}%')
    if errors:
        print('\n⚠️  Misclassified Samples:')
        for e in errors:
            print(f" - {e['file']}: True={e['true']} | Pred={e['pred']} ({e['conf']:.2f})")

if __name__ == '__main__':
    judge_pipeline('SWS3009Assg/real_samples')
