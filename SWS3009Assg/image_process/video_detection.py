import cv2
import os
import threading
import time
import uuid
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# 保存目录
SAVE_DIR = "SWS3009Assg/captured_cats"
os.makedirs(SAVE_DIR, exist_ok=True)

# 图像增强函数
def enhance_image(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    enhanced = cv2.merge((l, a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
    kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    return cv2.filter2D(enhanced, -1, kernel)

# 等比缩放并补白
def pad_resize(img, size=300, color=(255, 255, 255)):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), color, np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas

# 异步保存图像
def save_cropped_async(frame, bbox):
    x1, y1, x2, y2 = bbox
    crop = frame[y1:y2, x1:x2]
    enhanced = enhance_image(crop)
    padded = pad_resize(enhanced)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"cat_{timestamp}_{uuid.uuid4().hex[:6]}.jpg"
    path = os.path.join(SAVE_DIR, filename)
    cv2.imwrite(path, padded)
    print(f"✅ Saved: {path}")

# 实时检测 + 自动保存
def run_cat_detection_with_interval(conf_thres=0.25, interval=0.5, camera_id=0):
    model = YOLO("yolov8n.pt")
    class_names = model.model.names
    cat_class_id = 15

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print("✅ Started Real-Time Cat Detection... (ESC to exit)")
    last_save_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=conf_thres, verbose=False)[0]
        boxes = results.boxes

        best_box = None
        best_conf = 0

        for box in boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            if cls_id == cat_class_id and conf > best_conf:
                best_conf = conf
                best_box = box

        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            label = f"Cat {best_conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            current_time = time.time()
            if current_time - last_save_time >= interval:
                last_save_time = current_time
                threading.Thread(target=save_cropped_async, args=(frame.copy(), (x1, y1, x2, y2)), daemon=True).start()

        cv2.imshow("Real-Time Cat Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Detection stopped.")

# 启动主程序
if __name__ == "__main__":
    run_cat_detection_with_interval()
