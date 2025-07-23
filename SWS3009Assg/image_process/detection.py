from ultralytics import YOLO
import cv2, numpy as np, uuid
from pathlib import Path
from typing import Optional

_yolo_model: Optional[YOLO] = None 

def _load_yolo(weights: str) -> YOLO:
    global _yolo_model
    if _yolo_model is None:
        _yolo_model = YOLO(weights)
    return _yolo_model

def _pad_resize(crop, size=300, color=(255, 255, 255)):
    h, w = crop.shape[:2]
    s = size / max(w, h)
    nw, nh = int(w * s), int(h * s)
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), color, np.uint8)
    x0, y0 = (size - nw) // 2, (size - nh) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def extract_bbox(img_path: str,
                 yolo_weights: str,
                 out_size: int = 300,
                 save_dir: Optional[str] = None): 
    img = cv2.imread(img_path)
    if img is None:
        return None, None
    res = _load_yolo(yolo_weights).predict(img, conf=0.25, save=False)[0]
    if len(res.boxes) == 0:
        return None, None
    
    xyxy = res.boxes.xyxy[res.boxes.conf.argmax()].cpu().numpy().astype(int)
    x1, y1, x2, y2 = xyxy
    padded = _pad_resize(img[y1:y2, x1:x2], out_size)
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out_path = str(Path(save_dir) /
                       f'{Path(img_path).stem}_{uuid.uuid4().hex[:6]}.jpg')
        cv2.imwrite(out_path, padded)
        return padded, out_path
    return padded, None