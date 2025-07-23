import cv2, os, threading, time, uuid, numpy as np
from datetime import datetime
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import DepthwiseConv2D

SAVE_DIR = "SWS3009Assg/captured_cats"
YOLO_MODEL_PATH = "yolov8n.pt"
CLASSIFY_MODEL_PATH = "SWS3009Assg/classify/mix_dataset_model.h5"
CLASSES = ['Pallas cats','Persian cats','Ragdolls','Singapura cats','Sphynx cats']
os.makedirs(SAVE_DIR, exist_ok=True)

class DepthwiseConv2DCompat(DepthwiseConv2D):
    def __init__(self,*a,groups=1,**k): super().__init__(*a,**k)

def enhance(img, up=2):
    out = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    h, w = out.shape[:2]
    return cv2.resize(out,(w*up,h*up),interpolation=cv2.INTER_CUBIC)

def pad_resize(img, size=300):
    h,w = img.shape[:2]; s = size/max(h,w)
    nh,nw = int(h*s),int(w*s)
    canvas = np.full((size,size,3),255,np.uint8)
    resized = cv2.resize(img,(nw,nh),interpolation=cv2.INTER_AREA)
    y0,x0 = (size-nh)//2, (size-nw)//2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas

def classify(model,img):
    arr = preprocess_input(cv2.resize(img,(300,300)).astype("float32"))
    p = model.predict(np.expand_dims(arr,0),verbose=0)[0]
    idx = int(np.argmax(p)); return CLASSES[idx], float(p[idx])

def run(conf_th=0.25, interval=0.5, cam_id=0, cls_disp_th=0.70):
    detector = YOLO(YOLO_MODEL_PATH)
    classifier = tf.keras.models.load_model(CLASSIFY_MODEL_PATH,
        compile=False, custom_objects={'DepthwiseConv2D':DepthwiseConv2DCompat})

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print("❌ Camera open failed"); return
    print("✅ Camera started (ESC 退出)")

    last_save, last_pred = 0, {"label":"","conf":0.0}

    while True:
        ok, frame = cap.read(); 
        if not ok: break

        res = detector.predict(frame, conf=conf_th, verbose=False)[0]
        best_box, best_conf = None, 0.0
        for b in res.boxes:
            if int(b.cls)==15 and b.conf>best_conf:
                best_box, best_conf = b, float(b.conf)

        if best_box is not None:
            x1,y1,x2,y2 = map(int, best_box.xyxy[0])

            # draw bounding box and confidence of cats
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"Cat {best_conf:.2f}",(x1,max(0,y1-25)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

            # if we get high confidence, classify the cat
            if last_pred["conf"] >= cls_disp_th:
                txt = f"{last_pred['label']} ({last_pred['conf']:.2f})"
                cv2.putText(frame, txt, (x1, max(0, y1-5)),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

            # when the time interval is up, save the image and classify it
            if time.time()-last_save >= interval:
                last_save = time.time()
                def worker(img, box):
                    xi,yi,xa,ya = box
                    crop = img[yi:ya, xi:xa]
                    padded = pad_resize(enhance(crop))
                    fname = f"cat_{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid.uuid4().hex[:6]}.jpg"
                    cv2.imwrite(os.path.join(SAVE_DIR,fname), padded)
                    lbl, cf = classify(classifier, padded)
                    last_pred["label"], last_pred["conf"] = lbl, cf
                    print(f"✔ Saved {fname} | {lbl} ({cf:.2f})")
                threading.Thread(target=worker,args=(frame.copy(),(x1,y1,x2,y2)),daemon=True).start()

        cv2.imshow("Cat Detection & Breed Classification", frame)
        if cv2.waitKey(1)&0xFF==27: break

    cap.release(); cv2.destroyAllWindows(); print("👋 Done.")

if __name__=="__main__":
    run()
