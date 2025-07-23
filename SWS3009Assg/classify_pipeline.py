from pathlib import Path
import cv2
from image_process.detection import extract_bbox
from classify.predict_breed import predict_cat
import time
def classify_pipeline(raw_img):
    yolo_weight  = 'SWS3009Assg/image_process/yolov8n-seg.pt'
    classify_model     = 'SWS3009Assg/classify/mix_dataset_model.h5'
    classes = ['Pallas cats', 'Persian cats', 'Ragdolls','Singapura cats', 'Sphynx cats']
    tmp_dir = Path('SWS3009Assg/tmp/cat_pipeline_outputs')

    start_time = time.time()
    img_arr, save_path = extract_bbox(raw_img, yolo_weight, 300, tmp_dir)
    if save_path is None:
        save_path = tmp_dir / 'tmp.jpg'
        cv2.imwrite(str(save_path), img_arr)
    end_time=time.time()

    print("extract bbox time: ", end_time-start_time)

    start_time=time.time()
    label, conf = predict_cat(classify_model, str(save_path), classes)
    print(f'\nImage : {save_path}\nBreed : {label}  ({conf:.4f})')
    end_time=time.time()
    print("predict breed time: ", end_time-start_time)

    return label, conf
if __name__ == '__main__':
    classify_pipeline(raw_img = 'SWS3009Assg/img_trial/test3(2).jpeg')