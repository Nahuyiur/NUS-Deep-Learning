import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier

# This is an example of a callback. The callback is called
# every time detect finishes detecting objects and drawing
# the bounding boxes.
# The result_list parameter is a list of results, one for
# each object detected.
# Index 0 xyxy: Bounding box in xyxy format (actual coordinates)
# Index 1 xywh: Bounding box in xywh normalized format
# Index 2 cls: Class, 
# Index 3 name: Object name, 
# Index 4 color: Object color,
# im0: The original image without padding and normalization

def demo_callback(results, im0, waitvalue=0):
    for result in results:
        xyxy = result[0]
        name = result[3]
        color = result[4]
        coords = [t.tolist() for t in xyxy]
        print("Object ", name, " Detected at ", coords, 
        " Centre at ", [(coords[0] + coords[2])/2, (coords[1] + coords[3])/2])

# waitvalue is needed only if your callback calls cv2.imshow. This
# parameter is to be passed to cv2.waitKey after the cv2.imshow.
# This parameter should be 0 for images and 15 for videos
def detect(filename, imgsz = 640, device="cpu", weights="yolov7.pt",
callback = None, waitvalue=0):

    webcam = filename.isnumeric()
    device = select_device(device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(filename, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(filename, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        # im0s is the original image,
        # img is the padded and resized image
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img)[0]

        # Inference
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.65)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        if callback is not None:
            result_list = []

        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(p.name)  # img.jpg
            txt_path = str(p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh

                    ##
                    ## Append results to result list
                    ##

                    if callback is not None:
                        result_list.append([xyxy, xywh, cls,
                        names[int(cls)], colors[int(cls)]])

        if callback is not None:
            callback(result_list, im0, waitvalue)

def main():
    detect("0", 
    weights="yolov7-tiny.pt", callback=demo_callback)

    #detect("inference/images/horses.jpg",
    #weights="yolov7-tiny.pt", callback=demo_callback)

if __name__ == "__main__":
    main()
