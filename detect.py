import os  # Operating system related functions such as file path and directory operations
import time  # Used to obtain the current time and calculate the time difference
from pathlib import Path  # This command is used to operate file paths

import cv2  # OpenCV library for image and video processing
import numpy as np
import torch
from numpy import random

import gui
# Import modules and functions related to the YOLOv5 model
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import (
    check_img_size, non_max_suppression, scale_coords,
    plot_one_box, set_logging, xyxy2xywh)
from utils.torch_utils import select_device, time_synchronized
from utils_ds.parser import get_config
from deep_sort import build_tracker


# Repair the image quality after loading the image
def improve_image_quality(img):
    # 去噪
    img = cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10, templateWindowSize=7,
                                          searchWindowSize=21)
    # 直方图均衡
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(gray_image)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


# 定义主检测函数
def detect_human(source, output, conf_thres, iou_thres, quality_repair_enabled, progress_callback):
    out = output
    weights = 'yolov5s.pt'
    image_size = 640

    # initialize
    set_logging()  # Set logging
    device = select_device('')  # Select device (CPU or GPU)
    half = device.type != 'cpu'  # If the device is not a CPU, semi-precision is enabled

    # Initialize DeepSORT
    config_deepsort = "deep_sort.yaml"
    cfg = get_config()
    cfg.merge_from_file(config_deepsort)
    use_cuda = device.type != 'cpu' and torch.cuda.is_available()
    deepsort = build_tracker(cfg, use_cuda=use_cuda)

    # Load model
    model = attempt_load(weights, map_location=device)  # Load the FP32 model
    image_size = check_img_size(image_size, s=model.stride.max())  # Check image size
    if half:
        model.half()  # Convert to FP16

    # Set up the data loader
    vid_path, vid_writer = None, None
    save_img = True  # Enable save picture
    dataset = LoadImages(source, img_size=image_size)  # Loading image data

    # Gets the category name and color
    names = model.module.names if hasattr(model, 'module') else model.names

    # Use a dictionary to save the corresponding color for each track_id
    track_colors = {}

    # Run reasoning
    t0 = time.time()  # Record start time
    img = torch.zeros((1, 3, image_size, image_size), device=device)  # Initialize image
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # Run once to initialize
    count = 0

    for path, img, im0s, vid_cap in dataset:  # Iterate over every frame in the data set
        # The image quality is repaired before testing
        if quality_repair_enabled:
            im0s = improve_image_quality(im0s)
        total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Gets the total number of frames of the video
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # Convert data type
        img /= 255.0  # Scale the pixel values to the 0-1 range
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # Add a dimension

        # inference
        t1 = time_synchronized()  # Record the inference start time
        pred = model(img, augment=False)[0]  # inference

        # Applying Non-Maximum Suppression (NMS)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=[0], agnostic=False)
        # Where classes = [0] means that only people are tested
        t2 = time_synchronized()  # Record the NMS completion time

        save_path = str(Path(out) / Path(path).name)  # Save path
        # Processing test results
        for i, det in enumerate(pred):  # The results of each image were processed
            p, s, im0 = path, '', im0s
            s += '%gx%g ' % img.shape[2:]  # Output image size

            # If there are test results
            if det is not None and len(det):
                # Scale from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Convert test results to xywh format
                bbox_xywh = xyxy2xywh(det[:, :4])
                confs = det[:, 4]

                # Update the trace using DeepSORT
                outputs = deepsort.update(bbox_xywh, confs, im0)
                # Each element in outputs contains (xyxy, track_id)

                # Traverse the tracking result and draw the bounding box of the tracking target
                for output in outputs:
                    x1, y1, x2, y2, track_id = output

                    # Check if track_id already has a color, and if not, randomly generate a color for it
                    if track_id not in track_colors:
                        track_colors[track_id] = [random.randint(0, 255) for _ in range(3)]

                    color = track_colors[track_id]  # Gets the color corresponding to track_id

                    label = f"{names[0], track_id}"  # Use the tracking ID as the tag
                    plot_one_box([x1, y1, x2, y2], im0, label=label, color=color, line_thickness=2)

            # Print time (Inference and NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Save results (images with detection)
            if save_img:
                if vid_path != save_path:  # New video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # Release the previous video writer

                    fourcc = 'mp4v'  # Output video codec
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    # Check whether the file in the save path exists
                    save_path_without_ext = os.path.splitext(save_path)[0]
                    extension = os.path.splitext(save_path)[1]

                    file_suffix = 1
                    while os.path.exists(save_path):
                        save_path = f"{save_path_without_ext}_{file_suffix}{extension}"
                        file_suffix += 1

                    # Create a new video writer
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                vid_writer.write(im0)  # Write video frame

            # Call progress_callback to pass the current progress information
            if progress_callback:
                progress = (count + 1) / total_frames * 100
                progress_callback(progress)
            count += 1

    # If save picture is enabled
    if save_img:
        print('Results saved to %s' % Path(out))

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    gui.main()  # run GUI
