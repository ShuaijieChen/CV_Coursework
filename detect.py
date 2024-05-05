# 导入所需的库和模块
import os  # 操作系统相关的功能，如文件路径和目录操作
import shutil  # 用于删除文件和目录
import time  # 用于获取当前时间和计算时间差
from pathlib import Path  # 用于操作文件路径

import cv2  # OpenCV库，用于图像和视频处理
import numpy as np
import torch  # PyTorch库，用于深度学习模型
from numpy import random  # 用于随机数生成

import gui
# 导入YOLOv5模型相关的模块和函数
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import (
    check_img_size, non_max_suppression, scale_coords,
    plot_one_box, set_logging, xyxy2xywh)
from utils.torch_utils import select_device, time_synchronized
from utils_ds.parser import get_config
from deep_sort import build_tracker


# 加载图像后进行修复画质
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

    # 初始化
    set_logging()  # 设置日志记录
    device = select_device('')  # 选择设备（CPU或GPU）
    if os.path.exists(out):  # 如果输出目录存在
        shutil.rmtree(out)  # 删除目录
    os.makedirs(out)  # 创建新的输出目录
    half = device.type != 'cpu'  # 如果设备不是CPU，则启用半精度

    # 初始化DeepSORT
    config_deepsort = "deep_sort.yaml"
    cfg = get_config()
    cfg.merge_from_file(config_deepsort)
    use_cuda = device.type != 'cpu' and torch.cuda.is_available()
    deepsort = build_tracker(cfg, use_cuda=use_cuda)

    # 加载模型
    model = attempt_load(weights, map_location=device)  # 加载FP32模型
    image_size = check_img_size(image_size, s=model.stride.max())  # 检查图像尺寸
    if half:
        model.half()  # 转换为FP16

    # 设置数据加载器
    vid_path, vid_writer = None, None
    save_img = True  # 启用保存图片
    dataset = LoadImages(source, img_size=image_size)  # 加载图像数据

    # 获取类别名称和颜色
    names = model.module.names if hasattr(model, 'module') else model.names

    # 使用字典保存每个 track_id 对应的颜色
    track_colors = {}

    # 运行推理
    t0 = time.time()  # 记录开始时间
    img = torch.zeros((1, 3, image_size, image_size), device=device)  # 初始化图像
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # 运行一次以初始化
    count = 0

    for path, img, im0s, vid_cap in dataset:  # 遍历数据集中的每一帧
        # 在进行检测之前对图像进行修复画质
        if quality_repair_enabled:
            im0s = improve_image_quality(im0s)
        total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取视频的总帧数
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # 转换数据类型
        img /= 255.0  # 将像素值缩放到0-1范围
        if img.ndimension() == 3:
            img = img.unsqueeze(0)  # 增加一个维度

        # 推理
        t1 = time_synchronized()  # 记录推理开始时间
        pred = model(img, augment=False)[0]  # 进行推理

        # 应用非极大值抑制（NMS）
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=[0], agnostic=False)
        # 此处classes = [0]表示只检测人物
        t2 = time_synchronized()  # 记录NMS完成时间

        save_path = str(Path(out) / Path(path).name)  # 保存路径
        # 处理检测结果
        for i, det in enumerate(pred):  # 对每张图片的检测结果进行处理
            p, s, im0 = path, '', im0s
            s += '%gx%g ' % img.shape[2:]  # 输出图像大小

            # 如果有检测结果
            if det is not None and len(det):
                # 从img_size缩放到im0尺寸
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # 将检测结果转换为xywh格式
                bbox_xywh = xyxy2xywh(det[:, :4])
                confs = det[:, 4]

                # 使用DeepSORT更新跟踪
                outputs = deepsort.update(bbox_xywh, confs, im0)
                # outputs中每个元素包含(xyxy, track_id)

                # 遍历跟踪结果，绘制跟踪目标的边界框
                for output in outputs:
                    x1, y1, x2, y2, track_id = output

                    # 检查 track_id 是否已有颜色，如果没有则为其随机生成颜色
                    if track_id not in track_colors:
                        track_colors[track_id] = [random.randint(0, 255) for _ in range(3)]

                    color = track_colors[track_id]  # 获取 track_id 对应的颜色

                    label = f"{names[0], track_id}"  # 使用跟踪ID作为标签
                    plot_one_box([x1, y1, x2, y2], im0, label=label, color=color, line_thickness=2)

            # 打印时间（推理和NMS）
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # 保存结果（带检测的图像）
            if save_img:
                if vid_path != save_path:  # 新视频
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # 释放之前的视频写入器

                    fourcc = 'mp4v'  # 输出视频编解码器
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    # 查找保存路径中的文件是否存在
                    save_path_without_ext = os.path.splitext(save_path)[0]
                    extension = os.path.splitext(save_path)[1]

                    file_suffix = 1
                    while os.path.exists(save_path):
                        save_path = f"{save_path_without_ext}_{file_suffix}{extension}"
                        file_suffix += 1

                    # 创建新的视频写入器
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                vid_writer.write(im0)  # 写入视频帧

            # 调用 progress_callback 传递当前进度信息
            if progress_callback:
                progress = (count + 1) / total_frames * 100
                progress_callback(progress)
            count += 1

    # 如果启用保存文本或图片
    if save_img:
        print('Results saved to %s' % Path(out))

    print('Done. (%.3fs)' % (time.time() - t0))


# 主程序入口点
if __name__ == '__main__':
    gui.main()  # 运行GUI
