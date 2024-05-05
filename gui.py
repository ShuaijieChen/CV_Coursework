import tkinter as tk
from tkinter import filedialog, ttk
import threading
import os

import detect  # 导入现有的检测代码
global quality_repair_enabled


def main():
    # 创建函数来运行检测
    def run_detection(source, output, conf_thres, iou_thres, quality_repair_enabled):
        detect.detect_human(
            source,
            output,
            conf_thres,
            iou_thres,
            quality_repair_enabled,  # 将复选框状态传递给检测函数
            progress_callback=update_progress
        )

    # 创建主窗口
    root = tk.Tk()
    root.title("YOLOv5 Detection")

    # 创建控件
    source_label = tk.Label(root, text="Input Video File:")
    source_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
    source_entry = tk.Entry(root, width=40)
    source_entry.grid(row=0, column=1, padx=10, pady=10)
    source_button = tk.Button(root, text="Browse", command=lambda: browse_file(source_entry))
    source_button.grid(row=0, column=2, padx=10, pady=10)

    # 创建控件
    output_label = tk.Label(root, text="Output Video File:")
    output_label.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
    output_entry = tk.Entry(root, width=40)
    output_entry.grid(row=1, column=1, padx=10, pady=10)
    output_button = tk.Button(root, text="Browse", command=lambda: select_folder(output_entry))
    output_button.grid(row=1, column=2, padx=10, pady=10)

    conf_label = tk.Label(root, text="Confidence Threshold:")
    conf_label.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
    conf_entry = tk.Entry(root, width=10)
    conf_entry.grid(row=2, column=1, padx=10, pady=10, sticky=tk.W)
    conf_entry.insert(0, "0.25")

    iou_label = tk.Label(root, text="IOU Threshold:")
    iou_label.grid(row=3, column=0, padx=10, pady=10, sticky=tk.W)
    iou_entry = tk.Entry(root, width=10)
    iou_entry.grid(row=3, column=1, padx=10, pady=10, sticky=tk.W)
    iou_entry.insert(0, "0.45")
    # 创建质量修复复选框
    quality_repair_var = tk.BooleanVar()
    quality_repair_checkbox = tk.Checkbutton(
        root,
        text="Enable Quality Repair (This results in a significant increase in uptime)",
        variable=quality_repair_var,
        command=lambda: toggle_quality_repair(quality_repair_var)
    )
    quality_repair_checkbox.grid(row=4, column=1, padx=10, pady=10)
    # 创建进度条
    progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
    progress.grid(row=5, column=1, padx=10, pady=10)
    progress_label = tk.Label(root, text="Progress:")
    progress_label.grid(row=5, column=0, padx=10, pady=10, sticky=tk.W)
    # 创建百分比进度标签
    percent_label = tk.Label(root, text="0%")
    percent_label.grid(row=5, column=2, padx=10, pady=10)

    # 播放视频并退出应用的函数
    def exit_and_play_video():
        video_path = output_entry.get()
        # 打开文件所在的文件夹
        # Windows上使用 "explorer /select,"
        # macOS和Linux上使用 "open -R"
        video_path = video_path.replace('/', '\\')
        print(video_path)
        if os.name == 'nt':  # Windows
            os.startfile(video_path)
        elif os.name == 'posix':  # macOS和Linux
            os.system(f'open -R "{video_path}"')

        # 关闭应用
        root.destroy()

    # 创建退出按钮（初始时隐藏）
    exit_button = tk.Button(root, text="Exit and Play Video", command=exit_and_play_video)
    exit_button.grid(row=6, column=0, columnspan=3, pady=20)

    exit_button.grid_remove()  # 隐藏退出按钮

    # 创建按钮
    def start_detection():
        source = source_entry.get()
        output = output_entry.get()
        conf_thres = float(conf_entry.get())
        iou_thres = float(iou_entry.get())

        # 获取复选框的状态
        quality_repair_enabled = quality_repair_var.get()

        # 在新线程中运行检测函数，以避免冻结UI
        detection_thread = threading.Thread(target=run_detection, args=(source, output, conf_thres, iou_thres, quality_repair_enabled))
        detection_thread.start()

        # 点击按钮之后移除按钮
        start_button.grid_remove()

    # 创建开始按钮
    start_button = tk.Button(root, text="Start Detection", command=start_detection)
    start_button.grid(row=6, column=0, columnspan=3, pady=20)

    # 浏览文件对话框函数
    def browse_file(entry):
        file_path = filedialog.askopenfilename()
        entry.delete(0, tk.END)
        entry.insert(0, file_path)

    def select_folder(entry):
        # 使用filedialog.askdirectory()函数打开文件夹选择对话框
        folder_path = filedialog.askdirectory()

        # 如果用户选择了一个文件夹，则在entry中显示文件夹路径
        if folder_path:
            entry.delete(0, tk.END)
            entry.insert(0, folder_path)

    def toggle_quality_repair(var):
        # 获取复选框的当前状态
        quality_repair_enabled = var.get()

    # 更新进度条的函数
    def update_progress(progress_value):
        # 更新进度条
        progress['value'] = progress_value
        # 更新百分比进度标签
        percent_text = f"{progress_value:.2f}%"
        percent_label.config(text=percent_text)
        # 如果进度达到了 100%，显示退出按钮
        if progress_value == 100:
            show_exit_button()
        # 刷新UI
        root.update_idletasks()

    # 显示退出按钮的函数
    def show_exit_button():
        exit_button.grid()

    # 运行主循环
    root.mainloop()
