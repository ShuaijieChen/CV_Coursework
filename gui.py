import tkinter as tk
from tkinter import filedialog, ttk
import threading
import os

import detect  # Import existing detection code
global quality_repair_enabled


def main():
    # Create a function to run the test
    def run_detection(source, output, conf_thres, iou_thres, quality_repair_enabled):
        detect.detect_human(
            source,
            output,
            conf_thres,
            iou_thres,
            quality_repair_enabled,  # Pass the checkbox status to the detection function
            progress_callback=update_progress
        )

    # Create main window
    root = tk.Tk()
    root.title("YOLOv5 Detection")

    # Create control
    source_label = tk.Label(root, text="Input Video File:")
    source_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
    source_entry = tk.Entry(root, width=40)
    source_entry.grid(row=0, column=1, padx=10, pady=10)
    source_button = tk.Button(root, text="Browse", command=lambda: browse_file(source_entry))
    source_button.grid(row=0, column=2, padx=10, pady=10)

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
    # Create a quality repair check box
    quality_repair_var = tk.BooleanVar()
    quality_repair_checkbox = tk.Checkbutton(
        root,
        text="Enable Quality Repair (This results in a significant increase in uptime)",
        variable=quality_repair_var,
        command=lambda: toggle_quality_repair(quality_repair_var)
    )
    quality_repair_checkbox.grid(row=4, column=1, padx=10, pady=10)
    # Creating a progress bar
    progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
    progress.grid(row=5, column=1, padx=10, pady=10)
    progress_label = tk.Label(root, text="Progress:")
    progress_label.grid(row=5, column=0, padx=10, pady=10, sticky=tk.W)
    # Create a percentage progress label
    percent_label = tk.Label(root, text="0%")
    percent_label.grid(row=5, column=2, padx=10, pady=10)

    # Create a percentage progress label
    def exit_and_play_video():
        video_path = output_entry.get()
        # Open the folder where the file is located
        video_path = video_path.replace('/', '\\')
        print(video_path)
        if os.name == 'nt':  # Windows
            os.startfile(video_path)
        elif os.name == 'posix':  # macOS和Linux
            os.system(f'open -R "{video_path}"')

        # Close the app
        root.destroy()

    # Create exit button (hidden initially)
    exit_button = tk.Button(root, text="Exit and Play Video", command=exit_and_play_video)
    exit_button.grid(row=6, column=0, columnspan=3, pady=20)

    exit_button.grid_remove()  # Hide exit button

    # Create button
    def start_detection():
        source = source_entry.get()
        output = output_entry.get()
        conf_thres = float(conf_entry.get())
        iou_thres = float(iou_entry.get())

        # Gets the status of the check box
        quality_repair_enabled = quality_repair_var.get()

        # Run the detection function in the new thread to avoid freezing the UI
        detection_thread = threading.Thread(target=run_detection, args=(source, output, conf_thres, iou_thres, quality_repair_enabled))
        detection_thread.start()

        # Remove the button after clicking it
        start_button.grid_remove()

    # Create Start button
    start_button = tk.Button(root, text="Start Detection", command=start_detection)
    start_button.grid(row=6, column=0, columnspan=3, pady=20)

    # Browse file dialog function
    def browse_file(entry):
        file_path = filedialog.askopenfilename()
        entry.delete(0, tk.END)
        entry.insert(0, file_path)

    def select_folder(entry):
        # Open the folder selection dialog using the filedialog.askdirectory() function
        folder_path = filedialog.askdirectory()

        # If the user selects a folder, the folder path is displayed in entry
        if folder_path:
            entry.delete(0, tk.END)
            entry.insert(0, folder_path)

    def toggle_quality_repair(var):
        # Gets the current status of the check box
        quality_repair_enabled = var.get()

    # Update the progress bar function
    def update_progress(progress_value):
        # Update progress bar
        progress['value'] = progress_value
        # Update the percentage progress TAB
        percent_text = f"{progress_value:.2f}%"
        percent_label.config(text=percent_text)
        # If the progress reaches 100%, an exit button is displayed
        if progress_value == 100:
            show_exit_button()
        # 刷新UI
        root.update_idletasks()

    # A function that displays the exit button
    def show_exit_button():
        exit_button.grid()

    # Run main loop
    root.mainloop()
