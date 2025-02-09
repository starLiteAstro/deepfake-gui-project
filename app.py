"""A Tkinter GUI application to input an image and get the GAN prediction of the image."""
from tkinter import Tk, filedialog, Label, Toplevel, Frame, Button, Listbox
from PIL import Image, ImageTk, ImageFilter
from src.image import dct, fft, spectral_density
from functools import partial
import cytoolz
import config
import os
import time
import sys
import threading
import torch
import torch.nn
import argparse
import numpy as np
import paramiko
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from detectors.wang2020.networks.resnet import resnet50

root = Tk()
root.title("GAN Image Checker")
root.geometry("1000x800")

dirname = None
fake_imgs = []
real_imgs = []

def open_file():
    """Open a file dialog to select an image."""
    global dirname
    global fake_imgs
    global real_imgs
    global fake_img_list
    global real_img_list
    global img_frame
    global dft_img_frame

    dirname = filedialog.askdirectory(initialdir="/dcs/large/u2201755/FFHQ")
    if dirname:
        # Clear the image lists
        fake_img_list.delete(0, "end")
        real_img_list.delete(0, "end")
        fake_imgs = []
        real_imgs = []
        fake_path = "1_fake"
        real_path = "0_real"
        # Clear the image frame
        if img_frame is not None:
            img_frame.destroy()
        # Clear the DFT image frame
        if dft_img_frame is not None:
            dft_img_frame.destroy()
        
        real_img_list = Listbox(selection_container, width=20, height=15, bg="lightgrey")
        real_img_list.grid(row=1, column=0, rowspan=3, columnspan=1, padx=2, pady=2)
        real_img_list.bind("<<ListboxSelect>>", show_selected_real_image)

        fake_img_list = Listbox(selection_container, width=20, height=15, bg="lightgrey")
        fake_img_list.grid(row=1, column=1, rowspan=3, columnspan=1, padx=2, pady=2)
        fake_img_list.bind("<<ListboxSelect>>", show_selected_fake_image)

        # Create a new image frame
        img_frame = Frame(selection_container, width=150, height=150, bg="lightgrey")
        img_frame.grid(row=1, column=2, rowspan=2, columnspan=1, padx=2, pady=2)

        dft_img_frame = Frame(root, width=300, height=300, bg="lightgrey")
        dft_img_frame.grid(row=1, column=3, rowspan=3, columnspan=3, padx=20, pady=20)
        
        fake_fp = os.path.join(dirname, fake_path)
        real_fp = os.path.join(dirname, real_path)
        if os.path.isdir(fake_fp) and os.path.isdir(real_fp):
            # For each image in the real directory
            for file in os.listdir(real_fp):
                # If the file is not an image, skip it
                if not file.endswith((".jpg", ".jpeg", ".png")):
                    continue
                # Create image label
                img = Image.open(os.path.join(real_fp, file))
                real_imgs.append(img)
                # Add image to list of images
                real_img_list.insert("end", file)
            # For each image in the fake directory
            for file in os.listdir(fake_fp):
                # If the file is not an image, skip it
                if not file.endswith((".jpg", ".jpeg", ".png")):
                    continue
                # Create image label
                img = Image.open(os.path.join(fake_fp, file))
                fake_imgs.append(img)
                # Add image to list of images
                fake_img_list.insert("end", file)
        else:
            error = Toplevel(root)
            error.title("Error")
            error.geometry("200x100")
            error_label = Label(error, text=f"Missing 1_fake or 0_real folder.")
            error_label.place(relx=0.5, rely=0.5, anchor="center")
            return
        # Enable the buttons
        check_button.config(state="normal")
        update_status("Run 'Check fakeness' to get GAN prediction of fakeness. This may take a few seconds.")
        dft_button.config(state="normal")
        info_label.config(text="Run 'Visualise DFT' to see the 2D discrete Fourier transform of the image.", wraplength=280)
    print(dirname)

def update_status(status):
    """Update the status label."""
    global status_label
    status_label.config(text=status, wraplength=280)


def ssh_and_run():
    global dirname
    """SSH into the server and run the GAN prediction."""
    print(dirname)
    try:
        # SSH into the server
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect('kudu', username=config.username, password=config.password)
        # Delete old stdout files
        stdin, stdout, stderr = client.exec_command(f"rm cs310/cs310-project/error_log.err && rm cs310/cs310-project/output_log.out")
        # Run the GAN prediction
        stdin, stdout, stderr = client.exec_command(f"cd cs310/cs310-project && sbatch --export=DIR_PATH={dirname} demo_dir.sbatch")

        # Wait for output_log.out to contain the output, timeout after 60 seconds
        update_status("Waiting for GAN prediction output...")
        timeout = 60
        timeout_time = time.time() + timeout
        while True:
            stdin, stdout, stderr = client.exec_command(f"wc -l < cs310/cs310-project/output_log.out")
            lines = stdout.read().decode()
            if lines == "5":
                break
            stdin, stdout, stderr = client.exec_command(f"tail -1 cs310/cs310-project/error_log.err")
            progress = stdout.read().decode()
            if progress:
                print(progress)
                update_status(progress)
            if time.time() > timeout_time:
                # Timeout error root
                client.close()
                error = Toplevel(root)
                error.title("Error")
                error.geometry("500x100")
                error_label = Label(error, text=f"Timed out waiting for GAN prediction output after {timeout} seconds.")
                error_label.place(relx=0.5, rely=0.5, anchor="center")
                # Reactivate buttons
                open_button.config(state="normal")
                check_button.config(state="normal")
                dft_button.config(state="normal")
                return

        # Get the output of the job
        stdin, stdout, stderr = client.exec_command(f"cat cs310/cs310-project/output_log.out")
        client.close()
        
        # Output the GAN prediction from the log file
        with open("output_log.out", "r") as f:
            output = f.read()
        update_status(output)
        print(output)
        print("Successfully ran GAN prediction!")
    except Exception as e:
        print(e)

def ssh_in_bg():
    """Run the SSH in the background."""
    open_button.config(state="disabled")
    check_button.config(state="disabled")
    dft_button.config(state="disabled")
    ssh_thread = threading.Thread(target=ssh_and_run)
    ssh_thread.start()

def dft_image():
    global dft_img_frame
    global info_label
    global dft_button

    # Get average of all images
    avg = None
    for img in fake_imgs:
        img = np.array(img)
        if avg is None:
            avg = img.astype(np.float32)
        else:
            avg += img.astype(np.float32)
    
    avg = avg / len(fake_imgs)
    avg = avg.astype(np.uint8)
    avg = Image.fromarray(avg)
    # Convert the image to grayscale
    avg = avg.convert("L")
    # Resize the image to 300x300
    avg = avg.resize((300, 300))
    avg = avg.filter(ImageFilter.GaussianBlur(0.5))

    # Get the 2D discrete Fourier transform of the image
    dft = np.fft.fft2(avg)
    dft = np.fft.fftshift(dft)
    dft = np.abs(dft)
    dft = np.log(dft)
    dft = dft * 20
    dft = dft.astype(np.uint8)
    dft = Image.fromarray(dft)
    dft = dft.resize((300, 300))
    dft_tk = ImageTk.PhotoImage(dft)

    # Clear the previous DFT image frame
    if dft_img_frame is not None:
        dft_img_frame.destroy()

    # Create a new DFT image frame
    dft_img_frame = Frame(root, width=300, height=300, bg="lightgrey")
    dft_img_frame.grid(row=1, column=3, rowspan=3, columnspan=3, padx=20, pady=20)

    # Create a label to display the DFT image
    dft_img_label = Label(dft_img_frame, image=dft_tk, bg="lightgrey")
    dft_img_label.image = dft_tk
    dft_img_label.place(relx=0.5, rely=0.5, anchor="center")

    # Update the info label
    info_label.config(text="The 2D discrete Fourier transform of the image has been visualised.", wraplength=280)


def start_dft_in_bg():
    """Start the DFT function in the background."""
    dft_button.config(state="disabled")
    dft_thread = threading.Thread(target=dft_image)
    dft_thread.start()


def show_selected_fake_image(event):
    """Display the selected fake image in a frame."""
    global fake_img_list
    global img_frame

    # Get the selected index
    selected_index = fake_img_list.curselection()
    if not selected_index:
        return

    # Get the selected image
    selected_image = fake_imgs[selected_index[0]]

    # Clear the previous image display frame
    if img_frame is not None:
        img_frame.destroy()

    # Create a new frame to display the selected image
    img_frame = Frame(selection_container, width=150, height=250, bg="lightgrey")
    img_frame.grid(row=1, column=2, rowspan=2, columnspan=1, padx=2, pady=2)

    # Resize the image to fit the frame
    selected_image = selected_image.resize((150, 150))
    img_tk = ImageTk.PhotoImage(selected_image)

    # Create a label to display the image
    img_label = Label(img_frame, image=img_tk, bg="lightgrey")
    img_label.image = img_tk
    img_label.place(relx=0.5, rely=0.5, anchor="center")

    # Create a label to display the image number
    img_number_label = Label(img_frame, text=f"Image {selected_index[0] + 1} of {len(fake_imgs)}", bg="lightgrey")
    img_number_label.place(relx=0.5, rely=0.9, anchor="center")

def show_selected_real_image(event):
    """Display the selected real image in a frame."""
    global real_img_list
    global img_frame

    # Get the selected index
    selected_index = real_img_list.curselection()
    if not selected_index:
        return

    # Get the selected image
    selected_image = real_imgs[selected_index[0]]

    # Clear the previous image display frame
    if img_frame is not None:
        img_frame.destroy()

    # Create a new frame to display the selected image
    img_frame = Frame(selection_container, width=150, height=250, bg="lightgrey")
    img_frame.grid(row=1, column=2, rowspan=2, columnspan=1, padx=2, pady=2)

    # Resize the image to fit the frame
    selected_image = selected_image.resize((150, 150))
    img_tk = ImageTk.PhotoImage(selected_image)

    # Create a label to display the image
    img_label = Label(img_frame, image=img_tk, bg="lightgrey")
    img_label.image = img_tk
    img_label.place(relx=0.5, rely=0.5, anchor="center")

    # Create a label to display the image number
    img_number_label = Label(img_frame, text=f"Image {selected_index[0] + 1} of {len(real_imgs)}", bg="lightgrey")
    img_number_label.place(relx=0.5, rely=0.9, anchor="center")

# Button container on top-left corner
button_container = Frame(root, relief="raised", borderwidth=1)
button_container.grid(row=0, column=0, padx=20, pady=20)

# Open button to open an image
open_button = Button(button_container, text="Open folder", command=open_file)
open_button.grid(row=0, column=0)
# GAN check button initially disabled
check_button = Button(button_container, text="Check fakeness", command=ssh_in_bg, state="disabled")
check_button.grid(row=0, column=1)
# DFT button initially disabled
dft_button = Button(button_container, text="Visualise DFT", command=start_dft_in_bg, state="disabled")
dft_button.grid(row=0, column=2)

# Selection container on middle-left corner
selection_container = Frame(root, relief="raised", borderwidth=1)
selection_container.grid(row=1, column=0, padx=20, pady=20)

# Real image list frame border
real_img_list = Listbox(selection_container, width=20, height=15, bg="lightgrey")
real_img_list.grid(row=1, column=0, rowspan=3, columnspan=1, padx=2, pady=2)
# Fake image list frame border
fake_img_list = Listbox(selection_container, width=20, height=15, bg="lightgrey")
fake_img_list.grid(row=1, column=1, rowspan=3, columnspan=1, padx=2, pady=2)
# Image display frame border
img_frame = Frame(selection_container, width=150, height=150, bg="lightgrey")
img_frame.grid(row=1, column=2, rowspan=2, columnspan=1, padx=2, pady=2)

# Status border
status_frame = Frame(root, width=300, height=200, bg="lightgrey")
status_frame.grid(row=4, column=0, rowspan=2, columnspan=3, padx=20, pady=20)
status_label = Label(status_frame, text="No image loaded.", bg="lightgrey")
status_label.place(relx=0.5, rely=0.5, anchor="center")

# DFT image frame border
dft_img_frame = Frame(root, width=300, height=300, bg="lightgrey")
dft_img_frame.grid(row=1, column=3, rowspan=3, columnspan=3, padx=20, pady=20)
# DFT info border
info_frame = Frame(root, width=300, height=200, bg="lightgrey")
info_frame.grid(row=4, column=3, rowspan=2, columnspan=3, padx=20, pady=20)
info_label = Label(info_frame, text="No image loaded.", bg="lightgrey")
info_label.place(relx=0.5, rely=0.5, anchor="center")
root.mainloop()