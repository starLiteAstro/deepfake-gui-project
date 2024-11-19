"""A Tkinter GUI application to input an image and get the GAN prediction of the image."""
from tkinter import Tk, filedialog, Label, Toplevel, Frame, Button, Listbox
from PIL import Image, ImageTk
from src.image import dct, fft, spectral_density
from functools import partial
import cytoolz
import config
import os
import time
import sys
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
root.geometry("800x600")

filename = None
imgs = []

def open_file():
    """Open a file dialog to select an image."""
    global filename
    global imgs
    global img_list
    global img_frame
    global dft_img_frame

    filename = filedialog.askdirectory(initialdir="/dcs/large/u2201755/FFHQ")
    
    if filename:
        # Clear the image list
        img_list.delete(0, "end")
        imgs = []
        # Clear the image frame
        if img_frame is not None:
            img_frame.destroy()
        # Clear the DFT image frame
        if dft_img_frame is not None:
            dft_img_frame.destroy()
        
        img_list = Listbox(selection_container, width=20, height=15, bg="lightgrey")
        img_list.grid(row=1, column=0, rowspan=3, columnspan=1, padx=2, pady=2)
        img_list.bind("<<ListboxSelect>>", show_selected_image)
        # Create a new image frame
        img_frame = Frame(selection_container, width=150, height=150, bg="lightgrey")
        img_frame.grid(row=1, column=1, rowspan=2, columnspan=1, padx=2, pady=2)
        # Bind the Listbox selection event to the show_selected_image function

        dft_img_frame = Frame(root, width=300, height=300, bg="lightgrey")
        dft_img_frame.grid(row=1, column=3, rowspan=3, columnspan=3, padx=20, pady=20)

        # For each image in the directory
        for file in os.listdir(filename):
            # If the file is not an image, skip it
            if not file.endswith((".jpg", ".jpeg", ".png")):
                continue
            # Create image label
            img = Image.open(os.path.join(filename, file))
            imgs.append(img)
            # Add image to list of images
            img_list.insert("end", file)
        # Enable the buttons
        check_button.config(state="normal")
        update_status("Run 'Check fakeness' to get GAN prediction of fakeness. This may take a few seconds.")
        dft_button.config(state="normal")
        info_label.config(text="Run 'Visualise DFT' to see the 2D discrete Fourier transform of the image.", wraplength=280)

def update_status(status):
    """Update the status label."""
    global status_label
    status_label.config(text=status, wraplength=280)


def ssh_and_run():
    """SSH into the server and run the GAN prediction."""
    try:
        # SSH into the server
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect('kudu', username=config.username, password=config.password)
        
        # Run the GAN prediction
        update_status("Running GAN prediction...")
        stdin, stdout, stderr = client.exec_command(f"cd cs310/cs310-project && sbatch --export=FILE_PATH={filename} demo.sbatch")

        # Wait for output_log.out to contain the output, timeout after 20 seconds
        update_status("Waiting for GAN prediction output...")
        timeout = 20
        timeout_time = time.time() + timeout
        while True:
            stdin, stdout, stderr = client.exec_command(f"cat cs310/cs310-project/output_log.out")
            output = stdout.read().decode()
            if output:
                break
            if time.time() > timeout_time:
                # Timeout error root
                client.close()
                error = Toplevel(root)
                error.title("Error")
                error.geometry("200x100")
                error_label = Label(error, text=f"Timed out waiting for GAN prediction output after {timeout} seconds.")
                error_label.place(relx=0.5, rely=0.5, anchor="center")
                return

        # Get the output of the job
        update_status("Getting GAN prediction output...")
        stdin, stdout, stderr = client.exec_command(f"cat cs310/cs310-project/output_log.out")
        client.close()
    except Exception as e:
        print(e)

def check_image():
    """Update log file, SSH into the server and run the GAN prediction."""
    # Clear the log file
    with open("output_log.out", "w") as f:
        f.write("")
    # SSH into the server and run the GAN prediction
    ssh_and_run()
    # Output the GAN prediction from the log file
    with open("output_log.out", "r") as f:
        output = f.read()
    update_status(output)
    print(output)
    print("Successfully ran GAN prediction!")


def dft_image():
    global dft_img_frame
    global info_label
    global dft_button

    # Get average of all images
    avg = None
    for img in imgs:
        img = np.array(img)
        if avg is None:
            avg = img.astype(np.float32)
        else:
            avg += img.astype(np.float32)
    
    avg = avg / len(imgs)
    avg = avg.astype(np.uint8)
    avg = Image.fromarray(avg)
    # Convert the image to grayscale
    avg = avg.convert("L")
    # Resize the image to 300x300
    avg = avg.resize((300, 300))

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
    dft_button.config(state="disabled")

def show_selected_image(event):
    """Display the selected image in a frame."""
    global imgs
    global img_frame

    # Get the selected index
    selected_index = img_list.curselection()
    if not selected_index:
        return

    # Get the selected image
    selected_image = imgs[selected_index[0]]

    # Clear the previous image display frame
    if img_frame is not None:
        img_frame.destroy()

    # Create a new frame to display the selected image
    img_frame = Frame(selection_container, width=150, height=250, bg="lightgrey")
    img_frame.grid(row=1, column=1, rowspan=2, columnspan=1, padx=2, pady=2)

    # Resize the image to fit the frame
    selected_image = selected_image.resize((150, 150))
    img_tk = ImageTk.PhotoImage(selected_image)

    # Create a label to display the image
    img_label = Label(img_frame, image=img_tk, bg="lightgrey")
    img_label.image = img_tk
    img_label.place(relx=0.5, rely=0.5, anchor="center")

    # Create a label to display the image number
    img_number_label = Label(img_frame, text=f"Image {selected_index[0] + 1} of {len(imgs)}", bg="lightgrey")
    img_number_label.place(relx=0.5, rely=0.9, anchor="center")



# Button container on top-left corner
button_container = Frame(root, relief="raised", borderwidth=1)
button_container.grid(row=0, column=0, padx=20, pady=20)

# Open button to open an image
open_button = Button(button_container, text="Open folder", command=open_file)
open_button.grid(row=0, column=0)
# GAN check button initially disabled
check_button = Button(button_container, text="Check fakeness", command=check_image, state="disabled")
check_button.grid(row=0, column=1)
# DFT button initially disabled
dft_button = Button(button_container, text="Visualise DFT", command=dft_image, state="disabled")
dft_button.grid(row=0, column=2)

# Selection container on middle=left corner
selection_container = Frame(root, relief="raised", borderwidth=1)
selection_container.grid(row=1, column=0, padx=20, pady=20)

# Image list frame border
img_list = Listbox(selection_container, width=20, height=15, bg="lightgrey")
img_list.grid(row=1, column=0, rowspan=3, columnspan=1, padx=2, pady=2)
# Image display frame border
img_frame = Frame(selection_container, width=150, height=150, bg="lightgrey")
img_frame.grid(row=1, column=1, rowspan=2, columnspan=1, padx=2, pady=2)

# Status border
status_frame = Frame(root, width=300, height=100, bg="lightgrey")
status_frame.grid(row=4, column=0, rowspan=2, columnspan=3, padx=20, pady=20)
status_label = Label(status_frame, text="No image loaded.", bg="lightgrey")
status_label.place(relx=0.5, rely=0.5, anchor="center")

# DFT image frame border
dft_img_frame = Frame(root, width=300, height=300, bg="lightgrey")
dft_img_frame.grid(row=1, column=3, rowspan=3, columnspan=3, padx=20, pady=20)
# DFT info border
info_frame = Frame(root, width=300, height=100, bg="lightgrey")
info_frame.grid(row=4, column=3, rowspan=2, columnspan=3, padx=20, pady=20)
info_label = Label(info_frame, text="No image loaded.", bg="lightgrey")
info_label.place(relx=0.5, rely=0.5, anchor="center")
root.mainloop()