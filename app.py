"""A Tkinter GUI application to input an image and get the GAN prediction of the image."""
from tkinter import Tk, filedialog, Label, Toplevel, Frame, Button, messagebox
from PIL import Image, ImageTk
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

def open_file():
    """Open a file dialog to select an image."""
    global filename
    global img
    global img_frame
    global dft_img_frame

    if img_frame is not None:
        img_frame.destroy()

    if dft_img_frame is not None:
        dft_img_frame.destroy()

    img_frame = Frame(root, width=300, height=300, bg="lightgrey")
    img_frame.grid(row=1, column=0, rowspan=3, columnspan=3, padx=20, pady=20)

    filename = filedialog.askopenfilename(initialdir="~/cs310/cs310-project/detectors/wang2020/examples", title="Select Image", filetypes=(("Image files", "*.jpg, *.jpeg, *.png"), ("All files", "*.*")))
    if filename:
        # Create image label
        img = ImageTk.PhotoImage(Image.open(filename))
        # Create label to display image
        img_label = Label(img_frame, image=img, text="Image will go here", bg="lightgrey")
        img_label.place(relx=0.5, rely=0.5, anchor="center")
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
    print("Successfully ran GAN prediction!")


def dft_image():
    global filename
    global dft_img_frame
    global info_label

    if dft_img_frame is not None:
        dft_img_frame.destroy()

    dft_img_frame = Frame(root, width=300, height=300, bg="lightgrey")
    dft_img_frame.grid(row=1, column=3, rowspan=3, columnspan=3, padx=20, pady=20)

    if filename:
        # Load the image
        img = Image.open(filename)
        # Convert the image to grayscale
        img = img.convert("L")
        # Resize the image to 256x256
        img = img.resize((256, 256))
        # Create a numpy array from the image
        img_np = np.array(img)
        # Perform the DFT
        dft = np.fft.fft2(img_np)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(np.abs(dft_shift))

        # Create the DFT image
        dft_img = Image.fromarray(magnitude_spectrum)
        dft_img = ImageTk.PhotoImage(dft_img)

        # Create label to display DFT image
        dft_img_label = Label(dft_img_frame, image=dft_img, text="DFT Image will go here", bg="lightgrey")
        dft_img_label.image = dft_img
        dft_img_label.place(relx=0.5, rely=0.5, anchor="center")
        info_label.config(text="DFT image displayed.", wraplength=280)
        dft_button.config(state="disabled")
        print("Successfully visualised DFT!")
    else:
        info_label.config(text="No image loaded.", wraplength=280)


# Button container on top-left corner
button_container = Frame(root, relief="raised", borderwidth=1)
button_container.grid(row=0, column=0, padx=20, pady=20)

# Open button to open an image
open_button = Button(button_container, text="Open image", command=open_file)
open_button.grid(row=0, column=0)
# GAN check button initially disabled
check_button = Button(button_container, text="Check fakeness", command=check_image, state="disabled")
check_button.grid(row=0, column=1)
# DFT button initially disabled
dft_button = Button(button_container, text="Visualise DFT", command=dft_image, state="disabled")
dft_button.grid(row=0, column=2)

# Image frame border
img_frame = Frame(root, width=300, height=300, bg="lightgrey")
img_frame.grid(row=1, column=0, rowspan=3, columnspan=3, padx=20, pady=20)
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