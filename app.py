"""A Tkinter GUI application to input an image and get the GAN prediction of the image."""
from idlelib.tooltip import Hovertip
from tkinter import Tk, filedialog, Label, Toplevel, Frame, Button, Listbox, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image, ImageTk, ImageFilter
from src.image import dct, fft, spectral_density
import config
import matplotlib
import matplotlib.pyplot as plt
import os
import time
import subprocess
import threading
import numpy as np
import paramiko
import scipy.stats as stats
from PIL import Image
from detectors.wang2020.networks.resnet import resnet50
matplotlib.use('agg')

# Custom toolbar class to adjust the padding and size of the toolbar buttons
class CustomToolbar(NavigationToolbar2Tk):
    def __init__(self, canvas, parent, **kwargs):
        super().__init__(canvas, parent, **kwargs)
        # Adjust the padding and size of the toolbar buttons
        for button in self.winfo_children():
            button.pack_configure(padx=1, pady=1, ipadx=1, ipady=1)

    def set_message(self, s):
        pass  # Override to do nothing, effectively removing the coordinates display


root = Tk()
root.title("Image Deepfake Checker")
root.geometry("1000x800")

dirname = None
single_image = None
fake_imgs = []
real_imgs = []
tab_index = 0
dft_index = -1
res_val = None

def open_file():
    """Open a file dialog to select an image."""
    global dirname
    global fake_imgs
    global real_imgs
    global single_image
    global dft_index
    global fake_img_list
    global real_img_list
    global img_frame
    global dft_img_frame
    global dft_plot_frame
    global dft_power_frame
    global fingerprint_frame
    global dft_tab_control
    global res_combo
    global res_val
    global info_label
    global list_tab_control

    dirname = filedialog.askopenfilename(initialdir="/dcs/large/u2201755/FFHQ", title="Select an image")
    if dirname:
        # Clear the image lists
        if fake_img_list is not None:
            fake_img_list.delete(0, "end")
        if real_img_list is not None:
            real_img_list.delete(0, "end")
        fake_imgs = []
        real_imgs = []
        dft_index = -1

        # Clear the image frame
        if img_frame is not None:
            img_frame.destroy()

        # Clear the DFT image and plot frames
        if dft_img_frame is not None:
            dft_img_frame.destroy()
        if dft_plot_frame is not None:
            dft_plot_frame.destroy()
        if dft_power_frame is not None:
            dft_power_frame.destroy()
        if fingerprint_frame is not None:
            fingerprint_frame.destroy()

        # Hide the image lists
        real_img_list.grid_forget()
        fake_img_list.grid_forget()
        list_tab_control.grid_forget()

        # Create a new image frame
        img_frame = Frame(selection_container, width=350, height=350, bg="lightgrey")
        img_frame.grid(row=1, column=0, rowspan=4, columnspan=4, padx=2, pady=2)
        
        # Create new DFT image and plot frames
        dft_img_frame = Frame(dft_tab_control, width=350, height=350, bg="lightgrey")
        dft_img_frame.grid(row=1, column=3, rowspan=3, columnspan=3, padx=20, pady=20)
        dft_plot_frame = Frame(dft_tab_control, width=350, height=350, bg="lightgrey")
        dft_plot_frame.grid(row=1, column=3, rowspan=3, columnspan=3, padx=20, pady=20)
        dft_power_frame = Frame(dft_tab_control, width=350, height=350, bg="lightgrey")
        dft_power_frame.grid(row=1, column=3, rowspan=3, columnspan=3, padx=20, pady=20)
        fingerprint_frame = Frame(dft_tab_control, width=350, height=350, bg="lightgrey")
        fingerprint_frame.grid(row=1, column=3, rowspan=3, columnspan=3, padx=20, pady=20)

        dft_tab_control.add(dft_img_frame, text="2D DFT")
        dft_tab_control.add(dft_plot_frame, text="3D Plot")
        dft_tab_control.add(dft_power_frame, text="Power Spectrum")
        dft_tab_control.add(fingerprint_frame, text="Fingerprint")

        # Open image
        img = Image.open(dirname)
        single_image = img
        
        # Add image to list of images with image file name
        img_tk = ImageTk.PhotoImage(img)

        # Create a label to display the image
        img_label = Label(img_frame, image=img_tk, bg="lightgrey")
        img_label.image = img_tk
        img_label.place(relx=0.5, rely=0.5, anchor="center")
        
        # Enable the buttons
        check_button.config(state="normal")
        update_status("Run 'Check accuracy' to get the GAN prediction of realness. This may take a few seconds.")
        dft_button.config(state="normal")

        # Enable the residuals combobox
        res_combo.config(state="readonly", values=[1], textvariable=1)
        res_combo.current(0)
        res_val = 1

        # Update the info label
        info_label.config(text="Run 'Analyse' to see the 2D DFT, power spectrum and fingerprint of the image(s).", wraplength=280)


def open_folder():
    """Open a file dialog to select an image folder."""
    global dirname
    global single_image
    global fake_imgs
    global real_imgs
    global dft_index
    global fake_img_list
    global real_img_list
    global list_tab_control
    global img_frame
    global dft_img_frame
    global dft_plot_frame
    global dft_power_frame
    global fingerprint_frame
    global dft_tab_control
    global res_combo
    global res_val
    global info_label
    global dft_tab_control

    dirname = filedialog.askdirectory(initialdir="/dcs/large/u2201755/FFHQ")
    if dirname:
        # Clear the image lists
        single_image = None
        if real_img_list is not None:
            real_img_list.delete(0, "end")
        if fake_img_list is not None:
            fake_img_list.delete(0, "end")
        real_imgs = []
        fake_imgs = []
        dft_index = -1
        real_path = "0_real"
        fake_path = "1_fake"

        # Clear the image frame
        if img_frame is not None:
            img_frame.destroy()

        # Clear DFT image and plot frames
        if dft_img_frame is not None:
            dft_img_frame.destroy()
        if dft_plot_frame is not None:
            dft_plot_frame.destroy()
        if dft_power_frame is not None:
            dft_power_frame.destroy()
        if fingerprint_frame is not None:
            fingerprint_frame.destroy()

        # Create fake and real image lists
        fake_img_list.grid(row=1, column=1, rowspan=3, columnspan=1, padx=2, pady=2)
        fake_img_list.bind("<<ListboxSelect>>", show_selected_fake_image)

        real_img_list.grid(row=1, column=0, rowspan=3, columnspan=1, padx=2, pady=2)
        real_img_list.bind("<<ListboxSelect>>", show_selected_real_image)

        # Add the lists to the tab control
        list_tab_control.grid(row=1, column=0, rowspan=1, columnspan=3, padx=10, pady=10)

        list_tab_control.add(real_img_list, text="Real")
        list_tab_control.add(fake_img_list, text="Fake")

        # Create a new image frame
        img_frame = Frame(selection_container, width=150, height=150, bg="lightgrey")
        img_frame.grid(row=1, column=3, rowspan=2, columnspan=1, padx=10, pady=10)

        dft_img_frame = Frame(dft_tab_control, width=350, height=350, bg="lightgrey")
        dft_img_frame.grid(row=1, column=3, rowspan=3, columnspan=3, padx=20, pady=20)
        dft_plot_frame = Frame(dft_tab_control, width=350, height=350, bg="lightgrey")
        dft_plot_frame.grid(row=1, column=3, rowspan=3, columnspan=3, padx=20, pady=20)
        dft_power_frame = Frame(dft_tab_control, width=350, height=350, bg="lightgrey")
        dft_power_frame.grid(row=1, column=3, rowspan=3, columnspan=3, padx=20, pady=20)
        fingerprint_frame = Frame(dft_tab_control, width=350, height=350, bg="lightgrey")
        fingerprint_frame.grid(row=1, column=3, rowspan=3, columnspan=3, padx=20, pady=20)

        dft_tab_control.add(dft_img_frame, text="2D DFT")
        dft_tab_control.add(dft_plot_frame, text="3D Plot")
        dft_tab_control.add(dft_power_frame, text="Power Spectrum")
        dft_tab_control.add(fingerprint_frame, text="Fingerprint")

        # Check if the folder contains the fake and real directories
        fake_fp = os.path.join(dirname, fake_path)
        real_fp = os.path.join(dirname, real_path)
        if os.path.isdir(fake_fp) and os.path.isdir(real_fp):
            # For each image in the real directory
            for file in sorted(os.listdir(real_fp), key=lambda x: str(x.split(".")[0])):
                # If the file is not an image, skip it
                if not file.endswith((".jpg", ".jpeg", ".png")):
                    continue
                # Create image label
                img = Image.open(os.path.join(real_fp, file))
                real_imgs.append(img)
                # Add image to list of images
                real_img_list.insert("end", file)
            # For each image in the fake directory
            for file in sorted(os.listdir(fake_fp), key=lambda x: str(x.split(".")[0])):
                # If the file is not an image, skip it
                if not file.endswith((".jpg", ".jpeg", ".png")):
                    continue
                # Create image label
                img = Image.open(os.path.join(fake_fp, file))
                fake_imgs.append(img)
                # Add image to list of images
                fake_img_list.insert("end", file)
            # If the fake directory is empty, return an error
            if len(fake_imgs) == 0:
                messagebox.showerror("Error", "\"1_fake\" folder cannot be empty.")
                return
        else:
            messagebox.showerror("Error", "Missing 1_fake or 0_real folder.")
            return

        # Enable the buttons
        check_button.config(state="normal")
        update_status("Run 'Check accuracy' to get GAN prediction of fakeness. This may take a few seconds.")
        dft_button.config(state="normal")

        # Enable the residuals combobox
        if len(real_imgs) > 0:
            residuals = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
            residuals = [x for x in residuals if x <= len(real_imgs)]
            res_combo.config(state="readonly", values=residuals, textvariable=1)
            res_combo.current(0)
            res_val = 1
        assert len(fake_imgs) > 0, "Fake images list is empty."
        info_label.config(text="Run 'Analyse' to see the 2D DFT, power spectrum and fingerprint of the image(s).", wraplength=280)

def update_status(status):
    """Update the status label."""
    global status_label
    status_label.config(text=status, wraplength=280)
    root.update()


def ssh_and_run():
    """SSH into the server and run the GAN prediction."""
    global dirname
    try:
        # Delete old stdout files
        if os.path.exists(f"{config.cur_dir}/output_log.out"):
            os.remove(f"{config.cur_dir}/output_log.out")
        if os.path.exists(f"{config.cur_dir}/error_log.err"):
            os.remove(f"{config.cur_dir}/error_log.err")

        # SSH into the server
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect('kudu', username=config.username, password=config.password)
        
        # Run the GAN prediction
        if single_image is not None:
            client.exec_command(f"cd cs310/cs310-project && sbatch --export=FILE_PATH={dirname} demo.sbatch")
        else:
            client.exec_command(f"cd cs310/cs310-project && sbatch --export=DIR_PATH={dirname} demo_dir.sbatch")

        # Wait for output_log.out to contain the output, timeout after 60 seconds
        update_status("Computing accuracy...")
        timeout = 60
        timeout_time = time.time() + timeout
        result = ""

        while True:
            result = subprocess.run(["cat", "/dcs/22/u2201755/cs310/cs310-project/output_log.out"], capture_output=True, text=True)
            if "Probability of being synthetic" in result.stdout or "AP:" in result.stdout:
                break
            if time.time() > timeout_time:
                # Timeout error
                client.close()
                messagebox.showerror("Error", f"Timed out waiting for GAN prediction output after {timeout} seconds.")
                # Reactivate buttons
                open_file_button.config(state="normal")
                open_folder_button.config(state="normal")
                check_button.config(state="normal")
                dft_button.config(state="normal")
                update_status("Run 'Check accuracy' to get GAN prediction of realness. This may take a few seconds.")
                return

        # Get the output of the job
        update_status(result.stdout)
        client.close()
        print("Successfully ran GAN accuracy check!")
        # Reactivate buttons
        open_file_button.config(state="normal")
        open_folder_button.config(state="normal")
        check_button.config(state="normal")
        dft_button.config(state="normal")
    except Exception as e:
        print(e)


def ssh_in_bg():
    """Run the SSH in the background."""
    open_file_button.config(state="disabled")
    open_folder_button.config(state="disabled")
    check_button.config(state="disabled")
    dft_button.config(state="disabled")
    ssh_thread = threading.Thread(target=ssh_and_run)
    ssh_thread.start()


def analyse_image():
    """Compute the DFT, power spectrum and fingerprints of the image."""
    global dft_img_frame
    global dft_plot_frame
    global dft_power_frame
    global fingerprint_frame
    global dft_tab_control
    global tab_index
    global dft_index
    global res_val
    global info_label
    global dft_button

    # Clear the previous DFT image and plot frame
    if dft_img_frame is not None:
        dft_img_frame.destroy()
    if dft_plot_frame is not None:
        dft_plot_frame.destroy()
    # Clear the previous power spectrum frame
    if dft_power_frame is not None:
        dft_power_frame.destroy()
    # Clear the previous fingerprint frame
    if fingerprint_frame is not None:
        fingerprint_frame.destroy()

    dft_index = tab_index
    avg = None
    img_list = None
    if single_image is not None:
        img = np.array(single_image)
        avg = img.astype(np.uint8)
    else:
        if dft_index == 0:
            img_list = real_imgs
        else:
            img_list = fake_imgs
        # Get average of all images
        avg = np.mean(img_list, axis=0).astype(np.uint8)

    # Create an image from the average
    avg = Image.fromarray(avg)
    # Convert the image to grayscale
    avg_gray = avg.convert("L")
    # Resize the image to 256x256
    avg_gray = avg_gray.resize((256, 256))

    # Convert the image to a NumPy array
    avg_array = np.array(avg)

    def zero_pad(avg_array):
        # Determine the desired size for zero padding
        desired_size = (512, 512)  # Example size, can be adjusted

        # Create a new array of zeros with the desired size
        padded_array = np.zeros(desired_size, dtype=np.float32)

        # Calculate the position to place the original image in the center
        x_offset = (desired_size[0] - avg_array.shape[0]) // 2
        y_offset = (desired_size[1] - avg_array.shape[1]) // 2

        # Place the original image in the center of the new array
        padded_array[x_offset:x_offset + avg_array.shape[0], y_offset:y_offset + avg_array.shape[1]] = avg_array
        return padded_array

    #avg = zero_pad(avg_array)

    def compute_dft(avg):
        """Compute the 2D discrete Fourier transform of the image."""
        # Get the 2D discrete Fourier transform of the image
        dft = np.fft.fft2(avg)
        dft = np.fft.fftshift(dft)
        dft = np.abs(dft)
        dft = np.log(dft)
        dft = dft * 20
        return dft

    dft_norm = compute_dft(avg_gray)
    dft_img = dft_norm.astype(np.uint8)
    dft_img = Image.fromarray(dft_img)
    # dft_img.save(f"dft_adm.png")
    dft_img = dft_img.resize((350, 350))
    dft_tk = ImageTk.PhotoImage(dft_img)

    # Create a new DFT image frame
    dft_img_frame = Frame(dft_tab_control, width=350, height=350, bg="lightgrey")
    dft_img_frame.grid(row=1, column=3, rowspan=3, columnspan=3, padx=20, pady=20)

    # Create a label to display the DFT image
    dft_img_label = Label(dft_img_frame, image=dft_tk, bg="lightgrey")
    dft_img_label.image = dft_tk
    dft_img_label.place(relx=0.5, rely=0.5, anchor="center")

    # Update the tab control
    dft_tab_control.add(dft_img_frame, text="2D DFT")

    # Create a 3D plot of the FFT magnitude
    x = np.arange(dft_norm.shape[1])
    y = np.arange(dft_norm.shape[0])
    x, y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, dft_norm, cmap='viridis')

    ax.set_xlabel('Frequency X')
    ax.set_ylabel('Frequency Y')
    ax.set_zlabel('Magnitude')

    # Create a new DFT plot frame
    dft_plot_frame = Frame(dft_tab_control, width=600, height=600, bg="lightgrey")
    dft_plot_frame.grid(row=1, column=3, rowspan=5, columnspan=5, padx=20, pady=20)

    def embed_plot(frame):
        # Embed 3D plot in frame
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=1)
        toolbar = CustomToolbar(canvas, dft_plot_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="top", fill="both")

    root.after(0, embed_plot(dft_plot_frame))

    # Update the tab control
    dft_tab_control.add(dft_plot_frame, text="3D Plot")

    # Compute the power spectrum
    avg_gray_array = np.array(avg_gray)
    n = avg_gray_array.shape[0]

    fft_img = np.fft.fftn(avg_gray)
    fft_amps = np.abs(fft_img)**2

    k_freq = np.fft.fftfreq(n) * n
    k_freq_2d = np.meshgrid(k_freq, k_freq)
    k_norm = np.sqrt(k_freq_2d[0]**2 + k_freq_2d[1]**2)

    k_norm = k_norm.flatten()
    fft_amps = fft_amps.flatten()

    k_bins = np.arange(0.5, n//2+1, 1.)
    k_vals = 0.5 * (k_bins[1:] + k_bins[:-1])
    A_bins, _, _ = stats.binned_statistic(k_norm, fft_amps, bins=k_bins, statistic='mean')

    # Create a power spectrum plot
    fig_power = plt.figure(figsize=(3.5, 3.5))
    ax_power = fig_power.add_subplot(111)
    ax_power.plot(k_vals, A_bins)
    ax_power.set_xlabel('Frequency')
    ax_power.set_ylabel('Power')
    # Log scale for both axes for better visualisation
    ax_power.set_yscale('log')
    ax_power.set_xscale('log')

    # Adjust layout to prevent labels from being cut off
    fig_power.tight_layout()

    # Create a new power spectrum frame
    dft_power_frame = Frame(dft_tab_control, width=350, height=350, bg="lightgrey")
    dft_power_frame.grid(row=1, column=3, rowspan=3, columnspan=3, padx=20, pady=20)

    # Plot the power spectrum as an image
    canvas = FigureCanvasTkAgg(fig_power, master=dft_power_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side="top", fill="both", expand=1)

    # Update the tab control
    dft_tab_control.add(dft_power_frame, text="Power Spectrum")

    def compute_fingerprint(img):
        """Compute the fingerprint of the image."""
        # Apply median filter to the image
        img = Image.fromarray(img)
        img_filtered = img.filter(ImageFilter.MedianFilter(size=3))
        img_filtered = np.array(img_filtered)
        # Subtract the filtered image from the original image
        diff = img - img_filtered
        # Compute the fingerprint
        fingerprint = np.abs(diff)
        return fingerprint

    def avg_fingerprint(img_list, n) -> Image:
        """Select n images from the list and average them."""
        # Select n random images from the list
        len_list = len(img_list)
        selected_imgs = np.random.randint(0, len_list, n)
        # Compute the average of the selected images
        residuals = []
        for i in selected_imgs:
            img = img_list[i]
            # Convert to NumPy array
            img = np.array(img)
            residuals.append(compute_fingerprint(img))

        # Stack the images and compute the mean
        residuals_stack = np.stack(residuals, axis=0)
        avg_residual = np.mean(residuals_stack, axis=0)

        # Convert the average to an image
        avg_residual = avg_residual.astype(np.uint8)
        avg_residual = Image.fromarray(avg_residual, mode="RGB")
        # Save image
        # avg_residual.save(f"fingerprint_{n}_adm.png")
        return avg_residual

    if single_image is not None:
        fingerprint = compute_fingerprint(avg_array)
        fingerprint = Image.fromarray(fingerprint)
    else:
        fingerprint = avg_fingerprint(img_list, res_val)

    fingerprint = fingerprint.resize((350, 350))
    fingerprint_tk = ImageTk.PhotoImage(fingerprint)

    # Create a new fingerprint frame
    fingerprint_frame = Frame(dft_tab_control, width=350, height=350, bg="lightgrey")
    fingerprint_frame.grid(row=1, column=3, rowspan=3, columnspan=3, padx=20, pady=20)

    # Create a label to display the fingerprint
    fingerprint_label = Label(fingerprint_frame, image=fingerprint_tk, bg="lightgrey")
    fingerprint_label.image = fingerprint_tk
    fingerprint_label.place(relx=0.5, rely=0.5, anchor="center")

    # Update the tab control
    dft_tab_control.add(fingerprint_frame, text="Fingerprint")

    # Update the info label
    dft_button.config(state="normal")
    info_label.config(text="The 2D DFT, power spectrum and fingerprint of the image has been visualised.", wraplength=280)


def start_dft_in_bg():
    """Start the DFT function in the background."""
    dft_button.config(state="disabled")
    dft_thread = threading.Thread(target=analyse_image)
    dft_thread.start()


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
    img_frame.grid(row=1, column=3, rowspan=2, columnspan=1, padx=10, pady=10)

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
    img_frame.grid(row=1, column=3, rowspan=2, columnspan=1, padx=10, pady=10)

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


def on_tab_change(event):
    """Update the tab index when the tab is changed."""
    global tab_index
    global list_tab_control
    global res_combo

    # Set the tab index to the current tab
    tab_index = list_tab_control.index("current")
    residuals = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    # Enable the residuals combobox
    if tab_index == 0:
        residuals = [x for x in residuals if x <= len(real_imgs)]
    else:
        residuals = [x for x in residuals if x <= len(fake_imgs)]
    res_combo.config(state="readonly", values=residuals)
    """Disable the DFT button if the tab is the DFT tab
    if dft_index != -1:
        if tab_index == dft_index:
            dft_button.config(state="disabled")
        else:
            dft_button.config(state="normal")"""


def on_res_change(event):
    """Update the residuals when the combobox is changed."""
    global res_combo
    global res_val
    global info_label

    # Get the selected residuals value
    res_val = int(res_combo.get())


# Button container on top-left corner
button_container = Frame(root, relief="raised", borderwidth=1)
button_container.grid(row=0, column=0, padx=20, pady=20)

# Open button to open a single image
open_file_button = Button(button_container, text="Open image", command=open_file)
open_file_button.grid(row=0, column=0)
# Open button to open an image folder
open_folder_button = Button(button_container, text="Open folder", command=open_folder)
open_folder_button.grid(row=0, column=1)
open_tooltip = Hovertip(open_folder_button, "The folder must contain two subfolders: \"0_real\" containing real images and \"1_fake\" containing fake ones.\n\"1_fake\" cannot be empty.")
# GAN check button initially disabled
check_button = Button(button_container, text="Check accuracy", command=ssh_in_bg, state="disabled")
check_button.grid(row=0, column=2)
# DFT button initially disabled
dft_button = Button(button_container, text="Analyse", command=start_dft_in_bg, state="disabled")
dft_button.grid(row=0, column=3)

# Selection container on middle-left corner
selection_container = Frame(root, relief="raised", borderwidth=1)
selection_container.grid(row=1, column=0)

# List tab control
list_tab_control = ttk.Notebook(selection_container)
list_tab_control.grid(row=1, column=0, rowspan=1, columnspan=3, padx=10, pady=10)

# Real image list frame border
real_img_list = Listbox(selection_container, width=20, height=15, bg="lightgrey")
real_img_list.grid(row=1, column=0, rowspan=3, columnspan=1, padx=2, pady=2)

# Fake image list frame border
fake_img_list = Listbox(selection_container, width=20, height=15, bg="lightgrey")
fake_img_list.grid(row=1, column=0, rowspan=3, columnspan=1, padx=2, pady=2)

# Add the lists to the tab control
list_tab_control.add(real_img_list, text="Real")
list_tab_control.add(fake_img_list, text="Fake")
# Bind the <<NotebookTabChanged>> event to the on_tab_change function
list_tab_control.bind("<<NotebookTabChanged>>", on_tab_change)

# Image display frame border
img_frame = Frame(selection_container, width=150, height=150, bg="lightgrey")
img_frame.grid(row=1, column=3, rowspan=2, columnspan=1, padx=10, pady=10)

# Status border
status_frame = Frame(root, width=350, height=200, bg="lightgrey")
status_frame.grid(row=4, column=0, rowspan=2, columnspan=3, padx=20, pady=20)
status_label = Label(status_frame, text="No image loaded.", bg="lightgrey")
status_label.place(relx=0.5, rely=0.5, anchor="center")

# DFT tab control
dft_tab_control = ttk.Notebook(root)
dft_tab_control.grid(row=1, column=3, rowspan=3, columnspan=3, padx=20, pady=20)

# DFT image and 3D plot frames
dft_img_frame = Frame(dft_tab_control, width=350, height=350, bg="lightgrey")
dft_img_frame.grid(row=1, column=3, rowspan=3, columnspan=3, padx=20, pady=20)
dft_plot_frame = Frame(dft_tab_control, width=350, height=350, bg="lightgrey")
dft_plot_frame.grid(row=1, column=3, rowspan=3, columnspan=3, padx=20, pady=20)
dft_power_frame = Frame(dft_tab_control, width=350, height=350, bg="lightgrey")
dft_power_frame.grid(row=1, column=3, rowspan=3, columnspan=3, padx=20, pady=20)
fingerprint_frame = Frame(dft_tab_control, width=350, height=350, bg="lightgrey")
fingerprint_frame.grid(row=1, column=3, rowspan=3, columnspan=3, padx=20, pady=20)

# Add the DFT frames to the tab control
dft_tab_control.add(dft_img_frame, text="2D DFT")
dft_tab_control.add(dft_plot_frame, text="3D Plot")
dft_tab_control.add(dft_power_frame, text="Power Spectrum")
dft_tab_control.add(fingerprint_frame, text="Fingerprint")

# Residual combobox
res_combo = ttk.Combobox(root, state="disabled")
res_combo.grid(row=4, column=4, rowspan=1, columnspan=3, padx=1, pady=1)
# Bind the <<ComboboxSelected>> event to the on_res_change function
res_combo.bind("<<ComboboxSelected>>", on_res_change)

# Residual combobox label
res_label = Label(root, text="Residuals:", bg="lightgrey")
res_label.grid(row=4, column=3, rowspan=1, columnspan=1, padx=1, pady=1)

# DFT info border
info_frame = Frame(root, width=350, height=100, bg="lightgrey")
info_frame.grid(row=5, column=3, rowspan=1, columnspan=3, padx=20, pady=20)
info_label = Label(info_frame, text="No image loaded.", bg="lightgrey")
info_label.place(relx=0.5, rely=0.5, anchor="center")

root.mainloop()