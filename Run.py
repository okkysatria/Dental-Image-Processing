import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Listbox, SINGLE, Button, END, Label
from PIL import Image, ImageTk

def non_maximum_suppression(gradient_magnitude, gradient_direction):
    M, N = gradient_magnitude.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = gradient_direction * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            q = 255
            r = 255
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = gradient_magnitude[i, j + 1]
                r = gradient_magnitude[i, j - 1]
            elif 22.5 <= angle[i, j] < 67.5:
                q = gradient_magnitude[i + 1, j - 1]
                r = gradient_magnitude[i - 1, j + 1]
            elif 67.5 <= angle[i, j] < 112.5:
                q = gradient_magnitude[i + 1, j]
                r = gradient_magnitude[i - 1, j]
            elif 112.5 <= angle[i, j] < 157.5:
                q = gradient_magnitude[i - 1, j - 1]
                r = gradient_magnitude[i + 1, j + 1]

            if gradient_magnitude[i, j] >= q and gradient_magnitude[i, j] >= r:
                Z[i, j] = gradient_magnitude[i, j]
            else:
                Z[i, j] = 0

    return Z

def hysteresis_thresholding(img, low_threshold, high_threshold):
    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)
    
    strong_i, strong_j = np.where(img >= high_threshold)
    weak_i, weak_j = np.where((img <= high_threshold) & (img >= low_threshold))
    
    res[strong_i, strong_j] = 255
    res[weak_i, weak_j] = 75
    
    return res

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded correctly
    if image is None:
        print("Error loading image")
        return

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Threshold the image to isolate potential decay regions (dark areas)
    _, decay_thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

    # Threshold the image to segment the teeth (bright areas)
    _, teeth_thresholded = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

    # Use morphological operations to remove small noise and close gaps in the teeth area
    kernel = np.ones((5, 5), np.uint8)
    teeth_segmented = cv2.morphologyEx(teeth_thresholded, cv2.MORPH_CLOSE, kernel)
    teeth_segmented = cv2.morphologyEx(teeth_segmented, cv2.MORPH_OPEN, kernel)

    # Invert the teeth segmented image to create a mask for the teeth area
    teeth_mask = cv2.bitwise_not(teeth_segmented)

    # Combine the decay regions with the teeth mask
    combined_mask = cv2.bitwise_and(decay_thresholded, teeth_mask)

    # Apply the Sobel operator
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Compute the magnitude of the gradient
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    direction = np.arctan2(sobel_y, sobel_x)

    # Apply Non-Maximum Suppression
    nms_result = non_maximum_suppression(magnitude, direction)

    # Apply Hysteresis Thresholding
    hysteresis_result = hysteresis_thresholding(nms_result, low_threshold=50, high_threshold=100)

    # Normalize the magnitude image to the range [0, 255]
    magnitude = cv2.normalize(hysteresis_result, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to uint8 (8-bit)
    magnitude = np.uint8(magnitude)

    # Mask the Sobel result with the combined mask to focus on potential decay regions within teeth
    masked_edges = cv2.bitwise_and(magnitude, magnitude, mask=combined_mask)

    # Display the original, thresholded, and edge-detected images
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 4, 2)
    plt.title("Thresholded Decay Regions")
    plt.imshow(decay_thresholded, cmap='gray')

    plt.subplot(1, 4, 3)
    plt.title("Teeth Segmentation")
    plt.imshow(teeth_segmented, cmap='gray')

    plt.subplot(1, 4, 4)
    plt.title("Edges on Decay Regions within Teeth")
    plt.imshow(masked_edges, cmap='gray')

    plt.tight_layout()
    plt.savefig('method_sobel.jpg')
    plt.show()

def select_image_and_process():
    folder_path = "foto gigi"
    if not os.path.exists(folder_path):
        print("Folder 'foto gigi' tidak ditemukan.")
        return

    images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    if not images:
        print("Tidak ada gambar dalam folder 'foto gigi'.")
        return

    def on_select(evt):
        w = evt.widget
        index = int(w.curselection()[0])
        value = w.get(index)
        image_path = os.path.join(folder_path, value)
        image = Image.open(image_path)
        image.thumbnail((200, 200))
        img_preview = ImageTk.PhotoImage(image)
        preview_label.config(image=img_preview)
        preview_label.image = img_preview

    def on_process():
        selection = listbox.curselection()
        if selection:
            index = selection[0]
            image_path = os.path.join(folder_path, listbox.get(index))
            process_image(image_path)
            root.destroy()

    root = Tk()
    root.title("Pilih Gambar untuk Diproses")

    listbox = Listbox(root, selectmode=SINGLE)
    listbox.pack(fill='both', expand=True)

    for img in images:
        listbox.insert(END, img)

    listbox.bind('<<ListboxSelect>>', on_select)

    preview_label = Label(root)
    preview_label.pack()

    process_button = Button(root, text="Proses Gambar", command=on_process)
    process_button.pack()

    root.mainloop()

if __name__ == "__main__":
    select_image_and_process()
