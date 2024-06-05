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
    # Memuat gambar
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Memeriksa apakah gambar berhasil dimuat
    if image is None:
        print(f"Error memuat gambar: {image_path}")
        return

    # Terapkan Gaussian Blur untuk mengurangi noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Threshold gambar untuk menyegmentasi gigi (area terang)
    _, teeth_thresholded = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

    # Gunakan operasi morfologi untuk menghilangkan noise kecil dan menutup celah di area gigi
    kernel = np.ones((5, 5), np.uint8)
    teeth_segmented = cv2.morphologyEx(teeth_thresholded, cv2.MORPH_CLOSE, kernel)
    teeth_segmented = cv2.morphologyEx(teeth_segmented, cv2.MORPH_OPEN, kernel)

    # Inversi gambar segmen gigi untuk membuat masker area gigi
    teeth_mask = cv2.bitwise_not(teeth_segmented)

    # Terapkan operator Sobel
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Hitung magnitudo gradien
    magnitude = cv2.magnitude(sobel_x, sobel_y)
    direction = np.arctan2(sobel_y, sobel_x)

    # Terapkan Non-Maximum Suppression
    nms_result = non_maximum_suppression(magnitude, direction)

    # Terapkan Hysteresis Thresholding
    hysteresis_result = hysteresis_thresholding(nms_result, low_threshold=50, high_threshold=100)

    # Normalisasi gambar magnitudo ke rentang [0, 255]
    magnitude = cv2.normalize(hysteresis_result, None, 0, 255, cv2.NORM_MINMAX)

    # Konversi ke uint8 (8-bit)
    magnitude = np.uint8(magnitude)

    # Masker hasil Sobel dengan masker gigi untuk fokus pada potensi area kerusakan dalam gigi
    masked_edges = cv2.bitwise_and(magnitude, magnitude, mask=teeth_mask)

    # Threshold gambar untuk mengisolasi potensi area kerusakan (area gelap)
    _, decay_thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    decay_masked = cv2.bitwise_and(decay_thresholded, decay_thresholded, mask=teeth_mask)

    # Temukan kontur dari area kerusakan
    contours, _ = cv2.findContours(decay_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Buat gambar output untuk menggambar kontur
    output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Gambar kontur dengan garis merah
    cv2.drawContours(output_image, contours, -1, (255, 0, 0), 2)

    # Tambahkan teks "Lubang" pada gambar output
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(output_image, 'Lubang', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Tampilkan gambar asli, thresholded, dan hasil deteksi tepi
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.title("Gambar Asli")
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 4, 2)
    plt.title("Segmentasi Gigi")
    plt.imshow(teeth_segmented, cmap='gray')

    plt.subplot(1, 4, 3)
    plt.title("Wilayah Kerusakan yang di-Threshold")
    plt.imshow(decay_thresholded, cmap='gray')

    plt.subplot(1, 4, 4)
    plt.title("Wilayah Kerusakan dengan Garis Merah")
    plt.imshow(output_image)

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
