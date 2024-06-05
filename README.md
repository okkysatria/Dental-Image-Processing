# Dental-Image-Processing
Program ini bertujuan untuk mendeteksi tepi dan area kerusakan gigi dalam gambar menggunakan kombinasi metode pemrosesan gambar seperti Gaussian Blur, Sobel operator, dan teknik non-maximum suppression. Hasil akhir adalah gambar dengan tepi yang dideteksi, disimpan dalam file jpg

## Instruksi Penggunaan Program
  
### 1.Pastikan Semua File Gambar Tersedia:
Tempatkan semua file gambar gigi yang ingin diproses dalam folder bernama "foto gigi" di direktori yang sama dengan file script Python ini.

### 2.Menjalankan Program:

Pastikan Python dan semua dependensi yang diperlukan (seperti opencv-python, numpy, matplotlib, Pillow) sudah terinstal di lingkungan Python Anda.

**Install the required libraries**

```
pip install opencv-python numpy matplotlib Pillow
```

Buka terminal atau command prompt.
Navigasikan ke direktori tempat script Python ini berada.
Jalankan script dengan perintah berikut:

```
python run.py
```

### 3.Memilih Gambar:

![Screenshot_2](https://github.com/okkysatria/Dental-Image-Processing/assets/84434840/10051b27-34a9-42af-b104-4c68d95ad94c)

Setelah menjalankan script, sebuah jendela baru akan muncul menampilkan daftar gambar yang ditemukan dalam folder "foto gigi".
Klik pada salah satu gambar di daftar untuk melihat pratinjau gambar di jendela yang sama.
Jika gambar yang diinginkan telah dipilih, klik tombol "Proses Gambar".

### 4.Melihat Hasil:

![Screenshot_3](https://github.com/okkysatria/Dental-Image-Processing/assets/84434840/d789b40d-d45e-494a-942a-00d145feb980)

Gambar yang diproses akan ditampilkan dalam jendela matplotlib, menunjukkan hasil deteksi tepi menggunakan metode yang telah diimplementasikan.
Hasil juga akan disimpan sebagai file gambar dengan nama method_sobel.jpg di direktori yang sama dengan script.
