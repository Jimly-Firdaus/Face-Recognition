# Face Recognition
Tugas Besar 2 IF 2123 Aljabar Linier dan Geometri
Aplikasi Nilai Eigen dan EigenFace pada Pengenalan Wajah (Face Recognition) 

## Table of Contents
* [General Info](#general-information)
* [Tampilan Program](#tampilan-program)
* [How To Run](#how-to-run)
* [Project Structure](#project-structure)
* [Credits](#credits)

## General Information
Pengenalan wajah (Face Recognition) adalah teknologi biometrik yang bisa dipakai untuk mengidentifikasi wajah seseorang untuk berbagai kepentingan khususnya keamanan. Program pengenalan wajah melibatkan kumpulan citra wajah yang sudah disimpan pada database lalu berdasarkan kumpulan citra wajah tersebut, program dapat mempelajari bentuk wajah lalu mencocokkan antara kumpulan citra wajah yang sudah dipelajari dengan citra yang akan diidentifikasi.

## Tampilan Program
![Main View](./src/assets/tampilanProgram.jpg)

## How To Run
1. Pastikan anda berada pada dir `src` dengan :
```shell
cd src
```
2. Jalankan perintah berikut:
```shell
py interface.py
```
3. Jika berhasil, maka akan muncul prompt aplikasi seperti pada tampilan program di atas.

## Project Structure
```bash
.
│
├───bin
├───doc
│       README.md
│
├───src
│   │   beta.py
│   │   eigen.py
│   │   eigenface2.py
│   │   interface.py
│   │   main.py
│   │
│   └───assets
│
└───test
```

## Credits
This project is implemented by:
1. Brian Kheng (13521049)
2. Jimly Firdaus (13521102)
3. Marcel Ryan A. (13521127)