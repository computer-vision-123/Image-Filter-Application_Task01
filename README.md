<div align="center">

#  Smart Image Processing Studio

### Computer Vision Toolkit for Image Enhancement and Analysis

[![C++](https://img.shields.io/badge/C++-00599C?style=for-the-badge\&logo=cplusplus\&logoColor=white)]()
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge\&logo=python\&logoColor=white)]()
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge\&logo=opencv\&logoColor=white)]()
[![CMake](https://img.shields.io/badge/CMake-064F8C?style=for-the-badge\&logo=cmake\&logoColor=white)]()

</div>

---

# 📋 Overview

A modular **Computer Vision application** designed to explore classical image processing techniques through an interactive graphical interface.

The system allows users to experiment with fundamental computer vision operations including:

* Noise simulation and filtering
* Edge detection
* Frequency analysis
* Histogram processing
* Contrast enhancement
* Hybrid image generation

The application combines a **high-performance C++ backend** with a **Python GUI frontend**, allowing efficient computation alongside an intuitive user interface.

---

# ✨ Key Features

* **Noise Simulation** – Apply Gaussian and Salt & Pepper noise to images
* **Spatial Filtering** – Remove noise using Mean, Gaussian, and Median filters
* **Edge Detection** – Detect object boundaries using classical operators
* **Frequency Analysis** – Visualize frequency components of images
* **Histogram Processing** – Analyze pixel intensity distributions
* **Contrast Enhancement** – Improve visibility of low-contrast images
* **Hybrid Images** – Combine high and low frequency images to create perception-based effects
* **Interactive GUI** – Modular tab-based interface for exploring different algorithms

---

# 🖥️ Application Interface

The application provides an interactive **tab-based GUI** where each tab corresponds to a specific computer vision task.

### Available Tabs

| Module               | Description                                        |
| -------------------- | -------------------------------------------------- |
| Noise & Filters      | Apply noise and test filtering techniques          |
| Edge & Frequency     | Perform edge detection and frequency analysis      |
| Histogram & Contrast | Visualize histograms and enhance contrast          |
| Color & Hybrid       | Create hybrid images and manipulate color channels |

---

# 🏗️ System Architecture

```
┌─────────────────────┐
│  Python GUI         │
│  (Frontend)         │
│  Tab-based Interface│
└─────────┬───────────┘
          │ Python Bindings
          ↓
┌─────────────────────┐
│  C++ Backend        │
│  Image Processing   │
│  Algorithms         │
└─────────┬───────────┘
          │
          ↓
┌─────────────────────┐
│  Image Processing   │
│  Output & Results   │
└─────────────────────┘
```

The architecture separates **visual interaction from computation**, allowing optimized performance for image processing operations.

---

# 📂 Project Structure

```
Image-Filter-App/
│
├── CMakeLists.txt
│
├── backend/
│   ├── main_bindings.cpp
│   ├── noise_filters.cpp
│   ├── edge_freq.cpp
│   ├── hist_contrast.cpp
│   └── color_hybrid.cpp
│
├── frontend/
│   ├── main_window.py
│   ├── tab_noise_filters.py
│   ├── tab_edge_freq.py
│   ├── tab_hist_contrast.py
│   └── tab_color_hybrid.py
│
└── Helpers/
```

---

# ⚙️ Backend Modules

| Module            | Purpose                                      |
| ----------------- | -------------------------------------------- |
| noise_filters.cpp | Noise generation and filtering algorithms    |
| edge_freq.cpp     | Edge detection and frequency-domain analysis |
| hist_contrast.cpp | Histogram analysis and contrast enhancement  |
| color_hybrid.cpp  | Color processing and hybrid image creation   |
| main_bindings.cpp | Connects C++ backend with Python frontend    |

---

# 🚀 Getting Started

### Prerequisites

* Python **3.8+**
* C++ Compiler
* **CMake**
* Python libraries:

  * OpenCV
  * NumPy
  * Matplotlib

---

# 📥 Installation

## 1️⃣ Install Python Dependencies

```bash
pip install numpy opencv-python matplotlib
```

---

## 2️⃣ Build the Backend

```bash
mkdir build
cd build
cmake ..
make
```

---

## 3️⃣ Run the Application

```bash
python frontend/main_window.py
```

The graphical interface will open and allow users to explore the implemented image processing modules.

---

# 📊 Implemented Image Processing Techniques

| Category             | Algorithms                                  |
| -------------------- | ------------------------------------------- |
| Noise Modeling       | Gaussian Noise, Salt & Pepper Noise         |
| Filtering            | Mean Filter, Gaussian Filter, Median Filter |
| Edge Detection       | Sobel, Prewitt, Laplacian, Canny            |
| Histogram Processing | Histogram Visualization, Equalization       |
| Image Enhancement    | Contrast Adjustment                         |
| Hybrid Images        | Low-frequency + High-frequency blending     |

---

# 🧠 Learning Objectives

This project demonstrates core concepts in **Digital Image Processing and Computer Vision**, including:

* Pixel-level image manipulation
* Spatial filtering
* Edge detection techniques
* Frequency-domain analysis
* Histogram-based image enhancement
* Visual perception effects using hybrid images

---


## 👨‍💻 Contributors

- **Mohamed Ahmed Mahmoud**
- **Mariam Sherif Mohamed**
- **Mostafa Khaled Elsayed**
- **Sarah Sameh Mohamed**

<div align="center">

### Computer Vision Laboratory Project

Image Processing • Algorithm Design • Visual Computing


</div>
