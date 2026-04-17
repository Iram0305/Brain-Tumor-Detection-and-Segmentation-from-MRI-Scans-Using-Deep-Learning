# 🧠 Brain Tumor Detection and Segmentation from MRI Scans

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-OpenCV%20%7C%20Deep%20Learning-green)

A web-based computer vision application built with Streamlit that analyzes MRI scans to detect and segment brain tumors. This tool is designed to assist in rapid medical image analysis by highlighting potential tumor regions directly on the uploaded scans.

---

## 🚀 Live Application & Resources

| Resource | Link |
|---|---|
| 🌐 Live App | [Brain Tumor Detection Application](https://brain-tumor-detection-and-segmentation.streamlit.app/) |
| 🎥 Demo Video | [Watch the Application in Action](https://drive.google.com/file/d/1iATU5dDvwC9CtSFtIlFGHJm8Re6aP2lu/view?usp=sharing) |
| 🖼️ Sample MRI Images | [Google Drive Test Folder](https://drive.google.com/drive/folders/1IEpRBAXOwvFbiLAMAgaswJU8kkHgtvb5?usp=sharing) |

---

## ✨ Key Features

- **Intuitive Web Interface** — Built purely in Python using Streamlit for a seamless user experience.
- **Real-Time Inference** — Upload an MRI scan and get instantaneous detection and segmentation overlays.
- **High Accuracy** — Utilizes a custom-trained Deep Learning model tailored for medical imaging.
- **Plug-and-Play Testing** — Download sample images from the provided Drive folder to test the model immediately.

---

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| Frontend / Backend | Streamlit |
| Language | Python 3.8+ |
| Computer Vision | OpenCV, PIL |
| Machine Learning | PyTorch / TensorFlow / YOLOv8 *(update as applicable)* |

---

## 💻 Local Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/brain-tumor-detection.git
cd brain-tumor-detection
```

### 2. Set up a virtual environment *(Recommended)*

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
├── app.py                 # Main Streamlit application script
├── requirements.txt       # Project dependencies
├── models/                # Saved weights for the detection/segmentation model
├── utils/                 # Helper functions for image processing and inference
└── README.md              # Project documentation
```
