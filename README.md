# VisionGuard 🚦

### Real-Time Helmet Compliance and Vehicle Detection System

VisionGuard is a real-time computer vision system designed to enhance road safety by detecting helmet compliance among motorcyclists and extracting vehicle registration details using license plate recognition. The system leverages deep learning-based object detection and OCR to process live or recorded video streams.

---

## Overview

This project implements a **multi-stage computer vision pipeline** that combines object detection and text recognition to monitor traffic compliance. It identifies vehicles, detects whether riders are wearing helmets, and extracts license plate information from video feeds.

---

## Features

* Real-time detection of **vehicles and helmet compliance**
* Automatic **license plate extraction using OCR**
* Supports **live camera feeds and recorded video input**
* Modular pipeline for **detection + recognition tasks**
* Visual output with annotated bounding boxes and predictions

---

## System Architecture

```
Input Video Stream
        ↓
   YOLOv5 Detection
   (Vehicles + Helmet)
        ↓
 Number Plate Detection
        ↓
      OCR Engine
        ↓
Annotated Output + Extracted Data
```

---

## Tech Stack

* **Python** – Core programming language
* **YOLOv5** – Object detection (vehicles, helmets, plates)
* **OpenCV** – Video processing and visualization
* **PyTorch** – Model training and inference
* **OCR (e.g., Tesseract/EasyOCR)** – License plate recognition

---

## Project Structure

```
├── models/                  # YOLO models and configs  
├── number_plates/           # Number plate detection modules  
├── riders_pictures/         # Sample dataset/images  
├── utils/                   # Helper functions  
├── main.py                  # Entry point for inference  
├── my_functions.py          # Core pipeline functions  
├── requirements.txt         # Dependencies  
└── *.pt / *.pth             # Trained model weights  
```

---

## Installation

```bash
git clone https://github.com/saketpol10/Vehicle-Identification-and-Helmet-Compliance-Detection-System.git
cd Vehicle-Identification-and-Helmet-Compliance-Detection-System
```
1. Create a virtual environment
```bash
python -m venv .venv
```
2. Activate the environment
   
On macOS/Linux:
```bash
source .venv/bin/activate
```
On Windows:
```bash
.venv\Scripts\activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
---

## Usage

Run the main pipeline:

```bash
python main.py
```

You can modify the input source in `main.py` to:

* Webcam feed
* Video file
* External camera stream

---

## Performance

* Supports **low-latency per-frame processing** for near real-time inference
* Capable of processing continuous video streams efficiently
* Performance may vary depending on hardware (GPU recommended)

---

## Sample Output

* Annotated video output with:

  * Vehicle detection
  * Helmet classification
  * License plate recognition

(*Refer to `output.avi` in the repository*)

---

## Contributing

Contributions are welcome! Feel free to fork the repo and submit a pull request.

---

## License

This project is open-source and available under the MIT License.

---

## 👤 Author

**Saket Pol**

* GitHub: https://github.com/saketpol10
* LinkedIn: https://www.linkedin.com/in/saket-pol

---
