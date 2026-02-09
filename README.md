# Cashlifting R-2-1-D

A computer vision project that performs **object detection for cash / currency recognition and localization** using traditional image processing and/or machine learning methods.

This repository demonstrates how to detect and extract regions of interest (like currency notes) in images or video frames — a foundational skill for applications such as automated cash counters, smart kiosks, and document analysis systems.

---

## 🚀 Project Overview

Cash or currency detection has practical uses in:
✔ Automated ATM cash slot monitoring  
✔ Smart kiosks and payment terminals  
✔ Document scanning applications  
✔ Robotics that interact with physical currency  

This project provides a step-by-step pipeline to:
1. Load image or video input
2. Preprocess for smoothing / noise reduction
3. Detect currency regions
4. Highlight and optionally crop detected regions

---

## 🧠 Key Features

| Feature | Description |
|---------|-------------|
| Image / Video Input | Supports both image files and video feed |
| Preprocessing | Noise reduction, color/edge filtering |
| Contour Detection | Extract and localize shapes that resemble currency |
| Output | Draw bounding shapes and display results |

---

## 🛠️ Tech Stack

- Python 3.8+
- OpenCV (cv2)
- NumPy

---

## 📦 Installation

Clone this repository:

```bash
git clone https://github.com/Hari7383/Cashlifting_R-2-1-D.git
cd Cashlifting_R-2-1-D
```
Create and activate a virtual environment (recommended):
```
python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate         # Windows
```
