# 🔥 Thermal UAV Object Detection for Night Surveillance

A hybrid deep learning system combining **YOLOv8** and **CNN optimized via Mountain Gazelle Optimization (MGO)** for detecting and classifying objects in **thermal UAV images**, designed for real-time night-time surveillance applications.

## 📌 Project Overview

- 🚁 Thermal video from UAVs used for night surveillance
- 🧠 YOLOv8 for object detection
- 📊 CNN classifier tuned with MGO
- 📈 HIT-UAV dataset: thermal frames with 4 classes

## 🧠 Key Features

| Module        | Description                              |
|---------------|------------------------------------------|
| YOLOv8        | Real-time object detection               |
| MGO-CNN       | Optimized CNN for classification         |
| CLAHE         | Image contrast enhancement               |
| HIT-UAV       | Real-world UAV thermal dataset           |

## 📁 Project Structure
📦 thermal-uav-surveillance
├── notebooks/
├── scripts/
├── docs/
├── models/
├── dataset/
├── cropped_dataset/
├── data.yaml
├── requirements.txt
└── README.md


## 📊 Results

| Model             | Accuracy | Precision | Recall | F1 Score |
|------------------|----------|-----------|--------|----------|
| YOLOv8 + MGO-CNN | **83%**  | 81.4%     | 83%    | 79.1%    |

## 🚀 Try It Locally

```bash
git clone https://github.com/DHRUV\ DHAR/Thermal-UAV-Surveillance.git
cd Thermal-UAV-Surveillance
pip install -r requirements.txt

