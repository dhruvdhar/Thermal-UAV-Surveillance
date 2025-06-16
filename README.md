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

We evaluated four different object classification methods on the HIT-UAV thermal dataset. Below is a performance comparison:

| Method                        | Accuracy | Precision | Recall | F1 Score |
|------------------------------|----------|-----------|--------|----------|
| **YOLOv8 + MGO-CNN (Proposed)** | **0.83** | 0.814     | 0.83   | 0.791    |
| YOLOv5 + SVM                 | 0.823    | 0.802     | 0.823  | 0.796    |
| Faster R-CNN + Random Forest| 0.794    | 0.782     | 0.812  | 0.777    |
| SSD + K-Nearest Neighbors   | 0.820    | 0.759     | 0.805  | 0.781    |

> 📌 All models used CLAHE-enhanced thermal imagery from the HIT-UAV dataset. Evaluation was based on a consistent 80/20 train-test split.


## 🚀 Try It Locally

```bash
git clone https://github.com/DHRUV\ DHAR/Thermal-UAV-Surveillance.git
cd Thermal-UAV-Surveillance
pip install -r requirements.txt

