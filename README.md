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

We tested multiple models for object classification on the HIT-UAV thermal dataset. Below is a comparison of accuracy, precision, recall, and F1 score across four different approaches:

| Model                      | Accuracy | Precision | Recall | F1 Score |
|----------------------------|----------|-----------|--------|----------|
| YOLOv8 + MGO-CNN (Ours)    | **83.0%**| 81.4%     | 83.0%  | 79.1%    |
| YOLOv8 + Basic CNN         | 80.3%    | 79.0%     | 80.3%  | 78.6%    |
| YOLOv5 + MGO-CNN           | 82.3%    | 80.2%     | 82.3%  | 79.6%    |
| YOLOv5 + SVM (Baseline)    | 78.0%    | 75.6%     | 78.0%  | 74.2%    |

> 📌 **Note:** All experiments used CLAHE preprocessing, same dataset splits, and were tested on the HIT-UAV thermal dataset.

## 🚀 Try It Locally

```bash
git clone https://github.com/DHRUV\ DHAR/Thermal-UAV-Surveillance.git
cd Thermal-UAV-Surveillance
pip install -r requirements.txt

