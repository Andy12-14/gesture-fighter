# 🎮 Real-Time Gesture Controlled Gaming System (Street Fighter 6)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose%20Estimation-lightgrey)

> **Capstone Project 2026** > A computer vision pipeline that translates real-time physical body movements into low-latency game inputs using Deep Learning (LSTMs).

---

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [Demo](#-demo)
- [Installation](#-installation)
- [Usage Workflow](#-usage-workflow)
- [Project Structure](#-project-structure)
- [Troubleshooting & Performance](#-troubleshooting--performance)
- [Future Roadmap](#-future-roadmap)

---

## 📖 Project Overview
This project bridges the gap between **Computer Vision** and **Human-Computer Interaction (HCI)** in gaming. Instead of using a traditional controller or keyboard, the player uses their body to control the game character.

The system captures a video feed, extracts skeletal landmarks, recognizes temporal action sequences (like a punch or kick) using an LSTM Neural Network, and maps them to virtual controller inputs.

**Target Application:** *Street Fighter 6* (fighting games require low latency and precise distinct inputs).

---

## ⚙️ System Architecture

The project is divided into three core logical modules:

### 1. The Eyes (Vision) 👁️
* **Tech:** OpenCV & Google MediaPipe.
* **Function:** Captures real-time video (30-60 FPS) and extracts 33 3D body landmarks.
* **Optimization:** Filters data to upper-body keypoints to reduce noise.

### 2. The Brain (Inference) 🧠
* **Tech:** TensorFlow/Keras (LSTM Network).
* **Function:** Analyzes a sliding window of frames (e.g., the last 30 frames) to detect *motion* rather than static poses.
* **Classes:** `Idle`, `Punch`, `Kick`, `Block`, `Movements`.

### 3. The Hands (Actuation) 🎮
* **Tech:** `vgamepad` / `pydirectinput`.
* **Function:** Maps the high-probability prediction (e.g., "Punch" > 85%) to a virtual Xbox 360 controller input sent to the game engine.

---

## 💻 Installation

### Prerequisites
* Python 3.8 or higher
* Webcam (Standard USB or Laptop Integrated)
* Windows OS (Required for `vgamepad` virtual controller drivers)

### Setup
1.  **Clone the repository**
    ```bash
    git clone [https://github.com/yourusername/gesture-fighter.git](https://github.com/yourusername/gesture-fighter.git)
    cd gesture-fighter
    ```
2.  **Create and activate an conda environnement(if u have python > 3.10 )**
    ```bash
    conda create --name gesture-fighter python=3.10 -y
    conda activate gesture-fighter
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Usage Workflow

This project runs in 3 distinct phases.

### Phase 1: Data Collection
Before the AI can work, it must learn *your* specific movements.
```bash
python src/1_collect_data.py
```