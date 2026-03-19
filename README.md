# AI-Integrated CCTV Monitoring System (CCTV-AI)

## 📌 Overview

The **CCTV-AI** project is a real-time video surveillance application built with Python and PyQt6. It integrates powerful deep learning models like YOLOv8/v10 to automate the detection of safety and process violations in industrial environments.

The system not only displays video feeds but also provides instantaneous alerts via the user interface, records abnormal logs into a database, and triggers physical alarm devices (Advantech PLC).

---

## ✨ Key Features

### 1. Multi-Channel Monitoring

- Supports simultaneous connection and processing of multiple camera streams (RTSP, local cameras).
- Flexible interface for switching between cameras via a tree view list.

### 2. Intelligent AI Analysis

The system uses custom YOLO models to detect:

- **PPE Check (Safety):** Detects whether workers are wearing gloves when working in hazardous areas.
- **Crowd Detection:** Alerts when too many people gather in a specific area for a defined period.
- **Process Violations:**
  - Detects hands touching the control panel without gloves.
  - Detects products (Copper rolls) left on the floor instead of their proper positions.
  - Trajectory tracking to distinguish between static and moving objects.
- **Hazardous Tools:** Detects scissors or other unauthorized tools.

### 3. Tower Light Monitoring

- Automatically identifies the status of signal lights (Green, Yellow, Red).
- Issues alerts when signal lights do not follow operational rules or when Red/Yellow lights are active.

### 4. PLC & Alarms Integration

- Connects to **Advantech PLC** via **Modbus TCP** protocol.
- Automatically triggers physical sirens/warning lights upon detecting serious violations.

### 5. Logging & API Management

- Records violation images and detailed information (Time, Camera Name, Violation Type) to the database via REST API.
- Displays a list of the latest alarms directly on the main interface.

---

## 🛠 Tech Stack

- **Language:** Python 3.x
- **GUI Framework:** PyQt6 (Professional GUI Framework)
- **Image Processing:** OpenCV
- **AI Models:** Ultralytics YOLOv8 / YOLOv10
- **PLC Communication:** PyModbus (Modbus TCP)
- **Database & API:** Requests (interface with Backend API)
- **Configuration:** YAML, Configparser

---

## 📂 Project Structure

- `Application.py`: Main entry point, manages the GUI and coordinates workers.
- `CameraCaptureWorker.py`: Handles video streams ensuring the "Freshest Frame" is captured.
- `CentroidTracker.py`: Object tracking algorithm.
- `CBInside/`, `CBOutside/`, `SSGLogic/`: Distinct logic modules for different camera locations, including specific AI configurations and processing logic.
- `setting.ini`: System configuration (PLC IP, Version, etc.).

---

## 🚀 Installation Guide

### 1. System Requirements

- Windows 10/11
- Python 3.9+
- NVIDIA GPU (Recommended for smooth AI performance)

### 2. Install Dependencies

```bash
pip install PyQt6 opencv-python ultralytics pymodbus requests pyyaml shapely python-dateutil
```

### 3. Configuration

- Edit `config.yaml` files within the module directories (e.g., `SSGLogic/config.yaml`) to set model paths (`.pt`) and monitor area coordinates (Region of Interest - ROI).
- Edit `setting.ini` to configure the PLC IP and other general settings.

---

## 📖 Usage

1. Ensure cameras and PLC are connected to the network.
2. Run `App-CCTV-AI.bat` or use the command:
   ```bash
   python Application.py
   ```
3. On the interface, select a camera from the tree list on the left to start monitoring.
4. Click **Connect PLC** to enable physical alarm features.
