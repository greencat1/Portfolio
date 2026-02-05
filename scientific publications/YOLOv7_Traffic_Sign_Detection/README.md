# Application of YOLOv7 for Traffic Sign Detection

## 📄 Publication
**Title:** Application of YOLOv7 Neural Network Model for Traffic Sign Detection  
**Author:** I. Yu. Sazontov  
**Scientific Advisor:** K. A. Rybakov  
**Institution:** Moscow Aviation Institute (National Research University)  
**Year:** 2023  

🔗 **eLIBRARY:** https://elibrary.ru/item.asp?id=54215630

---

## 🧠 Abstract
This work investigates the application of the YOLOv7 object detection model for traffic sign recognition.  
The model was trained on a modified version of the RTSD (Russian Traffic Sign Dataset) with rare classes removed.

The final model performs detection of **12 traffic sign classes**, achieving a detection accuracy of approximately **0.99 per class**.

---

## 🗂 Dataset
- **Name:** RTSD (Russian Traffic Sign Dataset)
- **Images:** 59,188
- **Resolution:** 1280×720 – 1920×1080
- **Conditions:** different seasons, lighting conditions, and weather
- **Preprocessing:** removal of low-frequency classes (<1500 samples)

---

## ⚙️ Model & Methods
- Architecture: **YOLOv7**
- Task: Object Detection
- Techniques:
  - Bounding box regression
  - Class probability prediction
  - Non-Maximum Suppression (NMS)
  - IoU-based filtering

---

## 📊 Results
- High detection accuracy (~0.99 for each class)
- No significant overfitting observed
- Limitation: detection limited to 12 traffic sign classes

---

## 📌 Limitations & Future Work
- Expand dataset to include rare traffic sign classes
- Improve generalization for complex backgrounds
- Deploy model in real-time embedded systems

---

## 📎 Materials
- 📊 [Presentation](Presentation.pdf)
- 📄 [Abstract / Thesis](Abstract.pdf)
