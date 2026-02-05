# Application of YOLOv7 Neural Network Model for Traffic Sign Detection

## 📄 Publication Information
**Type:** Conference proceedings paper  
**Language:** Russian  
**Year:** 2023  

**Authors:**  
- Sazontov I. Yu.  

**Scientific Advisor:**  
- Rybakov K. A.

**Affiliation:**  
Moscow Aviation Institute (National Research University), Moscow, Russia

---

## 🏛 Conference
**XLIX International Youth Scientific Conference  
“Gagarin Readings – 2023”**  
Moscow, Russia, April 11–14, 2023

---

## 📚 Source
Published in the conference proceedings:

**Gagarin Readings – 2023**  
Proceedings of the XLIX International Youth Scientific Conference  
Publisher: *Pero Publishing House*, Moscow  
Pages: **525–526**

🔗 **eLIBRARY (RSCI indexed):**  
https://elibrary.ru/item.asp?id=54215630

---

## 🧠 Abstract
This paper explores the application of the YOLOv7 object detection model for traffic sign detection.  
The model was trained on a modified version of the RTSD (Russian Traffic Sign Dataset) with rare classes removed to address class imbalance.

The resulting model detects **12 traffic sign classes** and demonstrates a high detection accuracy of approximately **0.99 per class**.  
Despite the strong performance, the model is limited by the reduced number of detectable classes.

---

## 🗂 Dataset
- **Name:** RTSD (Russian Traffic Sign Dataset)
- **Images:** 59,188
- **Resolution:** 1280×720 – 1920×1080
- **Conditions:** different seasons, lighting conditions, and weather
- **Preprocessing:** removal of low-frequency classes (<1500 samples)

---

## ⚙️ Methods
- Model: **YOLOv7**
- Task: Object Detection
- Techniques:
  - Bounding box regression
  - Class probability prediction
  - Intersection over Union (IoU)
  - Non-Maximum Suppression (NMS)

---

## 📊 Results
- High detection accuracy (~0.99 per class)
- No clear overfitting observed
- Limitation: detection limited to 12 traffic sign classes

---

## 🔭 Future Work
- Dataset expansion to include rare traffic sign classes
- Improved robustness to complex backgrounds
- Deployment in real-time vision systems

---

## 📎 Supplementary Materials
- 📊 Conference presentation (PDF)
- 📄 Abstract / Thesis (PDF)
