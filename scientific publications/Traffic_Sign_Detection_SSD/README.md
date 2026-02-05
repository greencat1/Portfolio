# Traffic Sign Detection Using SSD

## 📄 Publication Information
**Type:** Conference proceedings paper  
**Language:** Russian  
**Year:** 2024  

**Author:**  
- Sazontov I. Yu.

**Affiliation:**  
Moscow Aviation Institute (National Research University), Moscow, Russia

---

## 🏛 Conference
**International Scientific Conference  
“Actual Problems of Applied Mathematics, Computer Science and Mechanics”**  
Voronezh, Russia, December 4–6, 2023

Organizer: Voronezh State University

---

## 📚 Source
Published in:

**Actual Problems of Applied Mathematics, Computer Science and Mechanics**  
Proceedings of the International Scientific Conference  
Voronezh, 2024  
Pages: **255–260**

eLIBRARY ID: **66562793**  
Indexed in: **Russian Science Citation Index (RSCI)**

🔗 https://elibrary.ru/item.asp?id=66562793

---

## 🧠 Abstract
This paper addresses the problem of traffic sign detection using deep learning methods.  
The proposed solution is based on the **Single Shot Detector (SSD)** architecture trained on the **Russian Traffic Sign Dataset (RTSD)**.

The dataset was preprocessed to reduce class imbalance by removing rare classes.  
The model was trained and evaluated using the **mean Average Precision (mAP)** metric and tested for overfitting.

---

## 🗂 Dataset
- **Name:** Russian Traffic Sign Dataset (RTSD)
- **Images:** 59,188
- **Resolution:** 1280×720 – 1920×1080
- **Conditions:** different seasons, lighting conditions, and weather
- **Annotations:** bounding boxes (full-gt.csv)
- **Preprocessing:** removal of classes with fewer than 1500 samples

---

## ⚙️ Methods
- Model: **Single Shot Detector (SSD)**
- Framework: **PyTorch**
- Training environment: **Google Colab**
- Optimization: **Adam**
- Loss monitoring and overfitting control
- Evaluation metric: **mAP (mean Average Precision)**
- Non-Maximum Suppression (NMS)

---

## 📊 Results
- Stable decrease of training loss
- Growth of mAP during training
- No significant overfitting observed
- Most frequent confusion: traffic sign vs background

---

## 📎 Supplementary Materials
- 📄 [Full paper (PDF)](Paper.pdf)
- 🏅 [Conference participation certificate (PDF)](Certificate.pdf)
