AI-POWERED CATTLE BREED IDENTIFICATION SYSYTEM:

📌 Overview
Field workers often misidentify cattle and buffalo breeds due to visual similarity between breeds. This leads to incorrect data collection, affecting research, breeding programs, and livestock management policies.
This project presents an AI-powered image classification system that automatically identifies cattle breeds from images. The system is designed as a real-world prototype, integrating a trained deep learning model with a backend API and a professional web-based user interface.

🎯 Objectives
- Automatically identify cattle breeds from images
- Reduce human error during breed registration
- Provide confidence-aware predictions for safer decision making
- Demonstrate an end-to-end AI system (ML + Backend + UI)

🧠 Key Features
- Trained on 23,350 images across 86 cattle breeds
- Deep Learning model using ResNet18
- Top-3 predictions with confidence scores
- Confidence-based decision logic
- Human-in-the-loop friendly (safe AI design)
- Professional web UI
- REST API using FastAPI

🏗️ System Architecture
User (Web UI)
     |
     v
FastAPI Backend (/predict)
     |
     v
PyTorch Model (ResNet18)
     |
     v
Prediction + Confidence
     |
     v
UI Display (Top-3 + Status)

🧪 Dataset Details
Total Images: 23,350
Number of Breeds: 86

Dataset split ensures proper evaluation and generalization.

🤖 Model Details
- Architecture: ResNet18 (Transfer Learning)
- Framework: PyTorch
- Training Strategy:Fine-tuning higher layers
- Strong data augmentation
- Confidence-based inference
- Training Environment: CPU (Laptop)

📊 Model Performance
- Metric	Value
- Training Epochs	10
- Validation Accuracy	74.22%
- Test Accuracy	77.55%
- Classes	86

Note: Due to the fine-grained nature of cattle breeds, confidence-aware predictions are used instead of forcing a single label.

🖥️ Web Interface
- The UI allows users to:
- Upload a cattle image
- Preview the uploaded image
- View Top-3 predicted breeds
- See confidence bars
- Understand prediction reliability through status labels

Prediction Status Logic
Confidence	Status
≥ 80%	CONFIDENT_PREDICTION
60–80%	NEEDS_HUMAN_CONFIRMATION
< 60%	UNKNOWN_BREED

⚙️ Tech Stack
Backend : FastAPI ,PyTorch ,Torchvision ,PIL

Frontend : HTML ,CSS ,JavaScript (Fetch API)

Storage : Local file system

JSON for class labels

🚀 How to Run the Project
1️⃣ Clone the repository
git clone <repository_url>
cd AI_breed_classification

2️⃣ Create & activate virtual environment
python -m venv venv
venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Train the model (optional)
python app/model.py

5️⃣ Start the backend server
uvicorn app.main:app

6️⃣ Open the web UI
http://127.0.0.1:8000

🧠 Design Decisions

 - ResNet18 chosen for balance between accuracy and computational efficiency
 - Top-3 predictions used instead of forcing Top-1
 - Confidence thresholding ensures real-world reliability
 - CPU-friendly training for accessibility
 - UI kept simple and framework-free for clarity

🎓 Learning Outcomes

- Deep learning model training & evaluation
- Handling large real-world datasets
- Transfer learning & fine-tuning
- API development using FastAPI
- Frontend-backend integration
- Practical AI system design

🔮 Future Enhancements

- GPU training for higher accuracy
- Add “Unknown” class explicitly
- Grad-CAM explainability
- Cloud deployment
- Prediction history & analytics
- Integrated chatbot assistant

🏁 Conclusion

This project demonstrates a complete AI pipeline — from data preparation and model training to backend integration and user-friendly deployment. It focuses on practical reliability rather than just raw accuracy, making it suitable for real-world applications.
