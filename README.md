# Recognizing-and-Detecting-Student-Alertness-Using-CNN

This project is a Flask-based web application that detects student alertness in real-time using Convolutional Neural Networks (CNNs). The system registers students, recognizes them using facial recognition, and evaluates their alertness state.

Features
- Student Registration: Users can register with their name and face data.
- Facial Recognition: Identifies students using a trained model.
- Alertness Detection: Determines whether a student is "Active" or "Not Active" using CNN models.
- Attendance Management: Stores student login time and status in an Excel sheet.
- Visualization: Displays confusion matrices and training results for performance evaluation.

Tech Stack
- Backend: Flask (Python)
- Frontend: HTML, CSS, JavaScript
- Database: SQLite (`users.db`)
- Machine Learning: Keras, TensorFlow (CNN-based alertness detection models)

1. Register a Student: Upload a face image to enroll.
2. Recognize Student: Face recognition is performed using OpenCV.
3. Detect Alertness: The model predicts whether the student is "Active" or "Not Active."
4. View Attendance: Results are logged in `attendance.xlsx`.



