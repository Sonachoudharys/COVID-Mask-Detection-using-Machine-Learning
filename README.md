# COVID-Mask-Detection-using-Machine-Learning

This project demonstrates real-time face mask detection using Python, OpenCV, and TensorFlow/Keras. It captures video from your webcam and detects whether a person is wearing a mask or not, displaying the result with colored bounding boxes and labels.

#🛠 Technologies Used

- Python
- OpenCV (opencv-python)
- TensorFlow / Keras
- NumPy
- Matplotlib

#🌟 Features

- Real-time video stream from webcam
- Mask detection with bounding box and label
- Displays “Mask” or “No Mask” with color indication
- Trained CNN model with binary classification
- Face detection using Haar Cascade
- Robust to frames with or without faces

#🧠 Labels Detected

- Mask (Green Box)
- No Mask (Red Box)

#⚙️ How It Works

- The webcam captures a live frame.
- OpenCV detects faces using Haar Cascade.
- The detected face is resized and passed to the trained CNN model.
- The model predicts whether the person is wearing a mask.
- A colored rectangle is drawn: 
o	✅ Green: Mask 
o	❌ Red: No Mask
- Emotion label is shown with bounding box for each face.

#📦 Install Required Libraries

Use the following pip commands:

- pip install opencv-python
- pip install tensorflow
- pip install matplotlib
- pip install numpy
- pip install scikit-learn
