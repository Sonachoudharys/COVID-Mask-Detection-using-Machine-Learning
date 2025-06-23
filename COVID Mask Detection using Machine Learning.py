import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import classification_report

#  STEP 1: Data Loading 
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(
    r"C:\Users\Anita\Desktop\Shripriti Educational & IT hub\Task-2_COVID Mask Detection Using ML\dataset",
    target_size=(100, 100),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    r"C:\Users\Anita\Desktop\Shripriti Educational & IT hub\Task-2_COVID Mask Detection Using ML\dataset",
    target_size=(100, 100),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

#  STEP 2: Model 
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

#  STEP 3: Training 
history = model.fit(train_data, epochs=5, validation_data=val_data)

# STEP 4: Plot Accuracy 
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.title('Accuracy')
plt.show()

# STEP 5: Save Model
model.save("mask_detector_model.h5")

#  STEP 6: Real-Time Detection 
def detect_mask(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for x, y, w, h in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (100, 100)) / 255.0
        face_input = np.expand_dims(face_resized, axis=0)
        result = model.predict(face_input)[0][0]

        label = "Mask" if result < 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame

#  STEP 7: Webcam 
cap = cv2.VideoCapture(0)
print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = detect_mask(frame)
    cv2.imshow("Mask Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
