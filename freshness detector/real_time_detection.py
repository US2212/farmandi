
# import cv2
# import tensorflow as tf
# import numpy as np
# from tensorflow.keras.preprocessing import image

# # Load the trained model
# model = tf.keras.models.load_model('freshness_model.h5')

# # Open the webcam
# cap = cv2.VideoCapture(0)

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break

#     # Resize the frame to match model input shape (150x150)
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (150, 150))
#     img_array = image.img_to_array(img) / 255.0  # Normalize the image
#     img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

#     # Predict freshness level
#     predictions = model.predict(img_array)
#     freshness_level = np.argmax(predictions[0])

#     # Map prediction to corresponding freshness level
#     if freshness_level == 0:
#         label = "Fresh"
#     elif freshness_level == 1:
#         label = "Medium"
#     else:
#         label = "Rotten"

#     # Display the prediction on the image
#     cv2.putText(frame, f'Freshness: {label}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     # Display the resulting frame
#     cv2.imshow('Webcam Feed', frame)

#     # Break the loop if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the webcam and close the OpenCV window
# cap.release()
# cv2.destroyAllWindows()
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('freshness_model.h5')

# Define class labels
classes = ['fresh', 'medium', 'rotten']

# Open the webcam
cap = cv2.VideoCapture(0)

# Set confidence threshold
confidence_threshold = 0.6  # Adjust as needed

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize the frame to match model input shape (150x150)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict freshness level
    predictions = model.predict(img_array)
    max_prob = np.max(predictions)
    predicted_class = classes[np.argmax(predictions)]

    if max_prob >= confidence_threshold:
        label = f"Freshness: {predicted_class.capitalize()} ({max_prob*100:.2f}%)"
    else:
        label = "No object detected"

    # Display the prediction on the image
    cv2.putText(frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Webcam Feed', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
