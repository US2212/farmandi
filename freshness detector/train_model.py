# import os
# import tensorflow as tf
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# data_dir = r"C:\Users\dhaarna goyal\OneDrive\Desktop\FreshnessDetector\ProcessedData"
# img_size = (244, 244)
# batch_size = 32

# datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2
# )

# train_gen = datagen.flow_from_directory(
#     data_dir,
#     target_size=img_size,
#     batch_size=batch_size,
#     subset='training',
#     class_mode='categorical'
# )

# val_gen = datagen.flow_from_directory(
#     data_dir,
#     target_size=img_size,
#     batch_size=batch_size,
#     subset='validation',
#     class_mode='categorical'
# )

# model = Sequential([
#     Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
#     MaxPooling2D(2,2),
#     Conv2D(64, (3,3), activation='relu'),
#     MaxPooling2D(2,2),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(3, activation='softmax')  # 3 classes: fresh, medium, rotten
# ])

# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(train_gen, validation_data=val_gen, epochs=5)
# model.save("freshness_model.h5")
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import classification_report

data_dir = r"C:\Users\dhaarna goyal\OneDrive\Desktop\FreshnessDetector\ProcessedData"
img_size = (150, 150)
batch_size = 32

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    subset='training',
    class_mode='categorical'
)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=img_size,
    batch_size=batch_size,
    subset='validation',
    class_mode='categorical'
)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: fresh, medium, rotten
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_gen, validation_data=val_gen, epochs=5)
model.save("freshness_model.h5")

val_gen.reset()
predictions = model.predict(val_gen, steps=val_gen.samples // val_gen.batch_size + 1)
y_pred = np.argmax(predictions, axis=1)
y_true = val_gen.classes

# Generate classification report
target_names = list(val_gen.class_indices.keys())
report = classification_report(y_true, y_pred, target_names=target_names)
print(report)