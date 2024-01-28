import os
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

from utils import split_data

from model import (
    get_model_parameter,
    load_example_data,
    get_data_for_cross_validation,
    CNN2D_Regression_Analysis,
    CNN2D_Classification_Analysis,
    CNN2D_Classifier,
    CNN2D,
)

# Define paths
base_img_dir = "../data/PD_Normal"  # Replace with your main folder path
positive_dir = "../data/PD_Normal/Pos"
negative_dir = "../data/PD_Normal/Neg"

# Create directories for train, validation, and test sets
sub_dirs = ["train", "val", "test"]
classes = ["pd_pos", "pd_neg"]
for sub_dir in sub_dirs:
    for class_name in classes:
        os.makedirs(os.path.join(base_img_dir, sub_dir, class_name), exist_ok=True)

# Splitting pd+ and pd- data
split_data(
    positive_dir,
    os.path.join(base_img_dir, "train", "pd_pos"),
    os.path.join(base_img_dir, "val", "pd_pos"),
    os.path.join(base_img_dir, "test", "pd_pos"),
)
split_data(
    negative_dir,
    os.path.join(base_img_dir, "train", "pd_neg"),
    os.path.join(base_img_dir, "val", "pd_neg"),
    os.path.join(base_img_dir, "test", "pd_neg"),
)

print("Done splitting data.....")

# Set the path for your directories
base_dir = "../data/PD_Normal"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

# Set some parameters
image_size = (40, 40)
batch_size = 32

# Create ImageDataGenerators for training and validation
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="binary",
)

validation_generator = val_datagen.flow_from_directory(
    val_dir, target_size=image_size, batch_size=batch_size, class_mode="binary"
)

print("Image shape: ", train_generator.image_shape)
# Model Parameters
para = get_model_parameter("../data/Example_Model_Parameters/FCNN_Classifier.txt")
subnetwork_para = get_model_parameter(
    "../data/Example_Model_Parameters/CNN2D_SubNetwork.txt"
)
para.update(subnetwork_para)

# model = Sequential(
#     [
#         Conv2D(16, (3, 3), activation="relu", input_shape=(40, 40, 3)),
#         BatchNormalization(),
#         MaxPooling2D(2, 2),
#         Conv2D(32, (3, 3), activation="relu"),
#         BatchNormalization(),
#         MaxPooling2D(2, 2),
#         Conv2D(64, (3, 3), activation="relu"),
#         BatchNormalization(),
#         MaxPooling2D(2, 2),
#         Flatten(),
#         Dense(64, activation="relu", kernel_regularizer="l2"),
#         Dropout(0.5),
#         Dense(1, activation="sigmoid"),
#     ]
# )

# Model Definition
CNN = CNN2D(para, input_data_dim=[[train_generator.image_shape[0], train_generator.image_shape[1]]], num_class=2, dropout=0.25)
model = CNN.model

# Define the EarlyStopping callback
early_stopping = EarlyStopping(
    monitor="val_accuracy",  # Monitor the validation accuracy
    patience=50,  # Number of epochs with no improvement after which training will be stopped
    mode="max",  # Mode 'max' because we want to monitor the increase of the metric
    restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored quantity
    verbose=1,
)

accuracy_threshold = 0.98  # 60%


class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("val_accuracy") > accuracy_threshold:
            print("\nReached 90% validation accuracy so cancelling training!")
            self.model.stop_training = True


# Compile the model
#model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=100,  # Set a high number since EarlyStopping will terminate the process
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[early_stopping, CustomCallback()],
)

history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ["loss", "val_loss"]].plot()
history_frame.loc[:, ["accuracy", "val_accuracy"]].plot()

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=image_size, batch_size=5, class_mode="binary", shuffle=False
)

# Predict
test_steps_per_epoch = math.ceil(test_generator.samples / test_generator.batch_size)
predictions = model.predict(test_generator, steps=test_steps_per_epoch)

# Convert predictions to binary (0 or 1)
binary_predictions = [1 if p > 0.5 else 0 for p in predictions]

# Get true labels from the test generator
true_labels = test_generator.classes

# Calculate confusion matrix
conf_matrix = confusion_matrix(true_labels, binary_predictions)

# Plot confusion matrix using seaborn for better visualization
plt.figure(figsize=(10, 7))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="g",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"],
)
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix")
plt.show()

# Calculate and print classification report
class_report = classification_report(
    true_labels, binary_predictions, target_names=["Negative", "Positive"]
)
print("Classification Report:\n", class_report)
