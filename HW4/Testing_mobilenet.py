import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dropout, Flatten, Dense, Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, RMSprop
# from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os
# from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.metrics import average_precision_score
import numpy as np 
from tensorflow.keras.models import load_model
import time

# loading model
model = load_model('models/MobileNet_tf1_v1.h5')

# loading testing data
test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory('Dataset/Test', target_size=(224,224),
                                                  batch_size=64, class_mode='binary', shuffle=False)

# Make predictions on the test data
start=time.time()

num_test_sample=len(test_set)
num_classes=len(test_set.class_indices)

predicted_probabilities=model.predict(test_set,steps=num_test_sample)
predicted_labels=np.argmax(predicted_probabilities,axis=1)

end = time.time()
accumulate_time = (end - start)

print(f"\nTime for predicting : {accumulate_time}\n")

true_labels=test_set.classes

# report=classification_report(true_labels,predicted_labels)


score = 0
for i in range(num_test_sample):
    if predicted_labels[i] == true_labels[i]:
        score += 1
        
report = (float(score) / num_test_sample) * 100
print(f"accuracy = {report:.2f}%")


# print(report)