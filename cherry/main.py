import os
import time
import cv2
import numpy as np
from keras.models import load_model

current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = "0"
np.set_printoptions(suppress=True)

model = load_model("models/keras_model.h5", compile=False)
class_names = open("models/labels.txt", "r").readlines()

camera = cv2.VideoCapture(0)

last_index = -1
last_change_time = 0
debounce_time = 0.5
stable_index = -1

while True:
    ret, image = camera.read()
    display_image = image.copy()

    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1

    prediction = model.predict(image, verbose=0)
    index = np.argmax(prediction)
    confidence_score = prediction[0][index]

    current_time = time.time()

    if index != last_index:
        last_index = index
        last_change_time = current_time

    if current_time - last_change_time >= debounce_time:
        stable_index = last_index

    if stable_index != -1:
        label = f"{class_names[stable_index].strip()} - {round(prediction[0][stable_index] * 100)}%"
        cv2.putText(display_image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Webcam Image", display_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
