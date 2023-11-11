import cv2
import numpy as np
import os
from keras.models import load_model

model = load_model('emotion_detection_model.h5')

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear',
               3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Đường dẫn đến thư mục chứa các ảnh
image_dir = 'test_dir'

# Lấy danh sách tệp ảnh trong thư mục
image_files = [os.path.join(image_dir, f) for f in os.listdir(
    image_dir) if os.path.isfile(os.path.join(image_dir, f))]

for image_file in image_files:
    frame = cv2.imread(image_file)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)

    for x, y, w, h in faces:
        sub_face_img = gray[y:y + h, x:x + w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        print(label)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, labels_dict[label], (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(0)

    # Kiểm tra nút 'q' để đóng cửa sổ hoặc tiếp tục xem ảnh tiếp theo
    if key == ord('q'):
        break

cv2.destroyAllWindows()
