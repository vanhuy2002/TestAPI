from fastapi import FastAPI, UploadFile, File
from typing import List
import os
import cv2
import numpy as np
from keras.models import load_model
import json

app = FastAPI()

word_dict = {0:'A',1:'B',2:'C',3:'D'}
model = load_model('model_hand.h5')
@app.post("/upload_images/")
async def upload_image(images: List[UploadFile] = File(...)):

    result = {}

    for image in images: 

        image_data = await image.read()
        img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

        prediction = detect(img) # hàm nhận dạng ký tự

        result[image.filename] = prediction # lưu kết quả vào dict
  
    # Chuyển kết quả sang JSON
    json_result = json.dumps(result)

    return json_result


def detect(img):
    max_letter = crop_letters_from_image(img)
    if max_letter is not None:
        # Thêm khoảng trắng bằng cách mở rộng ảnh
        padding_pixels = 2  # Số lượng pixel bạn muốn thêm vào từ mỗi phía
        max_letter = cv2.copyMakeBorder(max_letter, padding_pixels, padding_pixels, padding_pixels, padding_pixels, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        img = cv2.resize(max_letter, (28, 28))
        img_final = np.reshape(img, (1, 28, 28, 1))
        predictions = model.predict(img_final)
        max_prediction = np.max(predictions)
        img_pred = word_dict[np.argmax(predictions)]
        # Kiểm tra số lượng vật thể
        accuracy_threshold = 0.97  # Ngưỡng chấp nhận
        
        if max_prediction < accuracy_threshold:
            img_pred = 'O'
    else:
        img_pred = 'X'
    
    return img_pred


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


def crop_image(img, crop_height, crop_width):
    height, width = img.shape[:2]
    
    start_row = (height - crop_height) // 2
    start_col = (width - crop_width) // 2
    
    end_row = start_row + crop_height
    end_col = start_col + crop_width
    
    cropped_img = img[start_row:end_row, start_col:end_col]
    
    return cropped_img
    
import cv2
import numpy as np


def crop_letters_from_image(img):
    # Đọc hình ảnh từ tệp và xử lý

    img = crop_image(img, 33, 85)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # Loại bỏ nhiễu
    img = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Tìm các đối tượng (chữ cái) trong hình ảnh bằng contour detection
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    max_area = 0
    max_letter = None

    # Duyệt qua các đối tượng tìm thấy
    for contour in contours:
        # Tính toán diện tích của đối tượng
        area = cv2.contourArea(contour)

        # Nếu diện tích đủ lớn và lớn hơn diện tích lớn nhất hiện tại
        if area > 30 and area > max_area:
            x, y, w, h = cv2.boundingRect(contour)

            # Cắt chữ cái từ hình ảnh gốc
            max_letter = img[y:y+h, x:x+w]

            # Chuyển đổi thành hình ảnh nhị phân để làm nổi bật
            max_letter = cv2.threshold(max_letter, 160, 255, cv2.THRESH_BINARY_INV)[1]

             # Đảo ngược màu sắc của ảnh (nền đen, chữ trắng)
            # max_letter = cv2.bitwise_not(max_letter)

            max_area = area  # Cập nhật diện tích lớn nhất
    return max_letter


