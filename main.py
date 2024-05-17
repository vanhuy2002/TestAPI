import asyncio
from fastapi import FastAPI, UploadFile
from typing import List
import cv2
import numpy as np
from keras.models import load_model
import authentication
import time
from datetime import datetime

app = FastAPI()

word_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
dig_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}

model = load_model('model_hand.h5')
digitModel = load_model('model_digit_new.h5')

authentication.initialize_firebase_app()

@app.get("/")
async def get_processed_images():
    return {"message": "No data to get"}

async def process_image(image: UploadFile, index: int):
    image_data = await image.read()
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_GRAYSCALE)
    prediction = await detect(img)
    return index, prediction

async def process_image_digit(image: UploadFile, index: int):
    image_data = await image.read()
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_GRAYSCALE)
    prediction = await detect_digit(img)
    return index, prediction

@app.post("/upload_images/")
async def upload_image(uid: str, images: List[UploadFile]):
    start = time.time()
    current_time_seconds = time.time()
    current_datetime = datetime.fromtimestamp(current_time_seconds)
    milliseconds_since_epoch = int(current_datetime.timestamp() * 1000)
    print("Thoi diem nhan anh: " + str(milliseconds_since_epoch))

    if authentication.check_uid_exist(uid):
        tasks = []
        for index, image in enumerate(images):
            if index < 9:  # First 9 images are digits
                tasks.append(process_image_digit(image, index))
            else:  # Remaining images are words
                tasks.append(process_image(image, index))
        
        # Chạy tất cả các tác vụ đồng thời và chờ chúng hoàn thành
        results = await asyncio.gather(*tasks)
        
        # Sắp xếp kết quả theo chỉ số ban đầu
        results.sort(key=lambda x: x[0])
        
        # Lấy kết quả thực sự
        final_results = [result[1] for result in results]
        
        print("Tong time: " + str(time.time() - start))
        return {"message": "Request successful", "results": final_results}
    else:
        return {"message": "You have no permission to access"}

async def detect(img):
    max_letter = await crop_letter_from_image(img)
    if max_letter is not None:
        padding_pixels = 5  # Số lượng pixel bạn muốn thêm vào từ mỗi phía
        img_padded = await add_padding_and_resize(max_letter, padding_pixels)
        img_resize = cv2.resize(img_padded, (28, 28))
        img_final = np.reshape(img_resize, (1, 28, 28, 1))
        predictions = model.predict(img_final)
        max_prediction = np.max(predictions)
        img_pred = word_dict[np.argmax(predictions)]
        
        accuracy_threshold = 0.8  # Ngưỡng chấp nhận
        if max_prediction < accuracy_threshold:
            img_pred = 'O'
    else:
        img_pred = 'X'
    return img_pred

async def detect_digit(img):
    max_letter = await crop_letter_from_image1(img)
    if max_letter is not None:
        padding_pixels = 6  # Số lượng pixel bạn muốn thêm vào từ mỗi phía
        img_padded = await add_padding_and_resize(max_letter, padding_pixels)
        img_resize = cv2.resize(img_padded, (28, 28))
        img_final = np.reshape(img_resize, (1, 28, 28, 1))
        predictions = digitModel.predict(img_final)
        img_pred = dig_dict[np.argmax(predictions)]
    else:
        img_pred = '_'
    return img_pred

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

async def crop_image(img, crop_height, crop_width):
    height, width = img.shape[:2]
    start_row = (height - crop_height) // 2
    start_col = (width - crop_width) // 2
    end_row = start_row + crop_height
    end_col = start_col + crop_width
    cropped_img = img[start_row:end_row, start_col:end_col]
    return cropped_img

# Hàm thêm khoảng trắng màu đen vào ảnh grayscale và giữ nguyên tỷ lệ khi resize
async def add_padding_and_resize(image, padding_pixels):
    height, width = image.shape[:2]
    padded_image = np.zeros((height + 2 * padding_pixels, width + 2 * padding_pixels), dtype=np.uint8)
    padded_image[padding_pixels:padding_pixels + height, padding_pixels: padding_pixels + width] = image
    return padded_image

async def crop_letter_from_image(img):
    img = await crop_image(img, 33, 75)
    img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    non_zero_pixels = cv2.findNonZero(thresh)
    if non_zero_pixels is None:
        return None
    x, y, w, h = cv2.boundingRect(non_zero_pixels)
    cropped_letter = thresh[y:y+h, x:x+w]
    return cropped_letter

async def crop_letter_from_image1(img):
    img = cv2.fastNlMeansDenoising(img, None, 10, 5, 100)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    non_zero_pixels = cv2.findNonZero(thresh)
    if non_zero_pixels is None:
        return None
    x, y, w, h = cv2.boundingRect(non_zero_pixels)
    cropped_letter = thresh[y:y+h, x:x+w]
    return cropped_letter
