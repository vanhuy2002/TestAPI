from fastapi import FastAPI, UploadFile, File
from typing import List
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
        img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_GRAYSCALE)

        prediction = detect(img) # hàm nhận dạng ký tự

        result[image.filename] = prediction # lưu kết quả vào dict
  
    # Chuyển kết quả sang JSON
    json_result = json.dumps(result)

    return json_result


def detect(img):
    max_letter = crop_letter_from_image(img)
    if max_letter is not None:
        # Thêm khoảng trắng bằng cách mở rộng ảnh
        padding_pixels = 5  # Số lượng pixel bạn muốn thêm vào từ mỗi phía
        target_size = (28, 28)
        img_padded = add_padding_and_resize(max_letter, padding_pixels, target_size)
        img_resize = cv2.resize(img_padded, (28, 28))
        img_final = np.reshape(img_resize, (1, 28, 28, 1))
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

# Hàm thêm khoảng trắng màu đen vào ảnh grayscale và giữ nguyên tỷ lệ khi resize
def add_padding_and_resize(image, padding_pixels, target_size):
    height, width = image.shape[:2]

    # Thêm khoảng trắng màu đen
    padded_image = np.zeros((height + 2 * padding_pixels, width + 2 * padding_pixels), dtype=np.uint8)
    padded_image[padding_pixels:padding_pixels + height, padding_pixels:padding_pixels + width] = image

    # Resize ảnh và giữ nguyên tỷ lệ
    # resized_image = cv2.resize(padded_image, target_size, interpolation=cv2.INTER_AREA)

    return padded_image

def crop_letter_from_image(img):
    # Đọc hình ảnh từ tệp và xử lý
    img = crop_image(img, 33, 75)

    # Loại bỏ nhiễu
    img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    # printImg(img, "2")
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    # printImg(blur, "3")
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Tìm toàn bộ các điểm màu đen trong ảnh
    non_zero_pixels = cv2.findNonZero(thresh)

    # Nếu không có điểm màu đen, trả về None
    if non_zero_pixels is None:
        return None

    # Xác định bounding box của ảnh chữ cái
    x, y, w, h = cv2.boundingRect(non_zero_pixels)

    # Cắt ảnh theo bounding box
    cropped_letter = thresh[y:y+h, x:x+w]


    return cropped_letter




