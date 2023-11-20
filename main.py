from fastapi import FastAPI, UploadFile, File
from typing import List
import cv2
import numpy as np
from keras.models import load_model
import json
import authentication

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Có thể thay thế bằng danh sách các nguồn được phép
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

word_dict = {0:'A',1:'B',2:'C',3:'D'}
model = load_model('model_hand.h5')
processed_images = {}
@app.get("/get-processed-images/")
async def get_processed_images():
    global processed_images
    result_final = json.dumps(processed_images)
    return result_final

async def process_image(image: UploadFile = File(...)):
    image_data = await image.read()
    img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_GRAYSCALE)
    prediction = await detect(img)
    return {image.filename: prediction}

@app.post("/upload_images/")
async def upload_image(uid: str,images: List[UploadFile] = File(...)):
    if authentication.check_uid_exist(uid):
        results = []
        for image in images:
            result = await process_image(image)
            results.append(result)

        global processed_images
        processed_images = results
        return {"message": "Request successful", "results": results}
    else:
        return {"message": "You have no permisstion to access"}


async def detect(img):
    max_letter = await crop_letter_from_image(img)
    if max_letter is not None:
        # Thêm khoảng trắng bằng cách mở rộng ảnh
        padding_pixels = 5  # Số lượng pixel bạn muốn thêm vào từ mỗi phía
        img_padded = await add_padding_and_resize(max_letter, padding_pixels)
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
    authentication.initialize_firebase_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)


async def crop_image(img, crop_height, crop_width):
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
async def add_padding_and_resize(image, padding_pixels):
    height, width = image.shape[:2]

    # Thêm khoảng trắng màu đen
    padded_image = np.zeros((height + 2 * padding_pixels, width + 2 * padding_pixels), dtype=np.uint8)
    padded_image[padding_pixels:padding_pixels + height, padding_pixels:padding_pixels + width] = image

    return padded_image

async def crop_letter_from_image(img):
    # Đọc hình ảnh từ tệp và xử lý
    img = await crop_image(img, 33, 75)

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




