from fastapi import FastAPI, UploadFile, File

app = FastAPI()

@app.post("/upload_image/")
async def upload_image(image: UploadFile):
    # Đảm bảo file tải lên là hình ảnh
    if not image.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
        return {"error": "Only image files (jpg, jpeg, png, gif) are allowed."}

    # Lưu file ảnh vào thư mục lưu trữ
    # with open(f"uploads/{image.filename}", "wb") as file:
    #    file.write(image.file.read())

    return {"message": "Image uploaded successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
