import firebase_admin
from firebase_admin import credentials, auth

def initialize_firebase_app():
    # Đường dẫn đến tệp json của Firebase Admin SDK
    cred = credentials.Certificate("adminsdk.json")
    # Khởi tạo ứng dụng Firebase
    firebase_admin.initialize_app(cred)

def check_uid_exist(uid):
    try:
        # Lấy thông tin người dùng từ Firebase Authentication
        user = auth.get_user(uid)
        return True
    except firebase_admin.auth.UserNotFoundError as e:
        return False

