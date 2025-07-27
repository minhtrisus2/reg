import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify, url_for, send_from_directory
from werkzeug.utils import secure_filename

# --- CÀI ĐẶT BAN ĐẦU ---
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions

print("Đang tải mô hình EfficientNetB7, quá trình này có thể mất vài phút...")
model = EfficientNetB7(weights='imagenet')
print("Tải mô hình thành công!")

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- HÀM LOGIC CHÍNH ---

### THAY ĐỔI 1: Thêm từ điển dịch thuật ###
TRANSLATION_DICT = {
    'golden_retriever': 'Chó Golden Retriever',
    'miniature_pinscher': 'Chó Phốc',
    'chihuahua': 'Chó Chihuahua',
    'tabby': 'Mèo mướp',
    'tiger_cat': 'Mèo vằn',
    'lifeboat': 'Thuyền cứu sinh',
    'speedboat': 'Tàu cao tốc',
    'laptop': 'Máy tính xách tay',
    'sports_car': 'Xe thể thao',
    # Bạn có thể thêm các bản dịch khác vào đây
}

def recognize_image(image_path):
    """Hàm này nhận ảnh, phân tích và trả về một báo cáo nhận dạng chuyên nghiệp bằng tiếng Việt."""
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(600, 600))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=1)[0]
        
        top_prediction = decoded_predictions[0]
        
        english_label = top_prediction[1]
        confidence = top_prediction[2] * 100

        # Dịch nhãn sang tiếng Việt, nếu không có trong từ điển thì dùng tiếng Anh
        vietnamese_label = TRANSLATION_DICT.get(english_label, english_label.replace('_', ' '))

        ### THAY ĐỔI 2: Tạo báo cáo phân tích chuyên nghiệp ###
        # Code mới, chỉ bao gồm thông tin chính
        report = f"Đối tượng được xác định: <b>{vietnamese_label.capitalize()}</b><br>Độ tin cậy: {confidence:.2f}%"
        return report
        
    except Exception as e:
        return f"Đã có lỗi xảy ra khi xử lý ảnh: {e}"

# --- CÁC ĐƯỜNG DẪN (ROUTES) CỦA WEB --- (Giữ nguyên)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recognize_image', methods=['POST'])
def recognize_image_route():
    if 'file' not in request.files:
        return jsonify({'error': 'Không có file nào được gửi đi.'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Chưa chọn file nào.'})
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        description = recognize_image(filepath)
        
        return jsonify({'description': description})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- KHỞI ĐỘNG ỨNG DỤNG ---
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(port=5000, debug=True)