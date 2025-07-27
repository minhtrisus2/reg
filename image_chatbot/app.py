import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import json # SỬA LỖI: Thêm import json
from urllib.request import urlopen # SỬA LỖI: Thêm import urlopen

# --- PHẦN 1: CÀI ĐẶT VÀ TẢI MÔ HÌNH ---

@st.cache_resource
def load_model():
    """Tải mô hình AI và trả về."""
    print("Đang tải mô hình... (chỉ tải một lần)")
    model = tf.keras.applications.EfficientNetB0(weights='imagenet')
    print("Tải mô hình thành công.")
    return model

@st.cache_data
def load_labels():
    """Tải nhãn của ImageNet."""
    labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    # SỬA LỖI: Dùng json.load để đọc dữ liệu từ URL
    labels = json.load(urlopen(labels_url))
    return labels

model = load_model()
labels = load_labels()

# --- PHẦN 2: HÀM LOGIC NHẬN DẠNG ---

def recognize_image(image):
    """Nhận đối tượng ảnh từ Pillow, xử lý và trả về kết quả."""
    try:
        image = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

        predictions = model.predict(img_array)
        decoded_predictions = tf.keras.applications.efficientnet.decode_predictions(predictions, top=1)[0]
        
        top_prediction = decoded_predictions[0]
        english_label = top_prediction[1]
        confidence = top_prediction[2] * 100

        # Tạm thời chưa dùng từ điển dịch để đơn giản hóa
        label_name = english_label.replace('_', ' ')
        
        description = f"Đối tượng được xác định là **{label_name.capitalize()}** với độ tin cậy **{confidence:.2f}%**."
        return description

    except Exception as e:
        return f"Đã có lỗi xảy ra khi xử lý ảnh: {e}"

# --- PHẦN 3: XÂY DỰNG GIAO DIỆN WEB ---

st.set_page_config(layout="wide", page_title="Bot Nhận Dạng Ảnh")

st.title("🤖 Bot Nhận Dạng Hình Ảnh")
st.write("Tải lên một bức ảnh, và AI sẽ cho bạn biết nó nhìn thấy gì.")

uploaded_file = st.file_uploader("Chọn một tệp ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Ảnh bạn đã tải lên", use_container_width=True)
    
    with col2:
        with st.spinner("Bot đang phân tích..."):
            result = recognize_image(image)
            st.success("Phân tích hoàn tất!")
            st.markdown(f"### Kết quả:")
            st.markdown(result)
