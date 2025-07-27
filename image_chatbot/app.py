import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# --- PHẦN 1: CÀI ĐẶT VÀ TẢI MÔ HÌNH ---

# Sử dụng decorator của Streamlit để cache mô hình, giúp không phải tải lại mỗi lần
@st.cache_resource
def load_model():
    """Tải mô hình AI và trả về."""
    print("Đang tải mô hình... (chỉ tải một lần)")
    # Sử dụng EfficientNetB0, một mô hình cân bằng giữa tốc độ và độ chính xác
    model = tf.keras.applications.EfficientNetB0(weights='imagenet')
    print("Tải mô hình thành công.")
    return model

model = load_model()

# --- PHẦN 2: HÀM LOGIC NHẬN DẠNG ---

def recognize_image(image):
    """Nhận đối tượng ảnh từ Pillow, xử lý và trả về kết quả."""
    try:
        # Thay đổi kích thước ảnh theo yêu cầu của mô hình
        image = image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

        # Dự đoán
        predictions = model.predict(img_array)
        # Hàm decode_predictions đã bao gồm nhãn, không cần tải file JSON riêng
        decoded_predictions = tf.keras.applications.efficientnet.decode_predictions(predictions, top=1)[0]
        
        top_prediction = decoded_predictions[0]
        label_name = top_prediction[1].replace('_', ' ')
        confidence = top_prediction[2] * 100
        
        # Định dạng kết quả
        description = f"Đối tượng được xác định là **{label_name.capitalize()}** với độ tin cậy **{confidence:.2f}%**."
        return description

    except Exception as e:
        return f"Đã có lỗi xảy ra khi xử lý ảnh: {e}"

# --- PHẦN 3: XÂY DỰNG GIAO DIỆN WEB ---

st.set_page_config(layout="centered", page_title="Bot Nhận Dạng Ảnh")

st.title("🤖 Bot Nhận Dạng Hình Ảnh")
st.write("Tải lên một bức ảnh, và AI sẽ cho bạn biết nó nhìn thấy gì.")

# Widget để người dùng tải file lên
uploaded_file = st.file_uploader("Chọn một tệp ảnh...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Mở và hiển thị ảnh người dùng đã tải lên
    image = Image.open(uploaded_file).convert("RGB")
    
    st.image(image, caption="Ảnh bạn đã tải lên", use_column_width=True)
    
    # Khi có ảnh, bắt đầu phân tích
    with st.spinner("Bot đang phân tích..."):
        result = recognize_image(image)
        st.success("Phân tích hoàn tất!")
        st.markdown(f"### Kết quả:")
        st.markdown(result)
