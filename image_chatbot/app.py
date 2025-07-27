import streamlit as st
from PIL import Image
import torch
import timm
import json
from urllib.request import urlopen

# --- PHẦN 1: TẢI MÔ HÌNH VÀ NHÃN (SỬ DỤNG PYTORCH) ---

@st.cache_resource
def load_model():
    """Tải mô hình AI MobileNetV3 bằng PyTorch và Timm."""
    # Sử dụng MobileNetV3, một mô hình rất nhẹ và hiệu quả
    model = timm.create_model('mobilenetv3_large_100', pretrained=True)
    model.eval() # Chuyển mô hình sang chế độ đánh giá
    return model

@st.cache_data
def load_labels():
    """Tải nhãn của ImageNet."""
    labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    labels = json.load(urlopen(labels_url))
    return labels

model = load_model()
labels = load_labels()

# --- PHẦN 2: HÀM LOGIC NHẬN DẠNG (SỬ DỤNG PYTORCH) ---

def recognize_image(image):
    """Nhận đối tượng ảnh, xử lý và trả về kết quả."""
    try:
        # Lấy cấu hình tiền xử lý của mô hình
        data_config = timm.data.resolve_data_config(model)
        transforms = timm.data.create_transform(**data_config, is_training=False)
        
        # Tiền xử lý ảnh
        tensor = transforms(image).unsqueeze(0)
        
        # Dự đoán
        with torch.no_grad():
            out = model(tensor)
            
        # Xử lý kết quả
        probabilities = torch.nn.functional.softmax(out[0], dim=0)
        top_prob, top_catid = torch.topk(probabilities, 1)
        
        confidence = top_prob.item() * 100
        label_name = labels[top_catid.item()].replace('_', ' ')
        
        # Định dạng kết quả
        description = f"Đối tượng được xác định là **{label_name.capitalize()}** với độ tin cậy **{confidence:.2f}%**."
        return description

    except Exception as e:
        return f"Đã xảy ra lỗi khi xử lý ảnh: {e}"

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
            st.markdown("### Kết quả:")
            st.markdown(result)
