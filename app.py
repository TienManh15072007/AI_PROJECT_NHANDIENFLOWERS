import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import urllib.request

# 1. Cấu hình trang và chèn Background hình hoa thiên nhiên Việt Nam
st.set_page_config(page_title="App Nhận Diện Các Loại Hoa", layout="wide")

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1507290439931-a861b5a38200?q=80&w=1920&auto=format&fit=crop");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.85) !important;
}
.stMarkdown, .stText, stWidgetLabel {
    background-color: rgba(255, 255, 255, 0.6);
    padding: 5px;
    border-radius: 5px;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

@st.cache_resource
def download_and_load_model():
    model_path = "flower_model.h5"
    
    if not os.path.exists(model_path):
        # Giữ nguyên cấu trúc tải từ GitHub, chỉ thay đổi tên file lưu trữ tương ứng
        url = "https://github.com/TienManh15072007/AI_PROJECT_NHANDIENFLOWERS/releases/download/v1.0/flower_model.h5"
        try:
            urllib.request.urlretrieve(url, model_path)
        except Exception as e:
            st.error(f"Lỗi tải file mô hình hoa: {e}")
            return None
            
    if os.path.exists(model_path):
        return load_model(model_path)
    return None

model = download_and_load_model()

st.sidebar.title("Thanh Công Cụ")
app_mode = st.sidebar.selectbox("Chọn chức năng", ["Dự đoán qua Webcam/Ảnh", "Huấn luyện mô hình (Training)"])

if app_mode == "Dự đoán qua Webcam/Ảnh":
    st.title("🌸 Nhận diện Các Loài Hoa (CNN)")
    
    if model is None:
        st.warning("⚠️ Chưa tải được mô hình hoa. Vui lòng kiểm tra lại link tải!")
    else:
        st.success("✅ Đã tải mô hình nhận diện hoa thành công!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 1. Sử dụng Camera chụp hoa")
            camera_image = st.camera_input("Chụp ảnh hoa từ Webcam")
            
        with col2:
            st.markdown("### 2. Tải ảnh hoa từ máy")
            uploaded_file = st.file_uploader("Chọn file ảnh hoa (jpg, png)", type=["jpg", "jpeg", "png"])
            
        img_to_predict = camera_image if camera_image else uploaded_file
        
        if img_to_predict is not None:
            image = Image.open(img_to_predict).convert('RGB')
            st.image(image, caption="Ảnh hoa đầu vào", use_column_width=True)
            
            img_width, img_height = 200, 200
            img_resized = image.resize((img_width, img_height))
            img_array = np.array(img_resized)
            
            img_for_prediction = img_array.astype('float32') / 255.0
            img_for_prediction = np.expand_dims(img_for_prediction, axis=0)
            
            if st.button("🔍 Tiến hành Dự đoán", type="primary"):
                with st.spinner("Đang phân tích loài hoa..."):
                    preds = model.predict(img_for_prediction)
                    digit = int(np.argmax(preds))
                    confidence = np.max(preds) * 100
                    
                    # Giữ nguyên đúng tuyển tập 22 nhãn lớp, thay bằng 22 loại hoa đặc trưng
                   
                    class_names = [
                        "Hoa Cúc", "Hoa Hồng", "Hoa Hướng Dương", "Hoa Ly", "Hoa Sen"
                    ]
                    
                    predicted_name = class_names[digit]
                    
                    st.success(f"**Kết quả dự đoán loài hoa: {predicted_name}**")
                    st.info(f"Độ tin cậy (Confidence): {confidence:.2f}%")
            

elif app_mode == "Huấn luyện mô hình (Training)":
    st.title("⚙️ Huấn luyện Mô hình CNN Nhận Diện Hoa")
    
    st.info("Nhập đường dẫn chứa dữ liệu dữ liệu ảnh hoa để huấn luyện.")
    
    train_dir = st.text_input("Đường dẫn thư mục dữ liệu hoa:", "data_loai_hoa/")
    epochs = st.number_input("Số lần học (Epochs):", min_value=1, max_value=100, value=10)
    batch_size = st.number_input("Batch Size:", min_value=8, max_value=128, value=32)
    
    if st.button("🚀 Bắt đầu Huấn luyện"):
        if not os.path.exists(train_dir):
            st.error(f"Không tìm thấy thư mục: {train_dir}")
        else:
            with st.spinner("Đang tiến hành huấn luyện mô hình hoa..."):
                img_width, img_height = 200, 200
                
                train_datagen = ImageDataGenerator(
                    rescale=1.0/255, 
                    rotation_range=30,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode="nearest",
                    validation_split=0.2 
                )                                                                                                                                                                                                                                                                
                
                train_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(img_width, img_height),
                    batch_size=batch_size,
                    class_mode="categorical",
                    subset='training'
                )
                
                val_generator = train_datagen.flow_from_directory(
                    train_dir,
                    target_size=(img_width, img_height),
                    batch_size=batch_size,
                    class_mode="categorical",
                    subset='validation'
                )

                # Giữ nguyên cấu trúc mạng CNN gốc (22 nút đầu ra tại tầng Dense cuối)
                model = Sequential([
                    Conv2D(32, (3,3), activation="relu", input_shape=(img_width, img_height, 3)), 
                    MaxPooling2D(2,2),
                    Conv2D(64, (3,3), activation="relu"),
                    MaxPooling2D(2,2),
                    Conv2D(128, (3,3), activation="relu"),
                    MaxPooling2D(2,2),
                    Flatten(),
                    Dense(128, activation="relu"),
                    Dropout(0.5),
                    Dense(22, activation="softmax")
                ])                                                                                                                                                                                 
                
                model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
                
                st.text("Cấu trúc mô hình mạng hoa:")
                model.summary(print_fn=lambda x: st.text(x))

                history = model.fit(
                    train_generator, 
                    validation_data=val_generator,
                    epochs=epochs
                )
                
                model.save("flower_model.h5")
                st.success("✅ Huấn luyện hoàn tất! Mô hình hoa đã được lưu.")

                fig, ax = plt.subplots()
                ax.plot(history.history['accuracy'], label="Kết quả huấn luyện")
                ax.plot(history.history['val_accuracy'], label="Độ chính xác xác thực")
                ax.set_xlabel("Số lần học")
                ax.set_ylabel("Độ chính xác")
                ax.legend()
                st.pyplot(fig)
