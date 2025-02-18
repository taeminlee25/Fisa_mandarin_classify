import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# 페이지 설정
st.set_page_config(page_title="이미지 분류기", layout="wide")

# 타이틀 및 설명
st.title("🖼️ 이미지 분류기")
st.markdown("이 앱은 업로드된 이미지를 분류합니다. 카메라로 찍거나 파일을 업로드해보세요!")

# 모델 및 클래스 로드
@st.cache_resource
def load_model_and_classes():
    model = load_model('keras_model.h5', compile=False)
    class_names = open('labels.txt', 'r', encoding="utf-8").readlines()
    return model, class_names

model, class_names = load_model_and_classes()

# 이미지 전처리 함수
def preprocess_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data

# 사이드바: 입력 방식 선택
input_method = st.sidebar.radio("이미지 입력 방식 선택", ["카메라 사용", "파일 업로드"])

# 메인 영역
col1, col2 = st.columns(2)

with col1:
    if input_method == "카메라 사용":
        img_file_buffer = st.camera_input("📸 사진 찍기")
    else:
        img_file_buffer = st.file_uploader("📤 이미지 파일 업로드", type=["png", "jpg", "jpeg"])

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer).convert('RGB')
        st.image(image, caption="입력된 이미지", use_container_width=True)

with col2:
    if img_file_buffer is not None:
        # 이미지 처리 및 예측
        data = preprocess_image(image)
        prediction = model.predict(data)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()
        confidence_score = float(prediction[0][index])

        # 결과 표시
        st.subheader("분류 결과")
        st.markdown(f"**클래스:** {class_name}")
        st.markdown(f"**신뢰도:** {confidence_score:.2%}")

        # 프로그레스 바로 신뢰도 표시
        st.progress(confidence_score)

        # 추가 정보 (예: 상위 3개 클래스)
        st.subheader("상위 3개 클래스")
        top_3 = np.argsort(prediction[0])[-3:][::-1]
        for i in top_3:
            st.markdown(f"- {class_names[i].strip()}: {prediction[0][i]:.2%}")

# 푸터
st.markdown("---")
st.markdown("Made with ❤️ by TaeMin")
