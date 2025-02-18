import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model
import time

# 페이지 설정
st.set_page_config(page_title="귤 품종 분류기", layout="wide", initial_sidebar_state="expanded")

# 스타일 지정 (CSS)
st.markdown("""
    <style>
    /* ... (기존 스타일 유지) ... */
    
    /* 라디오 버튼 스타일 */
    .stRadio > label {
        color: #adb5bd;
        font-weight: bold;
    }
    .stRadio > div[role="radiogroup"] > label {
        background-color: #495057;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin-right: 0.5rem;
        color: white;
    }
    .stRadio > div[role="radiogroup"] > label:hover {
        background-color: #6c757d;
    }
    </style>
    """, unsafe_allow_html=True)

# 타이틀 및 설명
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>🍊 귤 품종 분류기</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>AI를 이용하여 귤 품종을 실시간으로 분류합니다. 카메라로 찍거나 파일을 업로드해보세요!</p>", unsafe_allow_html=True)

# 모델 및 클래스 로드 (캐싱 적용)
@st.cache_resource
def load_model_and_classes():
    model = load_model('keras_model.h5', compile=False)
    with open('labels.txt', 'r', encoding="utf-8") as f:
        class_names = [line.strip() for line in f.readlines()]
    return model, class_names

model, class_names = load_model_and_classes()

# 이미지 전처리 함수
def preprocess_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    return np.expand_dims(normalized_image_array, axis=0)

# 사이드바: 입력 방식 선택
with st.sidebar:
    st.markdown("<h2 style='color: white;'>설정</h2>", unsafe_allow_html=True)
    input_method = st.radio("이미지 입력 방식", ["카메라", "파일"])
    
    # 선택된 입력 방식에 따라 아이콘 표시
    if input_method == "카메라":
        st.markdown("📷 카메라로 사진 찍기")
    else:
        st.markdown("📁 파일 업로드하기")
    
    st.markdown("<div class='stAlert'>💡 팁: 선명한 이미지일수록 정확도가 높아집니다!</div>", unsafe_allow_html=True)

# 메인 레이아웃
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("<h2>이미지 입력</h2>", unsafe_allow_html=True)
    if input_method == "카메라":
        img_file_buffer = st.camera_input("사진 찍기")
    else:
        img_file_buffer = st.file_uploader("이미지 파일 선택", type=["png", "jpg", "jpeg"])

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer).convert('RGB')
        st.image(image, caption="입력된 이미지", use_column_width=True)

with col2:
    st.markdown("<h2>분석 결과</h2>", unsafe_allow_html=True)
    if img_file_buffer is not None:
        with st.spinner('이미지 분석 중...'):
            # 이미지 처리 및 예측
            start_time = time.time()
            data = preprocess_image(image)
            prediction = model.predict(data)
            end_time = time.time()

            # 결과 계산
            index = np.argmax(prediction)
            class_name = class_names[index]
            confidence_score = float(prediction[0][index])
            processing_time = end_time - start_time

            # 결과 표시
            st.success(f"분류 완료! (처리 시간: {processing_time:.2f}초)")
            st.metric("예측 품종", class_name)
            st.progress(confidence_score)
            st.metric("신뢰도", f"{confidence_score:.2%}")

            # 추가 정보: 상위 3개 클래스
            st.markdown("<h2>상위 3개 품종</h2>", unsafe_allow_html=True)
            top_3 = np.argsort(prediction[0])[-3:][::-1]
            for i in top_3:
                st.markdown(f"- **{class_names[i]}**: {prediction[0][i]:.2%}")

# 푸터
st.markdown("---")
st.markdown("Made with ❤️ by TaeMin | © 2025 귤 품종 분류기")
st.markdown("</div>", unsafe_allow_html=True)
