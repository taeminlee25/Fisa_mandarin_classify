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
    /* 전체적인 배경색 */
    body {
        background-color: #f8f9fa;
    }
    /* 메인 컨테이너 스타일 */
    .main {
        max-width: 1200px;
        padding: 2rem;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    /* 제목 스타일 */
    h1 {
        color: #343a40;
        text-align: center;
        margin-bottom: 1rem;
    }
    /* 부제목 스타일 */
    h2 {
        color: #495057;
        margin-top: 1.5rem;
    }
    /* 설명 스타일 */
    p {
        color: #6c757d;
    }
    /* 프로그레스 바 스타일 */
    .stProgress > div > div > div > div {
        background-color: #28a745; /* 초록색 */
    }
    /* 버튼 스타일 */
    .stButton>button {
        background-color: #007bff; /* 파란색 */
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        cursor: pointer;
    }
    /* 사이드바 스타일 */
    .sidebar .sidebar-content {
        background-color: #343a40;
        color: white;
    }
    .sidebar h2 {
        color: white;
    }
    .sidebar p {
        color: #adb5bd;
    }
    /* 정보 박스 스타일 */
    .stAlert {
        background-color: #d1ecf1;
        color: #0c5460;
        border: 1px solid #bfe2e5;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 타이틀 및 설명
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center;'>🖼️ AI 이미지 분류기</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>최첨단 딥러닝 모델을 사용하여 이미지를 실시간으로 분류합니다. 카메라로 찍거나 파일을 업로드해보세요!</p>", unsafe_allow_html=True)

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
    input_method = st.radio("<p style='color: #adb5bd;'>이미지 입력 방식</p>", ["📷 카메라 사용", "📁 파일 업로드"], help="이미지 입력 방식을 선택하세요.")
    st.markdown("<div class='stAlert'>💡 팁: 선명한 이미지일수록 정확도가 높아집니다!</div>", unsafe_allow_html=True)

# 메인 레이아웃
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("<h2>이미지 입력</h2>", unsafe_allow_html=True)
    if input_method == "📷 카메라 사용":
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
            st.metric("예측 클래스", class_name)
            st.progress(confidence_score)
            st.metric("신뢰도", f"{confidence_score:.2%}")

            # 추가 정보: 상위 3개 클래스
            st.markdown("<h2>상위 3개 클래스</h2>", unsafe_allow_html=True)
            top_3 = np.argsort(prediction[0])[-3:][::-1]
            for i in top_3:
                st.markdown(f"- **{class_names[i]}**: {prediction[0][i]:.2%}")

# 푸터
st.markdown("---")
st.markdown("Made with ❤️ by TaeMin | © 2025 AI Image Classifier")
st.markdown("</div>", unsafe_allow_html=True)
