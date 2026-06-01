import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image


# =========================
# 기본 설정
# =========================
st.set_page_config(
    page_title="X-ray 2단계 CNN 분류 시스템",
    page_icon="🩻",
    layout="wide"
)

DEVICE = torch.device("cpu")

STAGE1_MODEL_PATH = "models/densenet121_chest_xray_5class_best-v2.pth"
STAGE2_MODEL_PATH = "models/densenet121_chest_xray_4class_best.pth"


# =========================
# DenseNet121 모델 생성
# =========================
def build_densenet121(num_classes):
    model = models.densenet121(weights=None)
    in_features = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model


# =========================
# 모델 로드
# =========================
@st.cache_resource
def load_models():
    stage1_ckpt = torch.load(STAGE1_MODEL_PATH, map_location=DEVICE)
    stage2_ckpt = torch.load(STAGE2_MODEL_PATH, map_location=DEVICE)

    stage1_classes = stage1_ckpt["class_names"]
    stage2_classes = stage2_ckpt["class_names"]

    stage1_model = build_densenet121(num_classes=len(stage1_classes))
    stage1_model.load_state_dict(stage1_ckpt["model_state_dict"])
    stage1_model.to(DEVICE)
    stage1_model.eval()

    stage2_model = build_densenet121(num_classes=len(stage2_classes))
    stage2_model.load_state_dict(stage2_ckpt["model_state_dict"])
    stage2_model.to(DEVICE)
    stage2_model.eval()

    stage1_image_size = stage1_ckpt.get("image_size", 224)
    stage2_image_size = stage2_ckpt.get("image_size", 224)

    return (
        stage1_model,
        stage2_model,
        stage1_classes,
        stage2_classes,
        stage1_image_size,
        stage2_image_size
    )


# =========================
# 전처리
# =========================
def preprocess_image(image, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)
    return tensor.to(DEVICE)


# =========================
# 예측 함수
# =========================
def predict_multiclass(model, input_tensor, class_names):
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]

    pred_idx = torch.argmax(probs).item()
    pred_class = class_names[pred_idx]
    confidence = probs[pred_idx].item()

    prob_dict = {
        class_names[i]: probs[i].item()
        for i in range(len(class_names))
    }

    return pred_class, confidence, prob_dict


# =========================
# 앱 화면
# =========================
st.title("X-ray 2단계 CNN 분류 시스템")

st.warning(
    "이 시스템은 의료 진단용이 아니라 AI 기반 X-ray 이미지 분류 프로젝트용입니다."
)

try:
    (
        stage1_model,
        stage2_model,
        stage1_classes,
        stage2_classes,
        stage1_image_size,
        stage2_image_size
    ) = load_models()

except Exception as e:
    st.error("모델 로딩 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()


# =========================
# 사이드바
# =========================
st.sidebar.header("설정")

st.sidebar.write("1단계 클래스")
st.sidebar.write(stage1_classes)

st.sidebar.write("2단계 클래스")
st.sidebar.write(stage2_classes)

normal_class_name = st.sidebar.selectbox(
    "1단계에서 NORMAL로 판단할 클래스명",
    stage1_classes,
    index=stage1_classes.index("NORMAL") if "NORMAL" in stage1_classes else 0
)


# =========================
# 파일 업로드
# =========================
uploaded_files = st.file_uploader(
    "X-ray 이미지를 업로드하세요",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

results = []

if uploaded_files:
    st.subheader("예측 결과")

    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(image, caption=uploaded_file.name, use_container_width=True)

        with col2:
            stage1_input = preprocess_image(image, stage1_image_size)

            stage1_pred, stage1_conf, stage1_probs = predict_multiclass(
                stage1_model,
                stage1_input,
                stage1_classes
            )

            st.write(f"1단계 결과: **{stage1_pred}**")
            st.write(f"1단계 확률: **{stage1_conf:.4f}**")

            stage1_prob_df = pd.DataFrame(
                list(stage1_probs.items()),
                columns=["Class", "Probability"]
            )
            st.bar_chart(stage1_prob_df.set_index("Class"))

            if stage1_pred == normal_class_name:
                final_result = "NORMAL"
                stage2_pred = "-"
                stage2_conf = "-"

                st.success(f"최종 결과: {final_result}")

            else:
                stage2_input = preprocess_image(image, stage2_image_size)

                stage2_pred, stage2_conf, stage2_probs = predict_multiclass(
                    stage2_model,
                    stage2_input,
                    stage2_classes
                )

                final_result = stage2_pred

                st.error(f"최종 결과: {final_result}")
                st.write(f"2단계 결과: **{stage2_pred}**")
                st.write(f"2단계 확률: **{stage2_conf:.4f}**")

                stage2_prob_df = pd.DataFrame(
                    list(stage2_probs.items()),
                    columns=["Class", "Probability"]
                )
                st.bar_chart(stage2_prob_df.set_index("Class"))

            results.append({
                "filename": uploaded_file.name,
                "stage1_result": stage1_pred,
                "stage1_confidence": round(stage1_conf, 4),
                "stage2_result": stage2_pred,
                "stage2_confidence": stage2_conf if stage2_conf == "-" else round(stage2_conf, 4),
                "final_result": final_result
            })

        st.divider()

    result_df = pd.DataFrame(results)

    st.subheader("전체 결과표")
    st.dataframe(result_df, use_container_width=True)

    csv = result_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="결과 CSV 다운로드",
        data=csv,
        file_name="xray_prediction_results.csv",
        mime="text/csv"
    )

else:
    st.info("이미지를 업로드하면 예측이 시작됩니다.")