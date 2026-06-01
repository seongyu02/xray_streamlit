import streamlit as st
import pandas as pd
import torch
from PIL import Image

from utils.preprocess import preprocess_image
from utils.predict import predict_stage1, predict_stage2


st.set_page_config(
    page_title="X-ray 2단계 CNN 분류 시스템",
    page_icon="🩻",
    layout="wide"
)


@st.cache_resource
def load_models():
    stage1_model = torch.load("models/densenet121_chest_xray_5class_best-v2.pth", map_location="cpu")
    stage2_model = torch.load("models/densenet121_chest_xray_4class_best.pth", map_location="cpu")

    stage1_model.eval()
    stage2_model.eval()

    return stage1_model, stage2_model


st.title("X-ray 2단계 CNN 분류 시스템")

st.warning("이 시스템은 의료 진단용이 아니라 AI 기반 X-ray 이미지 분류 프로젝트용입니다.")

stage1_model, stage2_model = load_models()

threshold = st.sidebar.slider(
    "1단계 ABNORMAL 판단 threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

stage1_size = st.sidebar.selectbox("1단계 입력 크기", [224, 512], index=0)
stage2_size = st.sidebar.selectbox("2단계 입력 크기", [224, 512], index=0)

uploaded_files = st.file_uploader(
    "X-ray 이미지를 업로드하세요",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

results = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(image, caption=uploaded_file.name, use_container_width=True)

        with col2:
            stage1_input = preprocess_image(image, image_size=stage1_size)
            stage1_label, stage1_conf = predict_stage1(
                stage1_model,
                stage1_input,
                threshold
            )

            if stage1_label == "NORMAL":
                final_result = "NORMAL"
                stage2_label = "-"
                stage2_conf = "-"

                st.success(f"최종 결과: {final_result}")
                st.write(f"1단계 결과: {stage1_label}")
                st.write(f"1단계 확률: {stage1_conf:.4f}")

            else:
                stage2_input = preprocess_image(image, image_size=stage2_size)
                stage2_label, stage2_conf, stage2_probs = predict_stage2(
                    stage2_model,
                    stage2_input
                )

                final_result = stage2_label

                st.error(f"최종 결과: {final_result}")
                st.write(f"1단계 결과: {stage1_label}")
                st.write(f"1단계 확률: {stage1_conf:.4f}")
                st.write(f"2단계 결과: {stage2_label}")
                st.write(f"2단계 확률: {stage2_conf:.4f}")

                prob_df = pd.DataFrame(
                    list(stage2_probs.items()),
                    columns=["Class", "Probability"]
                )
                st.bar_chart(prob_df.set_index("Class"))

            results.append({
                "filename": uploaded_file.name,
                "stage1_result": stage1_label,
                "stage1_confidence": round(stage1_conf, 4),
                "stage2_result": stage2_label,
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