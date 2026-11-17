
import hashlib
from io import BytesIO
import os
import altair as alt
import pandas as pd
import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError


# ==========================================
# 1. 系統設定
# ==========================================

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)
MAX_FILE_SIZE = 10 * 1024 * 1024

CLASS_NAMES = {
    "crazing": "龜裂紋",
    "inclusion": "夾雜物",
    "patches": "斑塊",
    "pitted_surface": "點蝕",
    "rolled_in_scale": "軋入氧化皮",
    "scratches": "刮痕",
}

st.set_page_config(
    page_title="AI 工業瑕疵辨識系統",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================
# 2. API 狀態檢查
# ==========================================

def check_api_status():
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )
        response.raise_for_status()

        health = response.json()

        is_ready = (
            health.get("status") == "ok"
            and health.get("model_loaded") is True
        )

        return is_ready, health

    except (requests.RequestException, ValueError):
        return False, {}


api_ready, health = check_api_status()


# ==========================================
# 3. Sidebar
# ==========================================

with st.sidebar:

    st.header("系統狀態")

    if api_ready:
        st.success("AI API 正常運作")
        st.write(
            f"運算設備：{health.get('device', 'Unknown')}"
        )
    else:
        st.error("AI API 尚未連線")
        st.caption(
            "請確認 FastAPI 已在 127.0.0.1:8000 啟動"
        )

    if st.button(
        "重新檢查 API",
        use_container_width=True,
    ):
        st.rerun()

    st.divider()

    st.header("模型資訊")
    st.write("模型：ResNet18")
    st.write("訓練方式：Layer4 Fine-tuning")
    st.write("分類數量：6")
    st.write("資料集：NEU-CLS")

    st.info(
        "本系統為六類鋼板表面瑕疵分類展示，"
        "不具備正常品（OK/NG）判定功能。"
    )

    st.divider()

    st.caption(
        "Industrial Defect AI | Portfolio Project"
    )


# ==========================================
# 4. 主頁面
# ==========================================

st.title("AI 工業瑕疵辨識系統")

st.caption(
    "Industrial Surface Defect Classification "
    "| ResNet18 Fine-tuning | PyTorch + FastAPI"
)

st.divider()

left_col, right_col = st.columns(
    [1, 1],
    gap="large",
)


# ==========================================
# 5. 圖片上傳
# ==========================================

uploaded_file = None
image_bytes = None
image_hash = None
image_valid = False

with left_col:

    st.subheader("圖片上傳")

    uploaded_file = st.file_uploader(
        "選擇金屬表面圖片",
        type=["bmp", "jpg", "jpeg", "png"],
        help="支援 BMP、JPG、PNG，最大 10 MB",
    )

    if uploaded_file is not None:

        image_bytes = uploaded_file.getvalue()

        if len(image_bytes) > MAX_FILE_SIZE:

            st.error("圖片大小不可超過 10 MB")

        else:

            try:
                with Image.open(
                    BytesIO(image_bytes)
                ) as img:

                    img.load()
                    image = img.convert("RGB")

                image_valid = True

                # 以檔案內容建立唯一識別值
                image_hash = hashlib.sha256(
                    image_bytes
                ).hexdigest()

                st.image(
                    image,
                    caption=uploaded_file.name,
                    use_container_width=True,
                )

                info_col1, info_col2 = st.columns(2)

                with info_col1:
                    st.metric(
                        "圖片寬度",
                        f"{image.width} px",
                    )

                with info_col2:
                    st.metric(
                        "圖片高度",
                        f"{image.height} px",
                    )

                st.caption(
                    f"檔案大小："
                    f"{len(image_bytes) / 1024:.2f} KB"
                )

            except (
                UnidentifiedImageError,
                OSError,
                ValueError,
            ):
                st.error("圖片格式錯誤或檔案無法讀取")

    else:
        st.info("請上傳一張金屬表面圖片")


# ==========================================
# 6. AI 辨識
# ==========================================

with right_col:

    st.subheader("AI 辨識結果")

    can_predict = (
        api_ready
        and image_valid
        and image_hash is not None
    )

    if st.button(
        "開始 AI 辨識",
        type="primary",
        disabled=not can_predict,
        use_container_width=True,
    ):

        with st.spinner("正在執行 AI 推論..."):

            try:

                # 準備 API 上傳資料
                files = {
                    "file": (
                        uploaded_file.name,
                        image_bytes,
                        uploaded_file.type
                        or "application/octet-stream",
                    )
                }

                response = requests.post(
                    f"{API_URL}/predict",
                    files=files,
                    timeout=60,
                )

                response.raise_for_status()

                result = response.json()

                # 儲存本次辨識結果
                st.session_state["prediction_result"] = result
                st.session_state["prediction_hash"] = image_hash

            except requests.HTTPError as e:

                st.session_state.pop(
                    "prediction_result", None
                )
                st.session_state.pop(
                    "prediction_hash", None
                )

                detail = response.text

                try:
                    detail = response.json().get(
                        "detail", detail
                    )
                except ValueError:
                    pass

                st.error(f"API 回傳錯誤：{detail}")

            except (
                requests.RequestException,
                ValueError,
            ) as e:

                st.session_state.pop(
                    "prediction_result", None
                )
                st.session_state.pop(
                    "prediction_hash", None
                )

                st.error(f"AI 推論失敗：{e}")

    # ======================================
    # 7. 判斷是否為目前圖片的辨識結果
    # ======================================

    result = st.session_state.get(
        "prediction_result"
    )

    saved_hash = st.session_state.get(
        "prediction_hash"
    )

    same_image = (
        image_valid
        and image_hash is not None
        and saved_hash == image_hash
    )

    if result is not None and same_image:

        predicted_class = result["predicted_class"]
        confidence = result["confidence"]

        chinese_name = CLASS_NAMES.get(
            predicted_class,
            predicted_class,
        )

        st.success(
            f"辨識結果：{chinese_name}"
        )

        # ==================================
        # 8. 辨識資訊
        # ==================================

        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:

            st.metric(
                label="預測分數",
                value=f"{confidence * 100:.2f}%",
            )

        with metric_col2:

            st.metric(
                label="推論耗時",
                value=(
                    f"{result['inference_time_ms']:.2f} ms"
                ),
            )

        st.caption(
            f"GPU：{result.get('device', 'Unknown')} "
            f"| Model Epoch："
            f"{result.get('model_epoch', 'Unknown')}"
        )

        st.caption(
            "預測分數為 Softmax 輸出，"
            "不代表模型在實際環境中的保證正確率。"
        )

        st.divider()

        # ==================================
        # 9. 預測分數圖表
        # ==================================

        st.subheader("各類別預測分數")

        probabilities = result["probabilities"]

        chart_data = pd.DataFrame({
            "類別": [
                CLASS_NAMES.get(name, name)
                for name in probabilities.keys()
            ],
            "預測分數 (%)": [
                float(score) * 100
                for score in probabilities.values()
            ],
        })

        chart_data = chart_data.sort_values(
            "預測分數 (%)",
            ascending=False,
        )

        # 橫向長條圖
        bars = alt.Chart(chart_data).mark_bar(
            color="#2563EB",
            cornerRadiusEnd=5,
        ).encode(
            x=alt.X(
                "預測分數 (%):Q",
                scale=alt.Scale(domain=[0, 105]),
                title="預測分數 (%)",
            ),
            y=alt.Y(
                "類別:N",
                sort="-x",
                title=None,
            ),
            tooltip=[
                alt.Tooltip("類別:N"),
                alt.Tooltip(
                    "預測分數 (%):Q",
                    format=".4f",
                ),
            ],
        )

            # 數值標籤
        labels = alt.Chart(chart_data).mark_text(
        align="left",
        baseline="middle",
        dx=6,
        fontSize=12,
    ).encode(
        x=alt.X(
            "預測分數 (%):Q",
            scale=alt.Scale(domain=[0, 105]),
        ),
        y=alt.Y("類別:N", sort="-x"),
        text=alt.Text(
            "預測分數 (%):Q",
            format=".2f",
        ),
    ).encode(
            x=alt.X(
                "預測分數 (%):Q",
                scale=alt.Scale(domain=[0, 100]),
            ),
            y=alt.Y(
                "類別:N",
                sort="-x",
            ),
            text=alt.Text(
                "預測分數 (%):Q",
                format=".2f",
            ),
        )

        chart = (bars + labels).properties(
            height=300,
        )

        st.altair_chart(
            chart,
            use_container_width=True,
        )

        # ==================================
        # 10. 詳細分數表
        # ==================================

        with st.expander("查看各類別詳細分數"):

            st.dataframe(
                chart_data,
                use_container_width=True,
                hide_index=True,
            )

        # ==================================
        # 11. API 原始資料
        # ==================================

        with st.expander("查看完整 API 回傳資料"):
            st.json(result)

    elif uploaded_file is None:

        st.info("請先上傳圖片")

    elif not image_valid:

        st.warning("請上傳有效的圖片")

    elif not api_ready:

        st.warning("請先啟動 FastAPI 服務")

    else:

        st.info(
            "點擊「開始 AI 辨識」執行推論"
        )
