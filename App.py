import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
import tempfile
import os

# Page config
st.set_page_config(page_title="Oral Cancer Detection", layout="wide")

# Custom CSS styling
# Custom CSS styling (updated with readable font color)
st.markdown("""
    <style>
        .main {
            background-color: #fafafa;
        }
        .stApp {
            font-family: 'Segoe UI', sans-serif;
        }
        .status-box {
            background-color: #fff;
            color: #333;  /* ✅ Dark readable font */
            border: 1px solid #e0e0e0;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 15px;
            font-size: 16px;
        }
        .status-good {
            border-left: 5px solid #00c853;
        }
        .status-bad {
            border-left: 5px solid #d50000;
        }
        .result-image {
            border: 1px solid #ddd;
            border-radius: 8px;
            margin-top: 10px;
        }
    </style>
""", unsafe_allow_html=True)


# Load YOLO model
@st.cache_resource
def load_model():
    model_path = "runs/detect/train/weights/best.pt"
    return YOLO(model_path)

model = load_model()

# Sidebar
with st.sidebar:
    st.title("🩺 Oral Cancer Detector")
    st.markdown("Upload an image of the oral cavity to detect possible cancerous lesions.")
    uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])
    st.markdown("---")
    st.caption("Model: YOLOv8 • Confidence ≥ 0.25 • Class: Lesion")

# Main layout
if uploaded_file is not None:
    col1, col2 = st.columns([1, 1.2])

    # Load image
    image = Image.open(uploaded_file).convert("RGB")

    # Save temp image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)
        temp_path = tmp.name

    # Run prediction
    results = model(temp_path)
    boxes = results[0].boxes
    class_names = model.names
    img_area = np.array(image).shape[0] * np.array(image).shape[1]
    lesion_area = 0

    # Result image
    result_img = results[0].plot()
    result_pil = Image.fromarray(result_img)

    # -------- Left Column (Details) --------
    with col1:
        st.subheader("🔍 Detection Summary")

        # Cancer Status
        if len(boxes) > 0:
            st.markdown('<div class="status-box status-bad"><b>🔴 Cancer Detected</b><br>Lesions identified in the image.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-box status-good"><b>✅ No Cancer Detected</b><br>No lesions found.</div>', unsafe_allow_html=True)

        # Class and confidence
        with st.expander("📊 Detection Details", expanded=True):
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = class_names.get(cls, f"Class {cls}")

                x1, y1, x2, y2 = map(int, xyxy)
                area = (x2 - x1) * (y2 - y1)
                lesion_area += area

                st.write(f"• **{label}** | Confidence: `{conf:.2f}` | Area: `{area} px²`")

            if lesion_area > 0:
                lesion_percent = (lesion_area / img_area) * 100
                st.info(f"🧮 **Lesion Coverage:** `{lesion_percent:.2f}%` of the image")

        # Download result
        buffered = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        result_pil.save(buffered.name)
        with open(buffered.name, "rb") as f:
            st.download_button("📥 Download Annotated Image", f, file_name="prediction_result.png")

    # -------- Right Column (Images) --------
    with col2:
        tabs = st.tabs(["🖼 Uploaded", "📸 Detection Output"])
        with tabs[0]:
            st.image(image, caption="Original Image", use_container_width=True, output_format="PNG")
        with tabs[1]:
            st.image(result_pil, caption="Annotated Result", use_container_width=True, output_format="PNG")

    # Cleanup temp
    os.remove(temp_path)
