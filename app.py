"""
Banana Disease Detection & Ripeness Diagnosis App
Framework: Gradio 6
Dual Models:
  1. Banana Disease: DenseNet-121 (Version 24 from Kaggle, 66.67% Test Accuracy)
  2. Banana Ripeness: EfficientNet-B0 (Version 18 from Kaggle)
"""

import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from PIL import Image
import numpy as np

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Disable Gradio SSR mode for container stability
os.environ["GRADIO_SSR_MODE"] = "false"

# Hugging Face ZeroGPU compatibility
try:
    import spaces
except ImportError:
    class _MockSpaces:
        @staticmethod
        def GPU(fn=None, duration=None):
            def decorator(f):
                return f
            return decorator(fn) if fn else decorator
    spaces = _MockSpaces()

import gradio as gr

from inference import predict_banana_dual
from classes import BANANA_CLASSES, BANANA_VN_MAP, DISEASE_INFO, RIPENESS_CLASSES, RIPENESS_INFO

# Sample images directory
TEST_IMAGES_DIR = ROOT_DIR / "test-images"
SAMPLE_IMAGES = []
if TEST_IMAGES_DIR.is_dir():
    SAMPLE_IMAGES = [
        [str(TEST_IMAGES_DIR / f)]
        for f in os.listdir(TEST_IMAGES_DIR)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
    ]

# ============================================================
# Core Prediction Callback Function (Dual Inference)
# ============================================================
@spaces.GPU
def run_diagnosis(input_img, conf_threshold=0.15):
    """
    Handle dual inference (Disease + Ripeness) from Gradio interface.
    Returns:
        (annotated_cam_image, markdown_diagnosis_card)
    """
    if input_img is None:
        return None, "⚠️ **Vui lòng chọn hoặc tải một ảnh quả chuối lên để phân tích.**"

    result = predict_banana_dual(input_img, conf_threshold=conf_threshold, generate_cam=True)

    disease = result["disease"]
    d_info = result["disease_info"]
    ripeness = result["ripeness"]
    r_meta = result.get("ripeness_meta", {})

    confidence_val = disease.get("confidence", 0.0)

    # 1. Khối Chẩn Đoán Bệnh
    if confidence_val < conf_threshold:
        threshold_pct = int(round(conf_threshold * 100)) if conf_threshold <= 1.0 else int(conf_threshold)
        disease_md = f"""
### ⚠️ Chẩn Đoán Bệnh ({disease.get('percentage', 'N/A')} < {threshold_pct}%):
- **Tình trạng nhận diện**: **Chưa xác định rõ loại bệnh**

💡 *Khuyến nghị*: Hình ảnh có thể bị mờ, chụp xa hoặc triệu chứng bệnh chưa rõ nét.
"""
    else:
        fruit_type = d_info.get("fruit", "Quả Chuối")
        disease_md = f"""
### 🎯 Kết Quả Chẩn Đoán Bệnh:
- **Bệnh / Tình trạng**: **{disease.get('vn_name')}**
- **Độ tin cậy**: **{disease.get('percentage', 'N/A')}**
- **Đối tượng**: **{fruit_type}**
"""

    # 2. Khối Đánh Giá Mức Độ Chín (Version 18)
    ripeness_md = f"""
---
### 🍌 Đánh Giá Mức Độ Chín Của Chuối:
- **Cấp độ chín**: **{ripeness.get('icon', '🍌')} {ripeness.get('vn_name')}**
- **Độ tin cậy**: **{ripeness.get('percentage', 'N/A')}**
- **Đặc điểm dinh dưỡng**: {r_meta.get('description', 'Đang cập nhật...')}
- **Khuyến nghị sử dụng**: {r_meta.get('usage', 'Đang cập nhật...')}
"""

    final_card = f"{disease_md}{ripeness_md}"
    return result["gradcam_image"], final_card


# ============================================================
# Gradio User Interface Definition
# ============================================================
CUSTOM_CSS = """
/* App styling customized like Fruit_Detect */
.main-header {
    text-align: center;
    padding: 20px 10px 10px;
    background: linear-gradient(135deg, #10b981 0%, #059669 50%, #047857 100%);
    color: white;
    border-radius: 12px;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(5, 150, 105, 0.2);
}
.main-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 6px;
    color: #ffffff;
}
.main-header p {
    font-size: 1.05rem;
    opacity: 0.95;
    margin-bottom: 10px;
}
.badge-bar {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 8px;
}
.badge {
    background: rgba(255, 255, 255, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.35);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
}
"""

with gr.Blocks(title="Nhận Diện Bệnh & Độ Chín Trái Chuối") as demo:
    # Header
    gr.HTML("""
    <div class="main-header">
        <h1>🍌 Nhận Diện Bệnh & Đánh Giá Độ Chín Trái Chuối</h1>
        <p>Hệ thống AI tích hợp kép: Chẩn đoán 4 bệnh chuối (<strong>DenseNet-121</strong>) & Đánh giá 4 cấp độ chín (<strong>EfficientNet-B0</strong>)</p>
        <div class="badge-bar">
            <span class="badge">🎯 Bệnh chuối: DenseNet-121 (v24)</span>
            <span class="badge">🍌 Độ chín: EfficientNet-B0 (v18)</span>
            <span class="badge">⚡ F1-Score: 62.36%</span>
            <span class="badge">🚀 Dual AI Pipeline</span>
        </div>
    </div>
    """)

    # Main Tabs (Only Tab 1 and Tab 2)
    with gr.Tabs():
        # Tab 1: Image Upload
        with gr.TabItem("🖼️ Tải Ảnh Lên Phân Tích", id="tab_image"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(type="pil", label="Tải ảnh quả hoặc nải chuối")
                    btn_predict = gr.Button("🔍 Phân Tích Toàn Diện (Bệnh & Độ Chín)", variant="primary", size="lg")

                with gr.Column(scale=1):
                    img_output = gr.Image(type="numpy", label="Ảnh nhiệt Grad-CAM")
                    diagnosis_output = gr.Markdown("💡 *Tải ảnh lên và bấm 'Phân Tích Toàn Diện' để xem kết quả chẩn đoán bệnh và độ chín.*")

            btn_predict.click(
                fn=run_diagnosis,
                inputs=[img_input],
                outputs=[img_output, diagnosis_output]
            )

            # Sample Gallery
            if SAMPLE_IMAGES:
                gr.Markdown("### 📂 Thư Viện Ảnh Mẫu (Click để thử nhanh):")
                gr.Examples(
                    examples=SAMPLE_IMAGES,
                    inputs=img_input,
                    outputs=[img_output, diagnosis_output],
                    fn=run_diagnosis,
                    cache_examples=False
                )

        # Tab 2: Webcam Stream
        with gr.TabItem("📹 Quét Trực Tiếp Qua Camera", id="tab_cam"):
            with gr.Row():
                with gr.Column(scale=1):
                    cam_input = gr.Image(sources=["webcam"], type="pil", label="Bật Camera chụp trực tiếp quả/nải chuối")
                    btn_cam_predict = gr.Button("📸 Chụp & Nhận Diện Ngay", variant="primary", size="lg")
                with gr.Column(scale=1):
                    cam_img_out = gr.Image(type="numpy", label="Ảnh kết quả (Grad-CAM)")
                    cam_diag_out = gr.Markdown("💡 *Chụp ảnh qua camera để xem kết quả chẩn đoán.*")

            btn_cam_predict.click(
                fn=run_diagnosis,
                inputs=[cam_input],
                outputs=[cam_img_out, cam_diag_out]
            )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        theme=gr.themes.Soft(primary_hue="emerald"),
        css=CUSTOM_CSS
    )
