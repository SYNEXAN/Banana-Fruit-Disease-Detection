"""
Banana Disease & Ripeness Classification - Knowledge Base & Pathogen Database
Dual Models:
  1. Banana Disease: DenseNet-121 (Version 24 from Kaggle)
  2. Banana Ripeness: EfficientNet-B0 (Version 18 from Kaggle)
"""

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TAC_NHAN_PATH = BASE_DIR / "TacNhan.json"

# ============================================================
# 1. Banana Disease Classes (4 Classes)
# ============================================================
BANANA_CLASSES = [
    "black_pitting_banana_rust",
    "crown_rot",
    "fungal_disease",
    "healthy_banana"
]

BANANA_VN_MAP = {
    "black_pitting_banana_rust": "🍌 Chuối - Bệnh rỉ sắt / đốm gỉ quả chuối",
    "crown_rot": "🍌 Chuối - Bệnh thối cổ / thối cuống nải chuối",
    "fungal_disease": "🍌 Chuối - Bệnh thán thư / nấm đốm quả chuối",
    "healthy_banana": "🍌 Chuối - Cây quả khỏe mạnh"
}

DISEASE_INFO = {
    "black_pitting_banana_rust": {
        "status": "warning",
        "fruit": "Quả Chuối",
        "severity": "Trung bình",
        "description": "Vết đốm gỉ màu nâu đen lấm chấm trên vỏ quả chuối, làm giảm mẫu mã thương phẩm.",
        "treatment": "Vệ sinh vườn chuối thông thoáng, bao buồng chuối sớm và phòng trừ bọ trĩ kịp thời."
    },
    "crown_rot": {
        "status": "danger",
        "fruit": "Quả Chuối",
        "severity": "Nguy hiểm",
        "description": "Thối nhũn và đen cuống nải chuối sau thu hoạch, lan nhanh làm rụng quả khỏi nải.",
        "treatment": "Cắt tỉa dao sắc khử trùng, xử lý nước ấm hoặc ngâm dung dịch sinh học trước khi đóng thùng xuất khẩu."
    },
    "fungal_disease": {
        "status": "danger",
        "fruit": "Quả Chuối",
        "severity": "Nguy hiểm",
        "description": "Vết bệnh tròn lõm màu nâu sẫm trên vỏ, gây thối hỏng ruột quả khi chín.",
        "treatment": "Cắt tỉa lá bệnh, bao buồng sớm, tránh để vỏ quả bị trầy xước khi thu hoạch."
    },
    "healthy_banana": {
        "status": "healthy",
        "fruit": "Quả Chuối",
        "severity": "An toàn",
        "description": "Quả chuối phát triển tốt, màu sắc vỏ tươi sáng, không có vết nấm hay thối hỏng.",
        "treatment": "Duy trì chế độ dinh dưỡng cân đối và tưới tiêu hợp lý cho vườn chuối."
    }
}

# ============================================================
# 2. Banana Ripeness Classes (4 Levels - Version 18)
# ============================================================
RIPENESS_CLASSES = [
    "Unripe",
    "Half_Ripe",
    "Ripe",
    "Overripe"
]

RIPENESS_INFO = {
    "Unripe": {
        "vn_name": "Chuối xanh (Chưa chín)",
        "icon": "🟢",
        "description": "Vỏ xanh đậm, thịt quả cứng, vị chát, giàu tinh bột kháng (Resistant Starch).",
        "usage": "Thích hợp chế biến món luộc, nấu ốc chuối đậu hoặc tiếp tục ủ chín tự nhiên."
    },
    "Half_Ripe": {
        "vn_name": "Chuối ương (Nửa chín / Mới chuyển vàng)",
        "icon": "🟡",
        "description": "Vỏ ngả vàng xanh, vị ngọt dịu thanh nhẹ, giòn, chỉ số đường huyết (GI) thấp.",
        "usage": "Lý tưởng cho người ăn kiêng, tiểu đường hoặc người cần kiểm soát đường huyết."
    },
    "Ripe": {
        "vn_name": "Chuối chín vừa (Độ chín lý tưởng)",
        "icon": "🍌",
        "description": "Vỏ vàng ươm hoàn hảo, thịt mềm dẻo, ngọt thơm đậm đà, giàu Kali và Vitamin B6.",
        "usage": "Rất tốt để ăn trực tiếp, bổ sung năng lượng nhanh sau vận động hoặc ăn sáng."
    },
    "Overripe": {
        "vn_name": "Chuối quá chín (Chín trứng cuốc / Nẫu)",
        "icon": "🟤",
        "description": "Vỏ xuất hiện nhiều đốm chấm nâu đen, hàm lượng đường tự nhiên và chất chống oxy hóa cao nhất.",
        "usage": "Mùi thơm rất đậm, thích hợp nhất để làm bánh chuối nướng, sinh tố, chè chuối hoặc kem chuối."
    }
}


def _load_tac_nhan():
    if not TAC_NHAN_PATH.exists():
        return {}
    try:
        with open(TAC_NHAN_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[classes.py] Error loading TacNhan.json: {e}")
        return {}


TAC_NHAN_DATA = _load_tac_nhan()


def get_pathogen_info(class_name: str) -> dict:
    """Retrieve pathogen details for a banana disease class."""
    if not TAC_NHAN_DATA:
        return {}
    return TAC_NHAN_DATA.get(class_name, {})
