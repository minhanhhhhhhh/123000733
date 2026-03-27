"""
Lab NLP: Tự động gán nhãn & phân tách từ cho bình luận Facebook tiếng Việt
===========================================================================
File bài tập — Sinh viên hoàn thành các phần TODO bên dưới.

Hướng dẫn:
- Tìm tất cả dòng có "# TODO" và viết code thay thế
- Chạy app bằng: streamlit run app_auto_label_todo.py
- Tham khảo tài liệu underthesea: https://github.com/undertheseanlp/underthesea
"""

import re
import streamlit as st
import pandas as pd
# TODO 1: Import hai hàm word_tokenize và sentiment từ thư viện underthesea
from underthesea import word_tokenize, sentiment

# ============================================================================
# HÀM PHÁT HIỆN SPAM
# ============================================================================
def detect_spam(text: str) -> bool:
    """
    Phát hiện comment spam dựa trên các đặc điểm phổ biến.
    Trả về True nếu là spam, False nếu không.

    Gợi ý kiểm tra:
    - Chứa link (http, www, .com, .vn, bit.ly ...)
    - Chứa số điện thoại (chuỗi 10-11 chữ số)
    - Chứa từ khóa quảng cáo (inbox, giá rẻ, miễn phí, zalo ...)
    - Lặp ký tự bất thường (aaaaaaa, !!!!!!)
    """
    if not text:
        return False
    
    t = text.lower()

    # TODO 2: Dùng re.search để kiểm tra xem text có chứa link không
    if re.search(r"https?://|www\.|\.com|\.vn|\.net|bit\.ly", t):
        return True

    # TODO 3: Dùng re.search để kiểm tra số điện thoại
    if re.search(r"(\d[\d\.\-]{8,}\d)", t):
        return True

    # TODO 4: Tạo danh sách từ khóa spam và kiểm tra
    spam_keywords = ["inbox", "liên hệ", "zalo", "giá rẻ", "miễn phí", "khuyến mãi", 
                     "sale", "đặt hàng", "ship cod", "hotline", "090", "091", "098"]
    if any(kw in t for kw in spam_keywords):
        return True

    # TODO 5: Dùng re.search để phát hiện lặp ký tự bất thường
    if re.search(r"(.)\1{5,}", t):
        return True

    return False

# ============================================================================
# GIAO DIỆN STREAMLIT
# ============================================================================
st.set_page_config(
    page_title="Lab NLP - Tự động gán nhãn",
    layout="wide"
)

st.title("Lab NLP: Tự động gán nhãn & phân tách từ cho bình luận Facebook tiếng Việt")

st.markdown(
    "Upload file CSV (cột `id`, `text`) — ứng dụng sẽ dùng **underthesea** để:\n"
    "- **Phân tách từ** (word segmentation)\n"
    "- **Gán nhãn cảm xúc** tự động (positive / negative / neutral)"
)

# TODO 6: Tạo file uploader cho file CSV
uploaded_file = st.file_uploader("Chọn file CSV chứa bình luận", type=["csv"])

if uploaded_file is None:
    st.info("Vui lòng upload file CSV để bắt đầu.")
    st.stop()

# TODO 7: Đọc file CSV bằng pandas
df = pd.read_csv(uploaded_file)

# TODO 8: Kiểm tra file CSV có cột "id" và "text" không
if not {"id", "text"}.issubset(df.columns):
    st.error("File CSV phải có ít nhất hai cột: `id` và `text`!")
    st.stop()

st.success(f"Đã load {len(df)} dòng. Đang xử lý...")

progress = st.progress(0)

tokenized_list = []
sentiment_list = []

for i, row in df.iterrows():
    text = str(row["text"]).strip()

    # TODO 9: Dùng word_tokenize để phân tách từ (format="text")
    tokens = word_tokenize(text, format="text")
    tokenized_list.append(tokens)

    # TODO 10: Dùng hàm sentiment() để gán nhãn cảm xúc
    label = sentiment(text)
    sentiment_list.append(label)

    progress.progress((i + 1) / len(df))

# ============================================================================
# GÁN KẾT QUẢ VÀO DATAFRAME
# ============================================================================
df["tokenized"] = tokenized_list
df["sentiment_label"] = sentiment_list

# TODO 11: Dùng hàm detect_spam để tạo cột "spam" (True/False)
df["spam"] = df["text"].apply(lambda x: detect_spam(str(x)))

# TODO 12: Tạo cột spam_label ("spam" / "không spam") từ cột spam
df["spam_label"] = df["spam"].map({True: "spam", False: "không spam"})

# TODO 13: Tạo cột spam_label_vn ("Spam" / "Không spam") từ cột spam
df["spam_label_vn"] = df["spam"].map({True: "Spam", False: "Không spam"})

# TODO 14: Tạo dict map sentiment tiếng Anh -> tiếng Việt và tạo cột sentiment_label_vn
sentiment_vn_map = {
    "positive": "Tích cực", 
    "negative": "Tiêu cực", 
    "neutral": "Trung lập"
}
df["sentiment_label_vn"] = df["sentiment_label"].map(sentiment_vn_map).fillna(df["sentiment_label"])

progress.empty()
st.success("Hoàn tất xử lý!")

# ============================================================================
# HIỂN THỊ THỐNG KÊ & KẾT QUẢ
# ============================================================================

# TODO 15: Hiển thị biểu đồ thống kê cảm xúc và spam
st.subheader("Thống kê kết quả")
col1, col2 = st.columns(2)

with col1:
    st.bar_chart(df["sentiment_label_vn"].value_counts())

with col2:
    st.bar_chart(df["spam_label_vn"].value_counts())

# Hiển thị bảng kết quả
st.subheader("Kết quả chi tiết")
display_cols = ["id", "text", "tokenized", "spam_label", "spam_label_vn", "sentiment_label", "sentiment_label_vn"]
st.dataframe(df[display_cols], use_container_width=True)

# TODO 16: Xuất CSV và tạo nút download
csv_data = df[display_cols].to_csv(index=False, encoding="utf-8")
st.download_button(
    label="⬇ Tải về file kết quả",
    data=csv_data,
    file_name="auto_labels_output.csv",
    mime="text/csv"
)