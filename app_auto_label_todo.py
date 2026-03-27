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
# HIỂN THỊ THỐNG KÊ & KẾT QUẢ   ←   TODO 15 (ĐÃ SỬA)
# ============================================================================

st.subheader("📊 Thống kê kết quả")

col1, col2 = st.columns(2)

# ==================== BIỂU ĐỒ CẢM XÚC ====================
with col1:
    st.markdown("### 📈 Phân bố cảm xúc")

    sentiment_counts = df["sentiment_label_vn"].value_counts().reset_index()
    sentiment_counts.columns = ["Cảm xúc", "Số lượng"]

    emoji_map = {
        "Tích cực": "😊",
        "Tiêu cực": "😠",
        "Trung lập": "😐"
    }

    sentiment_counts["Cảm xúc"] = sentiment_counts["Cảm xúc"].map(
        lambda x: f"{emoji_map.get(x, '❓')} {x}"
    )

    import altair as alt

    sentiment_chart = alt.Chart(sentiment_counts).mark_bar().encode(
        x=alt.X("Cảm xúc:N", sort=None, axis=alt.Axis(labelFontSize=14, labelAngle=0)),
        y=alt.Y("Số lượng:Q", title="Số lượng bình luận"),
        color=alt.Color("Cảm xúc:N", 
                       scale=alt.Scale(range=["#9E9E9E","#4CAF50", "#f44336" ]), 
                       legend=None),
        tooltip=["Cảm xúc", "Số lượng"]
    ).properties(
        title="Phân bố cảm xúc",
        height=380
    )

    st.altair_chart(sentiment_chart, use_container_width=True)

# ==================== BIỂU ĐỒ SPAM (ĐÃ SỬA - thêm icon) ====================
with col2:
    st.markdown("### 🚫 Phân bố Spam / Không spam")

    spam_counts = df["spam_label_vn"].value_counts().reset_index()
    spam_counts.columns = ["Loại", "Số lượng"]

    spam_emoji_map = {
        "Spam": "🚫",
        "Không spam": "✅"
    }

    spam_counts["Loại"] = spam_counts["Loại"].map(
        lambda x: f"{spam_emoji_map.get(x, '❓')} {x}"
    )

    spam_chart = alt.Chart(spam_counts).mark_bar().encode(
        x=alt.X("Loại:N", sort=None, axis=alt.Axis(labelFontSize=14, labelAngle=0)),
        y=alt.Y("Số lượng:Q", title="Số lượng bình luận"),
        color=alt.Color("Loại:N", 
                       scale=alt.Scale(range=["#4CAF50","#f44336"]), 
                       legend=None),
        tooltip=["Loại", "Số lượng"]
    ).properties(
        title="Phân bố Spam",
        height=380
    )

    st.altair_chart(spam_chart, use_container_width=True)


# ============================================================================
# HIỂN THỊ BẢNG KẾT QUẢ & DOWNLOAD
# ============================================================================

# TODO 16: Định nghĩa display_cols trước khi sử dụng
display_cols = ["id", "text", "tokenized", "spam_label", "spam_label_vn", 
                "sentiment_label", "sentiment_label_vn"]

st.subheader("📋 Kết quả chi tiết")
st.dataframe(df[display_cols], use_container_width=True)

# TODO 16: Xuất CSV và tạo nút download
output_df = df[display_cols].copy()
output_df = output_df.astype(str)   # Đảm bảo mọi cột đều là text

csv_data = output_df.to_csv(
    index=False, 
    encoding="utf-8-sig"   # Quan trọng nhất: utf-8-sig (có BOM) để Excel đọc đúng
)

st.download_button(
    label="⬇ Tải về file kết quả (CSV - hỗ trợ tiếng Việt)",
    data=csv_data,
    file_name="auto_labels_output.csv",
    mime="text/csv",
    help="Mở bằng Excel: File → Import → chọn UTF-8 hoặc mở trực tiếp thường sẽ đúng font"
)