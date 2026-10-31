
import io

import pandas as pd
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/dbwlgn7/student_project/"
    "refs/heads/main/data/"
    "%EC%B2%AD%EC%86%8C%EB%85%84_AI%EC%9D%B4%EC%9A%A9%EB%B6%84%EB%A5%98_"
    "%EC%A0%95%EC%A0%9C%EB%8D%B0%EC%9D%B4%ED%84%B0.csv"
)

st.set_page_config(page_title="청소년 AI 이용 데이터 분석", page_icon="📊", layout="wide")
st.title("📊 청소년 AI 이용 데이터 기초통계 분석")
st.caption("GitHub CSV 데이터의 크기, 결측치, 기초통계 및 변수별 분포를 살펴봅니다.")


@st.cache_data(ttl=3600, show_spinner="CSV 데이터를 불러오는 중입니다...")
def load_from_url(url: str) -> pd.DataFrame:
    return pd.read_csv(url, encoding="utf-8-sig", low_memory=False)


def load_uploaded(file) -> pd.DataFrame:
    content = file.getvalue()
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(io.BytesIO(content), encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError("파일 인코딩을 확인해 주세요. UTF-8 또는 CP949 CSV를 지원합니다.")


def csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


with st.sidebar:
    st.header("데이터 설정")
    uploaded = st.file_uploader("다른 CSV 파일 사용 (선택)", type=["csv"])
    st.caption("파일을 선택하지 않으면 GitHub의 기본 데이터를 이용합니다.")
    if st.button("GitHub 데이터 새로 불러오기"):
        load_from_url.clear()
        st.rerun()

try:
    df = load_uploaded(uploaded) if uploaded is not None else load_from_url(DATA_URL)
except Exception as exc:
    st.error(f"데이터를 불러오지 못했습니다: {exc}")
    st.info("GitHub 파일이 공개 상태인지 확인하거나 왼쪽에서 CSV 파일을 직접 올려주세요.")
    st.stop()

if df.empty:
    st.warning("데이터에 행이 없습니다.")
    st.stop()

rows, cols = df.shape
missing_total = int(df.isna().sum().sum())
all_cells = rows * cols
missing_rate = missing_total / all_cells * 100 if all_cells else 0
duplicate_count = int(df.duplicated().sum())

st.subheader("1. 데이터 요약")
a, b, c, d, e = st.columns(5)
a.metric("행 개수", f"{rows:,}")
b.metric("열 개수", f"{cols:,}")
c.metric("결측치 개수", f"{missing_total:,}")
d.metric("결측치 비율", f"{missing_rate:.2f}%")
e.metric("중복 행", f"{duplicate_count:,}")

with st.expander("원본 데이터 미리보기", expanded=True):
    limit = st.slider("표시할 행 개수", 5, 100, 10, 5)
    st.dataframe(df.head(limit), use_container_width=True)
    st.download_button("전체 데이터 CSV 다운로드", csv_bytes(df), "original_data.csv", "text/csv")

st.subheader("2. 열별 자료형 및 결측치")
column_info = pd.DataFrame({
    "변수명": df.columns,
    "자료형": [str(t) for t in df.dtypes],
    "결측치 개수": df.isna().sum().values,
    "결측치 비율(%)": (df.isna().mean() * 100).round(2).values,
    "고유값 개수": df.nunique(dropna=True).values,
})
st.dataframe(column_info, use_container_width=True, hide_index=True)
st.download_button(
    "열별 정보 CSV 다운로드", csv_bytes(column_info), "column_info.csv", "text/csv"
)

missing = df.isna().sum().sort_values(ascending=False)
missing = missing[missing > 0]
if missing.empty:
    st.success("결측치가 없습니다.")
else:
    st.caption("결측치가 많은 상위 20개 변수")
    st.bar_chart(missing.head(20))
    st.info("결측치는 설문 문항의 비해당·건너뛰기 응답일 수도 있으므로 무조건 삭제하면 안 됩니다.")
