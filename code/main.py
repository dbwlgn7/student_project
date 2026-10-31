"""청소년 디지털 매체 이용 특성 기반 생성형 AI 이용 여부 분류 대시보드."""
from pathlib import Path
import io
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(page_title="청소년 AI 이용 분류 탐구", page_icon="🧭", layout="wide")
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = "청소년_AI이용분류_ID포함13열.csv"
ORIG_FILE = "청소년_AI이용분류_정제데이터.csv"
KEEP = [1,3,4,5,6,7,9,11,12,13,14,36,40]

@st.cache_data(ttl=3600)
def read_file(path):
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

def load_file(upload, filename):
    if upload is not None:
        return pd.read_csv(io.BytesIO(upload.getvalue()), encoding="utf-8-sig", low_memory=False)
    path = ROOT / "data" / filename
    if path.exists():
        return read_file(str(path))
    return None

def show_chart(chart):
    st.altair_chart(chart, use_container_width=True)

def selected_columns(df):
    cols=list(df.columns)
    target = next((c for c in cols if "생성형 AI" in c and "이용 경험" in c), None)
    ident = next((c for c in cols if "응답자 ID" in c), cols[0])
    grade = next((c for c in cols if "학교급(기본)" in c), None)
    year = next((c for c in cols if "학년별" in c), None)
    media = [c for c in cols if c not in (target,ident,grade,year)]
    return target,ident,grade,year,media

def short(c):
    if " - " in c: return c.split(" - ",1)[1].split(") ",1)[-1]
    return c

def encode_binary(df, media):
    # 원본 데이터는 수정하지 않고 분석 복사본만 변환
    out=df[media].copy()
    for c in media:
        out[c]=pd.to_numeric(out[c],errors="coerce").map({1:1,2:0})
    return out

def source_reason(i):
    if i==1: return "응답자를 구분하기 위해 유지했다. 머신러닝 입력에는 사용하지 않는다."
    if i==12: return "생성형 AI 이용 여부는 분류 모델이 맞혀야 할 정답(Y)이므로 유지했다."
    if i in (36,40): return "학교급·학년별 차이도 살펴보기 위한 보조 입력 변수로 유지했다."
    if i in KEEP: return "디지털 매체 이용 경험을 직접 나타내므로 모델 입력(X)에 활용하기 위해 유지했다."
    if i==2: return "학교를 구분하는 번호로, 디지털 매체 이용 특성이 아니어서 분석용 열에서 제외했다."
    if i in (8,10,15,16): return "디지털 매체 이용과 관련이 있지만 이번 모델의 매체 종류를 일부로 한정하여 제외했다. 예측력이 없다고 검증된 것은 아니다."
    if i in (17,18): return "종이 매체 이용 경험으로, 디지털 매체 이용 특성에 집중하려는 탐구 범위에서 제외했다."
    if i in (19,20,21): return "자주 이용하는 매체 목록에 생성형 AI가 직접 포함될 수 있어 정답 누수 위험을 줄이기 위해 제외했다."
    if i==22: return "보호 프로그램 설치 여부는 이용하는 매체의 종류 자체가 아니므로 제외했다."
    if i in (23,24): return "타인의 정보 사용 경험으로, 디지털 매체의 종류별 이용 여부를 직접 나타내지 않아 제외했다."
    if i in (25,26,27,28,29): return "개인정보 노출·제공 경험으로, 매체별 이용 여부에 초점을 맞춘 분석 범위에서 제외했다."
    if i in (30,31): return "교육 경험을 나타내며 이용하는 디지털 매체의 종류를 나타내지는 않으므로 제외했다."
    if i in (32,33): return "교육 도움 정도는 매체 이용 특성이 아니고 교육 미경험자의 비해당 결측치가 있어 제외했다."
    if i==34: return "성별은 매체 이용 여부가 아닌 인구통계적 정보로, 이번 분석에서는 제외했다."
    if i==35: return "학교 환경 특성으로 디지털 매체 이용 자체가 아니어서 제외했다."
    if i==37: return "기본 학교급을 유지해 유사한 학교급 세부 변수는 제외했다."
    if i in (38,39,43,44): return "기본 학교급과 정보가 일부 겹치며 특정 학교급에 비해당하는 결측치가 있으므로 제외했다."
    if i in (41,42): return "지역 정보는 매체 이용 자체를 나타내지 않아 이번 기본 모델에서 제외했다."
    if i in (45,46): return "조사 가중치는 이용 경험이 아닌 설문 추정 정보이므로 분류 입력 변수에서 제외했다."
    return "디지털 매체 이용 특성 중심으로 탐구 범위를 정하여 제외했다."

st.sidebar.title("🧭 탐구 메뉴")
menu=st.sidebar.radio("이동할 메뉴",[
    "① 탐구 소개","② 전처리 및 근거","③ 데이터 분포 시각화",
    "④ 변수 간 관계","⑤ 분류 모델 평가","⑥ 탐구 결론"
])
upload=st.sidebar.file_uploader("13열 CSV 직접 올리기 (선택)",type="csv")
df=load_file(upload,DATA_FILE)
if df is None:
    st.error(f"data/{DATA_FILE} 파일을 찾지 못했어. GitHub의 data 폴더를 확인하거나 왼쪽에서 업로드해 줘.")
    st.stop()
target,ident,grade,year,media=selected_columns(df)
if target is None or len(media)<1:
    st.error("생성형 AI 타깃 또는 매체 이용 변수를 찾지 못했습니다. 13열 CSV의 열 이름을 확인해 주세요.")
    st.dataframe(pd.DataFrame({"열 이름":df.columns}))
    st.stop()
bad={c: sorted(set(pd.to_numeric(df[c],errors="coerce").dropna())-{1,2}) for c in media+[target]}
bad={k:v for k,v in bad.items() if v}
xmedia=encode_binary(df,media)
ytarget=pd.to_numeric(df[target],errors="coerce").map({1:1,2:0})
st.title("🧭 청소년 디지털 매체 이용과 생성형 AI 분류 탐구")
st.caption("청소년의 디지털 매체 이용 특성으로 생성형 AI 이용 여부를 분류할 수 있을까? · CSV 원본 값은 변경하지 않습니다.")

if menu.startswith("①"):
    st.header("탐구 소개")
    st.write("**탐구 질문:** 다른 디지털 매체 이용 경험과 학교급·학년을 바탕으로 청소년의 생성형 AI 이용 여부를 예측할 수 있을까?")
