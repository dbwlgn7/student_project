"""청소년 디지털 매체 이용 특성 기반 생성형 AI 이진 분류 탐구 대시보드."""
from pathlib import Path
import io
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

st.set_page_config(page_title="청소년 AI 이용 분류 탐구", page_icon="🧭", layout="wide")
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "청소년_AI이용분류_ID포함13열.csv"
ORIGINAL_PATH = ROOT / "data" / "청소년_AI이용분류_정제데이터.csv"
ALTERNATIVE_PATH = ROOT / "data" / "청소년_AI이용분류_정제데이터(1).csv"
SELECTED_NUMBERS = [1,3,4,5,6,7,9,11,12,13,14,36,40]
# 원본에서 순서대로 가져온 46개 열 (코드북과 원본 CSV를 기준으로 작성)
ORIGINAL_NAMES = [
"응답자 ID", "학교 ID", "문1. 매체 이용 경험 - 1) TV방송", "문1. 매체 이용 경험 - 2) 온라인 동영상 제공서비스(OTT)",
"문1. 매체 이용 경험 - 3) 인터넷 개인방송 및 동영상 사이트", "문1. 매체 이용 경험 - 4) 숏폼 콘텐츠", "문1. 매체 이용 경험 - 5) 포털 미디어 플랫폼", "문1. 매체 이용 경험 - 6) 인터넷/모바일 메신저",
"문1. 매체 이용 경험 - 7) 인터넷 소셜네트워크서비스(SNS)", "문1. 매체 이용 경험 - 8) 화상회의 플랫폼", "문1. 매체 이용 경험 - 9) 메타버스", "문1. 매체 이용 경험 - 10) 생성형 AI",
"문1. 매체 이용 경험 - 11) 온라인/모바일 게임", "문1. 매체 이용 경험 - 12) 인터넷 만화(웹툰)", "문1. 매체 이용 경험 - 13) 인터넷 잡지 및 전자책(E-book)", "문1. 매체 이용 경험 - 14) 인터넷 신문",
"문1. 매체 이용 경험 - 15) 종이 신문", "문1. 매체 이용 경험 - 16) 종이책(참고서나 문제집 제외)", "문1-1. 자주 이용한 매체 (최대 3순위)", "(M)2 문1-1. 자주 이용한 매체 (최대 3순위)",
"(M)3 문1-1. 자주 이용한 매체 (최대 3순위)", "문5. 스마트폰 내 청소년 유해사이트 차단 프로그램이나 앱 설치 여부",
"문6. 경험 유무 - 1) 다른 사람의 아이디(ID) 사용", "문6. 경험 유무 - 2) 다른 사람의 주민등록번호 사용",
"문7. 개인정보 노출 - 1) 인터넷(SNS, 게임, 메신저 등)에서 모르는 사람이 내 개인정보를 알려달라고 했던 적이 있다",
"문7. 개인정보 노출 - 2) 인터넷(SNS, 게임, 메신저 등)에서 모르는 사람에게 내 개인정보를 알려 준 적이 있다",
"문7. 개인정보 노출 - 3) 가족이 나의 동의 없이 내 사진이나 내 개인정보를 SNS 등에 공개한 적이 있다",
"문7. 개인정보 노출 - 4) 내 친구가 나의 동의 없이 내 사진이나 내 개인정보를 SNS 등에 공개한 적이 있다",
"문7. 개인정보 노출 - 5) 모르는 사람이 나의 동의 없이 내 사진이나 내 개인정보를 SNS 등에 공개한 적이 있다",
"문10. 교육 경험 - 3) 건전한 인터넷/스마트폰 이용 교육", "문10. 교육 경험 - 4) 개인정보보호 교육",
"문10-1. 도움 정도 - 3) 건전한 인터넷/스마트폰 이용 교육", "문10-1. 도움 정도 - 4) 개인정보보호 교육",
"성별", "남녀공학", "학교급(기본)", "학교급(세부)", "학교급(초등)", "학교급(고등)", "학년별", "지역별", "지역규모",
"학교급(중고등 제외)", "학교급(초등4학년 제외)", "모집단 기준 가중치", "표본크기 기준 가중치"
]
assert len(ORIGINAL_NAMES) == 46

@st.cache_data(show_spinner=False)
def read_csv_from_path(path):
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

def get_data(upload):
    if upload is not None:
        return pd.read_csv(io.BytesIO(upload.getvalue()), encoding="utf-8-sig", low_memory=False), "직접 업로드한 13열 파일"
    if DATA_PATH.exists():
        return read_csv_from_path(str(DATA_PATH)), str(DATA_PATH)
    st.error("13열 CSV를 찾지 못했습니다. GitHub의 data 폴더에 파일을 업로드하거나 사이드바에서 CSV를 선택하세요.")
    st.code("data/청소년_AI이용분류_ID포함13열.csv")
    st.stop()

def get_original():
    for path in (ORIGINAL_PATH, ALTERNATIVE_PATH):
        if path.exists():
            return read_csv_from_path(str(path))
    return None

def fmt(s):
    return s.replace("문1. 매체 이용 경험 - ", "")

def clean_name(s):
    return s.split(") ", 1)[-1] if ") " in s else s

def retained_reason(i):
    if i == 1: return "응답자를 구분하고 결과를 추적하기 위해 남겼다. 다만 모델 입력에서는 제외한다."
    if i == 12: return "생성형 AI 이용 여부는 이번 탐구에서 모델이 예측해야 하는 정답(Y)이므로 반드시 남겼다."
    if i in [36,40]: return "학교급·학년에 따라 매체 이용 특성이 달라질 가능성이 있어 보조 입력 변수로 남겼다."
    return "디지털 매체 이용 경험을 직접 나타내므로 생성형 AI 이용 여부를 분류할 입력 변수(X)로 남겼다."

def excluded_reason(i):
    if i == 2: return "학교 식별용 번호로 디지털 매체 이용 특성이 아니기 때문에 모델 입력용 데이터에서 제외했다. 학교별 검증을 진행한다면 별도로 보존할 수 있다."
    if i in [8,10,15,16]: return "디지털 매체 이용과 관련이 있지만, 이번에는 일부 매체를 중심으로 입력 변수의 범위를 제한하여 제외했다. 예측력이 없다고 확인된 것은 아니다."
    if i in [17,18]: return "종이 매체 이용 경험은 이번 탐구에서 초점을 맞춘 디지털 매체 이용 특성과 직접적인 관련성이 낮아 제외했다."
    if i in [19,20,21]: return "가장 자주 이용하는 매체 응답에 생성형 AI가 포함될 수 있어 정답을 간접적으로 알려 주는 데이터 누수 위험을 줄이고자 제외했다."
    if i == 22: return "유해사이트 차단 앱 설치 여부는 매체 이용 경험 자체보다 보호 환경을 나타내므로 탐구 범위를 좁히기 위해 제외했다."
    if i in [23,24]: return "다른 사람의 계정·신분정보 사용 경험은 디지털 매체의 종류별 이용 특성을 측정하는 문항이 아니므로 제외했다."
    if i in [25,26,27,28,29]: return "개인정보 노출·제공 경험을 묻는 문항으로, 매체별 이용 여부를 중심으로 분류하려는 이번 분석 범위에서 제외했다."
    if i in [30,31]: return "교육을 받은 경험에 관한 문항으로 디지털 매체 이용 특성 자체를 설명하는 변수는 아니므로 제외했다."
    if i in [32,33]: return "교육 도움 정도는 매체 이용 특성과 직접적인 관련성이 낮고, 교육 미경험자에게 비해당 결측치가 있어 제외했다."
    if i == 34: return "성별은 매체 이용 경험이 아닌 인구통계적 특성이므로 이번 기본 모델에서는 제외했다."
    if i == 35: return "남녀공학 여부는 학교 환경에 관한 정보로, 매체 이용 특성에 집중하기 위해 제외했다."
    if i == 37: return "학교급(기본)을 이미 유지하므로 학교급을 더 세분화한 정보는 기본 모델에서 제외했다."
    if i in [38,39,43,44]: return "학교급(기본)과 일부 정보가 겹치며 특정 학교급에만 적용되는 값이 존재하므로 제외했다."
    if i in [41,42]: return "거주 지역에 관한 정보로, 디지털 매체 이용 특성 자체를 중심으로 분석하기 위해 제외했다."
    if i in [45,46]: return "가중치는 설문 추정에 사용하는 값이지 응답자의 매체 이용 특성이 아니므로 분류 입력에서 제외했다. 대표성 분석이 필요하면 원본에서 따로 활용할 수 있다."
    return "이번 분류 탐구의 입력 변수 범위와 직접적인 관련성이 낮아 제외했다."

st.sidebar.title("🧭 탐구 메뉴")
menu = st.sidebar.radio("이동할 메뉴", ["① 탐구 소개", "② 전처리 및 근거", "③ 데이터 분포 시각화", "④ 변수 간 관계", "⑤ 분류 모델 평가", "⑥ 탐구 결론"])
upload = st.sidebar.file_uploader("13열 CSV 직접 올리기 (선택)", type="csv")
df, source = get_data(upload)
expected = [ORIGINAL_NAMES[i-1] for i in SELECTED_NUMBERS]
