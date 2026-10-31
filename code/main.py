
from pathlib import Path
import re

import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)


# ==========================================
# 1. 기본 설정
# ==========================================

st.set_page_config(
    page_title="청소년 생성형 AI 이용 분류 탐구",
    page_icon="📊",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

SELECTED_FILE = DATA_DIR / "청소년_AI이용분류_ID포함13열.csv"
ORIGINAL_FILE = DATA_DIR / "청소년_AI이용분류_정제데이터.csv"

KEEP = [1, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 36, 40]

MEDIA_NAMES = [
    "TV방송", "OTT", "인터넷 개인방송", "숏폼",
    "포털 플랫폼", "SNS", "메타버스",
    "온라인 게임", "웹툰"
]

GRADE_LABELS = {
    1: "초4", 2: "초5", 3: "초6",
    4: "중1", 5: "중2", 6: "중3",
    7: "고1", 8: "고2", 9: "고3"
}

SCHOOL_LABELS = {
    1: "초등학교",
    2: "중학교",
    3: "고등학교"
}

MODEL_NAMES = [
    "로지스틱 회귀",
    "의사결정트리",
    "랜덤 포레스트"
]


# ==========================================
# 2. 공통 함수
# ==========================================

@st.cache_data
def load_csv(path):
    return pd.read_csv(
        path, encoding="utf-8-sig", low_memory=False
    )


def show_chart(chart):
    st.altair_chart(chart, width="stretch")


def get_reason(n):
    if n == 1:
        return "응답자 구분과 중복 검사용으로 유지하며 모델 입력에서 제외한다."
    if n == 12:
        return "생성형 AI 이용 여부는 예측 대상인 타깃 Y이므로 유지한다."
    if n in [36, 40]:
        return "학교급과 학년에 따른 차이를 분석하기 위해 유지한다."
    if n in KEEP:
        return "디지털 매체 이용 경험을 나타내는 입력 X로 유지한다."
    if n == 2:
        return "학교 식별 번호이므로 제외한다."
    if n in [8, 10, 15, 16]:
        return "관련된 디지털 매체지만 탐구 범위를 제한하기 위해 제외한다."
    if n in [17, 18]:
        return "종이 매체이므로 디지털 매체 중심 분석에서 제외한다."
    if n in [19, 20, 21]:
        return "생성형 AI가 응답에 포함될 수 있어 타깃 정보 누출을 방지하기 위해 제외한다."
    if 22 <= n <= 29:
        return "온라인 안전 및 개인정보 보호 관련 경험으로 제외한다."
    if 30 <= n <= 33:
        return "교육 경험과 교육 효과 관련 변수이므로 제외한다."
    if n in [34, 35]:
        return "개인 및 학교 배경 특성으로 탐구 범위에서 제외한다."
    if n in [37, 38, 39, 43, 44]:
        return "기본 학교급·학년 정보를 사용하기 위해 추가 정보를 제외한다."
    if n in [41, 42]:
        return "지역 특성으로 이번 분석 범위에서 제외한다."
    if n in [45, 46]:
        return "조사 가중치이므로 이번 분류 모델의 입력에서 제외한다."
    return "탐구 범위를 고려해 제외한다."


def make_models():
    return {
        "로지스틱 회귀": LogisticRegression(
            max_iter=2000,
            random_state=42
        ),
        "의사결정트리": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=30,
            random_state=42
        ),
        "랜덤 포레스트": RandomForestClassifier(
            n_estimators=120,
            max_depth=10,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        )
    }


def metrics(y_true, pred, prob):
    return {
        "정확도": accuracy_score(y_true, pred),
        "정밀도": precision_score(
            y_true, pred, zero_division=0
        ),
        "재현율": recall_score(
            y_true, pred, zero_division=0
        ),
        "F1": f1_score(
            y_true, pred, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_true, prob)
    }


def make_pipeline(algorithm, columns):
    # 각 매체는 이진형이므로 중복된 두 범주 중 하나를 제외
    encoder = OneHotEncoder(
        drop="if_binary",
        handle_unknown="ignore"
    )

    preprocessing = ColumnTransformer([
        ("categorical", encoder, list(columns))
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", algorithm)
    ])


def readable_feature(raw):
    """
    categorical__SNS_1 -> SNS 이용함
    categorical__학교급_고등학교 -> 학교급: 고등학교
    categorical__학년_초4 -> 학년: 초4
    """

    name = str(raw).replace("categorical__", "")

    for media in MEDIA_NAMES:
        if name == media + "_1":
            return media + " 이용함"
        if name == media + "_0":
            return media + " 이용 안함"

    if name.startswith("학교급_"):
        return "학교급: " + name[len("학교급_"):]

    if name.startswith("학년_"):
        return "학년: " + name[len("학년_"):]

    return name.replace("_", " ")


def original_group(raw):
    """인코딩된 항목을 원래 변수 단위로 묶는다."""
    name = str(raw).replace("categorical__", "")

    if name.startswith("학교급_"):
        return "학교급"

    if name.startswith("학년_"):
        return "학년"

    for media in MEDIA_NAMES:
        if name.startswith(media + "_"):
            return media

    return name


def feature_table(pipe, values, value_name):
    names = pipe.named_steps[
        "preprocessing"
    ].get_feature_names_out()

    result = pd.DataFrame({
        "원래 변수": [original_group(x) for x in names],
        "인코딩 항목": [readable_feature(x) for x in names],
        value_name: values
    })

    return result


@st.cache_resource(show_spinner="분류 모델을 학습하고 있습니다.")
def train_models(X, y):

    # 학습 60%, 검증 20%, 최종 평가 20%
    X_build, X_test, y_build, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_build, y_build,
        test_size=0.25,
        random_state=42,
        stratify=y_build
    )

    results = {}

    for name, algorithm in make_models().items():
        pipe = make_pipeline(algorithm, X.columns)
        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_valid)
        prob = pipe.predict_proba(X_valid)[:, 1]

        results[name] = {
            "pipeline": pipe,
            "pred": pred,
            "prob": prob,
            "scores": metrics(y_valid, pred, prob)
        }

    # 검증 데이터의 F1을 주된 선정 기준으로 사용
    # 동점이면 ROC-AUC, 그다음 정확도 비교
    best_name = max(
        results,
        key=lambda name: (
            results[name]["scores"]["F1"],
            results[name]["scores"]["ROC-AUC"],
            results[name]["scores"]["정확도"]
        )
    )

    final_pipe = make_pipeline(
        make_models()[best_name],
        X.columns
    )

    # 선정한 모델만 학습+검증 데이터로 재학습
    final_pipe.fit(X_build, y_build)

    test_pred = final_pipe.predict(X_test)
    test_prob = final_pipe.predict_proba(X_test)[:, 1]

    final_scores = metrics(
        y_test, test_pred, test_prob
    )

    return {
        "results": results,
        "best_name": best_name,
        "final_pipe": final_pipe,
        "final_scores": final_scores,
        "valid_y": y_valid,
        "test_y": y_test,
        "test_pred": test_pred,
        "counts": {
            "학습": len(X_train),
            "검증": len(X_valid),
            "평가": len(X_test)
        }
    }


def scores_dataframe(results):
    return pd.DataFrame([
        {"모델": name, **item["scores"]}
        for name, item in results.items()
    ])


# ==========================================
# 3. 메뉴
# ==========================================

st.sidebar.title("🧭 탐구 메뉴")

menu = st.sidebar.radio(
    "이동할 메뉴",
    [
        "① 탐구 소개",
        "② 전처리 및 근거",
        "③ 데이터 분포 시각화",
        "④ 입력 변수와 타깃의 관계",
        "⑤ 분류 모델 학습",
        "⑥ 모델 성능 비교 및 선정"
    ]
)

uploaded = st.sidebar.file_uploader(
    "13열 CSV 업로드 (선택)",
    type=["csv"]
)


# ==========================================
# 4. 데이터 불러오기
# ==========================================

if uploaded is not None:
    df = pd.read_csv(uploaded, encoding="utf-8-sig")

elif SELECTED_FILE.exists():
    df = load_csv(str(SELECTED_FILE))

else:
    st.error("13열 CSV 파일을 찾을 수 없습니다.")
    st.stop()

if df.shape[1] != 13:
    st.error("13개 열로 구성된 CSV가 필요합니다.")
    st.stop()

id_col = df.columns[0]
media_cols = (
    list(df.columns[1:8])
    + list(df.columns[9:11])
)

target_col = df.columns[8]
school_col = df.columns[11]
grade_col = df.columns[12]

if (
    "응답자 ID" not in str(id_col)
    or "생성형 AI" not in str(target_col)
    or "학교급" not in str(school_col)
    or "학년별" not in str(grade_col)
):
    st.error("CSV의 열 이름 또는 순서가 예상한 구성과 다릅니다.")
    st.stop()

X_media = pd.DataFrame(index=df.index)

for name, col in zip(MEDIA_NAMES, media_cols):
    X_media[name] = pd.to_numeric(
        df[col], errors="coerce"
    ).map({1: 1, 2: 0})

Y = pd.to_numeric(
    df[target_col], errors="coerce"
).map({1: 1, 2: 0})

school_codes = pd.to_numeric(
    df[school_col], errors="coerce"
)

grade_codes = pd.to_numeric(
    df[grade_col], errors="coerce"
)

school_labels = school_codes.map(SCHOOL_LABELS)
grade_labels = grade_codes.map(GRADE_LABELS)

if (
    X_media.isna().any().any()
    or Y.isna().any()
    or school_labels.isna().any()
    or grade_labels.isna().any()
):
    st.error(
        "결측치 또는 예상하지 못한 응답 코드가 있습니다. "
        "CSV와 코드북을 확인해 주세요."
    )
    st.stop()

X = X_media.copy()
X["학교급"] = school_labels
X["학년"] = grade_labels

st.title("📊 청소년 생성형 AI 이용 분류 탐구")

st.caption(
    "청소년의 디지털 매체 이용 특성으로 "
    "생성형 AI 이용 여부를 분류할 수 있을까?"
)


# ==========================================
# 메뉴 1. 탐구 소개
# ==========================================

if menu == "① 탐구 소개":

    st.header("탐구 소개")

    a, b, c = st.columns(3)
    a.metric("응답자", f"{len(df):,}명")
    b.metric("선택 변수", 13)
    c.metric("모델 입력 변수", 11)

    st.subheader("왜 분류 모델을 사용하는가?")

    methods = pd.DataFrame([
        ["회귀", "연속적인 숫자를 예측", "선택하지 않음"],
        ["분류", "이용함 / 이용 안함 예측", "선택"],
        ["군집", "정답 없이 비슷한 집단 구성", "선택하지 않음"]
    ], columns=["방법", "목적", "탐구 적용"])

    st.dataframe(methods, hide_index=True, width="stretch")

    st.success(
        "생성형 AI 이용 여부는 두 범주 중 하나를 "
        "예측하는 문제이므로 이진 분류가 적합합니다."
    )

    st.subheader("변수의 역할과 데이터 유형")

    types = pd.DataFrame([
        ["응답자 ID", "식별자", "중복 검사", "제외"],
        ["9개 디지털 매체", "이진 범주형", "입력 X", "포함"],
        ["학교급", "순서형 범주", "입력 X", "포함"],
        ["학년", "순서형 범주", "입력 X", "포함"],
        ["AI 이용 여부", "이진 범주형", "타깃 Y", "입력에서 제외"]
    ], columns=["속성", "데이터 유형", "역할", "모델 입력"])

    st.dataframe(types, hide_index=True, width="stretch")

    st.subheader("사용할 분류 모델")

    for name, description in [
        ("로지스틱 회귀", "확률을 계산해 분류하는 기본 모델"),
        ("의사결정트리", "조건에 따라 데이터를 나누는 모델"),
        ("랜덤 포레스트", "여러 의사결정트리를 결합한 모델")
    ]:
        st.write(f"**{name}:** {description}")


# ==========================================
# 메뉴 2. 전처리 및 근거
# ==========================================

elif menu == "② 전처리 및 근거":

    st.header("데이터 전처리 및 근거")

    a, b, c = st.columns(3)
    a.metric("원본 변수", 46)
    b.metric("선택 변수", 13)
    c.metric("제외 변수", 33)

    st.write(
        "원본 46개 변수 중 분석에 필요한 13개를 선정했습니다. "
        "원본 CSV의 셀 값은 변경하지 않았습니다."
    )

    a, b, c = st.columns(3)
    a.metric("응답자", f"{len(df):,}")
    b.metric("결측치", int(df.isna().sum().sum()))
    c.metric("중복 ID", int(df[id_col].duplicated().sum()))

    if ORIGINAL_FILE.exists():
        original = load_csv(str(ORIGINAL_FILE))

        report = pd.DataFrame([
            {
                "번호": i,
                "변수명": col,
                "처리": "유지" if i in KEEP else "제외",
                "근거": get_reason(i)
            }
            for i, col in enumerate(original.columns, start=1)
        ])

        choice = st.selectbox(
            "표시할 변수",
            ["전체", "유지", "제외"]
        )

        if choice != "전체":
            report = report[report["처리"] == choice]

        st.dataframe(
            report, hide_index=True, width="stretch"
        )

        expected = original.iloc[
            :, [i - 1 for i in KEEP]
        ].reset_index(drop=True)

        actual = df.reset_index(drop=True)

        same = (
            expected.columns.tolist() == actual.columns.tolist()
            and expected.astype("string").fillna("__NA__").equals(
                actual.astype("string").fillna("__NA__")
            )
        )

        if same:
            st.success("원본과 선택 데이터의 열 및 값이 일치합니다.")
        else:
            st.warning("원본과 선택 데이터가 일치하지 않습니다.")

    else:
        st.warning("원본 46열 CSV가 없어 비교표를 표시할 수 없습니다.")

    st.subheader("모델 학습 전처리")

    st.write(
        "매체 이용 여부는 분석용 데이터에서 "
        "이용함=1, 이용 안함=0으로 변환했습니다."
    )

    st.write(
        "학교급과 학년은 원-핫 인코딩합니다. "
        "이진 변수는 한 범주를 기준으로 제외해 "
        "중복된 입력 열을 줄였습니다."
    )

    st.write(
        "데이터를 학습 60%, 검증 20%, 최종 평가 20%로 "
        "분리하고, 인코딩은 학습 데이터에 대해서만 학습합니다."
    )

    st.info(
        "응답자 ID와 타깃 Y는 입력 X에 포함하지 않습니다."
    )


# ==========================================
# 메뉴 3. 데이터 분포 시각화
# ==========================================

elif menu == "③ 데이터 분포 시각화":

    st.header("데이터 분포 시각화")

    st.subheader("① 매체별 이용률")

    st.caption(
        "이진 범주형 / 가로 막대그래프 / "
        "매체별 이용률을 비교하기 위해 사용"
    )

    rate = (X_media.mean() * 100).sort_values(
        ascending=False
    )

    rate_df = rate.reset_index()
    rate_df.columns = ["매체", "이용률"]

    show_chart(
        alt.Chart(rate_df).mark_bar().encode(
            x=alt.X(
                "이용률:Q",
                scale=alt.Scale(domain=[0, 100])
            ),
            y=alt.Y("매체:N", sort="-x"),
            tooltip=[
                "매체",
                alt.Tooltip("이용률:Q", format=".2f")
            ]
        ).properties(height=380)
    )

    highest = rate_df.iloc[0]
    lowest = rate_df.iloc[-1]

    st.write(
        f"이용률이 가장 높은 매체는 {highest['매체']} "
        f"({highest['이용률']:.1f}%), "
        f"가장 낮은 매체는 {lowest['매체']} "
        f"({lowest['이용률']:.1f}%)입니다."
    )

    st.subheader("② 생성형 AI 이용 여부")

    target_df = pd.DataFrame({
        "이용 여부": ["이용함", "이용 안함"],
        "응답자 수": [
            int((Y == 1).sum()),
            int((Y == 0).sum())
        ]
    })

    show_chart(
        alt.Chart(target_df).mark_bar().encode(
            x="이용 여부:N",
            y="응답자 수:Q",
            color="이용 여부:N",
            tooltip=["이용 여부", "응답자 수"]
        ).properties(height=280)
    )

    st.write(
        f"생성형 AI 이용자는 {(Y == 1).sum():,}명 "
        f"({Y.mean() * 100:.1f}%)입니다."
    )

    st.subheader("③ 학교급별 응답자 분포")

    school_counts = school_labels.value_counts().reindex(
        list(SCHOOL_LABELS.values()), fill_value=0
    )

    st.bar_chart(school_counts)

    st.subheader("④ 학년별 응답자 분포")

    grade_counts = grade_labels.value_counts().reindex(
        list(GRADE_LABELS.values()), fill_value=0
    )

    st.bar_chart(grade_counts)

    st.caption(
        "학년 코드 1~9는 초4부터 고3까지의 "
        "순서로 표시합니다."
    )

    st.info(
        "이용률이 높은 변수가 반드시 분류에 "
        "가장 도움이 되는 것은 아닙니다."
    )


# ==========================================
# 메뉴 4. X와 Y의 관계
# ==========================================

elif menu == "④ 입력 변수와 타깃의 관계":

    st.header("입력 변수와 생성형 AI 이용 여부의 관계")

    st.subheader("① 매체별 상관계수")

    corr_rows = []

    for col in X_media.columns:
        value = X_media[col].corr(Y)
        corr_rows.append({
            "매체": col,
            "상관계수": value,
            "절댓값": abs(value)
        })

    corr_df = pd.DataFrame(corr_rows).sort_values(
        "절댓값", ascending=False
    )

    show_chart(
        alt.Chart(corr_df).mark_bar().encode(
            x=alt.X(
                "상관계수:Q",
                scale=alt.Scale(domain=[-1, 1])
            ),
            y=alt.Y(
                "매체:N",
                sort=list(corr_df["매체"])
            ),
            color=alt.condition(
                alt.datum.상관계수 >= 0,
                alt.value("#3B82F6"),
                alt.value("#F97316")
            ),
            tooltip=[
                "매체",
                alt.Tooltip("상관계수:Q", format=".3f")
            ]
        ).properties(height=380)
    )

    strongest = corr_df.iloc[0]

    st.write(
        f"상관계수의 절댓값이 가장 큰 매체는 "
        f"**{strongest['매체']}**이며, "
        f"상관계수는 {strongest['상관계수']:.3f}입니다."
    )

    st.subheader("② AI 이용자와 미이용자 비교")

    rows = []
    differences = []

    for col in X_media.columns:
        yes_rate = X_media.loc[Y == 1, col].mean() * 100
        no_rate = X_media.loc[Y == 0, col].mean() * 100

        rows.extend([
            {"매체": col, "집단": "AI 이용자", "이용률": yes_rate},
            {"매체": col, "집단": "AI 미이용자", "이용률": no_rate}
        ])

        differences.append({
            "매체": col,
            "차이(%p)": yes_rate - no_rate
        })

    group_df = pd.DataFrame(rows)

    show_chart(
        alt.Chart(group_df).mark_bar().encode(
            x=alt.X(
                "이용률:Q",
                scale=alt.Scale(domain=[0, 100])
            ),
            y="매체:N",
            yOffset="집단:N",
            color="집단:N",
            tooltip=[
                "매체",
                "집단",
                alt.Tooltip("이용률:Q", format=".2f")
            ]
        ).properties(height=430)
    )

    diff_df = pd.DataFrame(differences)
    diff_df["절댓값"] = diff_df["차이(%p)"].abs()
    diff_df = diff_df.sort_values(
        "절댓값", ascending=False
    )

    st.dataframe(
        diff_df.drop(columns="절댓값").round(2),
        hide_index=True, width="stretch"
    )

    top_diff = diff_df.iloc[0]

    st.write(
        f"두 집단의 이용률 차이가 가장 큰 매체는 "
        f"**{top_diff['매체']}**이며, "
        f"차이는 {abs(top_diff['차이(%p)']):.1f}%p입니다."
    )

    st.subheader("③ 학교급별 AI 이용률")

    school_rates = (
        pd.DataFrame({
            "학교급": school_labels,
            "AI 이용": Y
        })
        .groupby("학교급")["AI 이용"]
        .mean().mul(100)
        .reindex(list(SCHOOL_LABELS.values()))
        .reset_index(name="AI 이용률")
    )

    show_chart(
        alt.Chart(school_rates).mark_bar().encode(
            x=alt.X(
                "학교급:N",
                sort=list(SCHOOL_LABELS.values())
            ),
            y=alt.Y(
                "AI 이용률:Q",
                scale=alt.Scale(domain=[0, 100])
            ),
            tooltip=["학교급", "AI 이용률"]
        ).properties(height=290)
    )

    st.subheader("④ 학년별 AI 이용률")

    grade_rates = (
        pd.DataFrame({
            "학년": grade_labels,
            "AI 이용": Y
        })
        .groupby("학년")["AI 이용"]
        .mean().mul(100)
        .reindex(list(GRADE_LABELS.values()))
        .reset_index(name="AI 이용률")
    )

    show_chart(
        alt.Chart(grade_rates).mark_bar().encode(
            x=alt.X(
                "학년:N",
                sort=list(GRADE_LABELS.values())
            ),
            y=alt.Y(
                "AI 이용률:Q",
                scale=alt.Scale(domain=[0, 100])
            ),
            tooltip=["학년", "AI 이용률"]
        ).properties(height=310)
    )

    st.info(
        "상관관계는 두 변수의 관련성을 보여주지만 "
        "원인과 결과를 증명하지는 않습니다."
    )


# ==========================================
# 메뉴 5. 분류 모델 학습
# ==========================================

elif menu == "⑤ 분류 모델 학습":

    st.header("분류 모델 3개 학습")

    st.write(
        "세 모델을 동일한 데이터로 학습시키고, "
        "각 모델의 작동 원리와 실제 학습 결과를 살펴봅니다."
    )

    st.subheader("① 모델 선정 이유")

    model_info = pd.DataFrame([
        [
            "로지스틱 회귀",
            "입력 변수로 AI 이용 확률 계산",
            "기본 성능과 변수 계수 확인"
        ],
        [
            "의사결정트리",
            "조건에 따라 데이터를 반복 분할",
            "분류 과정 시각화"
        ],
        [
            "랜덤 포레스트",
            "여러 트리의 예측 결과 종합",
            "복잡한 관계 학습 및 과적합 완화"
        ]
    ], columns=["모델", "작동 원리", "선정 이유"])

    st.dataframe(
        model_info, hide_index=True, width="stretch"
    )

    trained = train_models(X, Y.astype(int))

    st.subheader("② 학습 데이터 구성")

    a, b, c = st.columns(3)

    a.metric("학습 데이터", f"{trained['counts']['학습']:,}명")
    b.metric("검증 데이터", f"{trained['counts']['검증']:,}명")
    c.metric("최종 평가", f"{trained['counts']['평가']:,}명")

    st.caption(
        "학습 60%, 검증 20%, 평가 20%로 나누었습니다. "
        "세 모델은 동일한 학습·검증 데이터를 사용합니다."
    )

    st.subheader("③ 모델별 작동 원리와 시각화")

    selected_model = st.selectbox(
        "확인할 분류 모델",
        MODEL_NAMES
    )

    result = trained["results"][selected_model]
    pipe = result["pipeline"]
    algorithm = pipe.named_steps["model"]

    # ======================================
    # 로지스틱 회귀
    # ======================================

    if selected_model == "로지스틱 회귀":

        st.markdown("### 로지스틱 회귀")

        st.write(
            "**작동 원리:** SNS, OTT, 숏폼 등 입력 변수에 "
            "학습된 가중치를 적용하여 생성형 AI를 "
            "이용할 확률을 계산합니다. "
            "확률이 0.5 이상이면 이용함으로 분류합니다."
        )

        st.write(
            "**선정 이유:** 작동 방식이 비교적 단순하며, "
            "다른 모델과 성능을 비교할 기준 모델로 적합합니다."
        )

        st.subheader("학습된 변수별 계수")

        coef_df = feature_table(
            pipe,
            algorithm.coef_[0],
            "계수"
        )

        coef_df["절댓값"] = coef_df["계수"].abs()

        coef_df = coef_df.sort_values(
            "절댓값", ascending=False
        )

        # 긴 인코딩 이름 대신 한글 표시
        coef_df["표시 계수"] = coef_df["계수"].map(
            lambda value: f"{value:+.3f}"
        )

        chart = alt.Chart(coef_df).mark_bar(
            color="#8BB9F0"
        ).encode(
            x=alt.X(
                "계수:Q",
                title="로지스틱 회귀 계수"
            ),
            y=alt.Y(
                "인코딩 항목:N",
                sort=list(coef_df["인코딩 항목"]),
                title=None,
                axis=alt.Axis(labelLimit=320)
            ),
            tooltip=[
                alt.Tooltip("원래 변수:N", title="원래 변수"),
                alt.Tooltip("인코딩 항목:N", title="전체 항목명"),
                alt.Tooltip("계수:Q", format="+.4f")
            ]
        ).properties(
            height=max(450, len(coef_df) * 30)
        )

        show_chart(chart)

        st.subheader("계수 전체 목록")

        st.dataframe(
            coef_df[
                ["원래 변수", "인코딩 항목", "계수"]
            ].round(4),
            hide_index=True,
            width="stretch"
        )

        st.info(
            "계수가 양수이면 해당 인코딩 항목이 "
            "AI 이용 확률을 높이는 방향과 관련되고, "
            "음수이면 낮추는 방향과 관련됩니다. "
            "다른 입력 조건이 같다는 가정에서의 관계이며 "
            "인과관계는 아닙니다."
        )

        st.caption(
            "학교급과 학년의 계수는 인코딩에서 제외된 "
            "기준 범주와 비교한 값입니다. "
            "계수의 절댓값만으로 변수 전체의 중요도를 "
            "단정하지 않습니다."
        )

    # ======================================
    # 의사결정트리
    # ======================================

    elif selected_model == "의사결정트리":

        st.markdown("### 의사결정트리")

        st.write(
            "**작동 원리:** 학교급, 학년, 매체 이용 여부 등의 "
            "조건으로 응답자를 단계적으로 나눕니다. "
            "마지막에 도달한 그룹의 AI 이용 비율을 바탕으로 "
            "이용 여부를 예측합니다."
        )

        st.write(
            "**선정 이유:** 조건이 나뉘는 과정을 "
            "나무 모양으로 확인할 수 있어 "
            "모델의 판단 과정을 이해하기 쉽습니다."
        )

        st.subheader("학습된 의사결정트리")

        depth = st.slider(
            "표시할 트리 깊이",
            min_value=2,
            max_value=5,
            value=3,
            step=1
        )

        raw_names = pipe.named_steps[
            "preprocessing"
        ].get_feature_names_out()

        feature_names = [
            readable_feature(name)
            for name in raw_names
        ]

        try:
            import matplotlib.pyplot as plt
            from matplotlib import font_manager

            # 사용 가능한 한글 글꼴 자동 탐색
            available = {
                font.name for font in font_manager.fontManager.ttflist
            }

            korean_fonts = [
                "NanumGothic",
                "Noto Sans CJK KR",
                "Noto Sans KR",
                "Malgun Gothic",
                "AppleGothic"
            ]

            found_font = next(
                (font for font in korean_fonts if font in available),
                None
            )

            if found_font:
                plt.rcParams["font.family"] = found_font
                plt.rcParams["axes.unicode_minus"] = False

            fig_width = max(18, 5 * (2 ** (depth - 1)))
            fig, ax = plt.subplots(
                figsize=(fig_width, 10)
            )

            plot_tree(
                algorithm,
                feature_names=feature_names,
                class_names=["AI 미이용", "AI 이용"],
                filled=True,
                rounded=True,
                proportion=False,
                impurity=False,
                max_depth=depth,
                fontsize=9,
                ax=ax
            )

            ax.set_title(
                "학습된 의사결정트리",
                fontsize=16,
                pad=20
            )

            fig.tight_layout()

            st.pyplot(fig, width="content")
            plt.close(fig)

        except ImportError:
            st.error(
                "트리 시각화에는 matplotlib가 필요합니다. "
                "requirements.txt에 matplotlib를 추가하세요."
            )

        st.subheader("그래프 읽는 방법")

        st.write(
            "각 상자는 하나의 분류 조건 또는 예측 결과입니다. "
            "조건이 참이면 왼쪽, 거짓이면 오른쪽으로 이동합니다."
        )

        st.write(
            "**samples:** 해당 상자에 포함된 학습 응답자 수\n\n"
            "**value:** AI 미이용자와 이용자의 학습 응답자 수\n\n"
            "**class:** 해당 상자에서 예측하는 범주"
        )

        st.write(
            "나무의 위쪽일수록 먼저 확인하는 조건입니다. "
            "아래쪽으로 이동할수록 더 세부적인 조건으로 나뉩니다."
        )

        # 최상위 조건 설명
        root_feature_index = algorithm.tree_.feature[0]

        if root_feature_index >= 0:
            root_feature_name = feature_names[root_feature_index]

            st.success(
                f"이 의사결정트리가 가장 먼저 확인하는 "
                f"조건은 **{root_feature_name}**입니다."
            )

        st.caption(
            "모델은 최대 깊이 5로 학습되었으며, "
            "화면에서 선택한 깊이는 표시 범위만 바꿉니다. "
            "학습된 모델 자체가 바뀌지는 않습니다."
        )

    # ======================================
    # 랜덤 포레스트
    # ======================================

    else:

        st.markdown("### 랜덤 포레스트")

        st.write(
            "**작동 원리:** 여러 개의 의사결정트리를 "
            "학습시킨 다음 각 트리의 예측을 종합하여 "
            "최종 이용 여부를 결정합니다."
        )

        st.write(
            "**선정 이유:** 단일 의사결정트리보다 "
            "예측이 안정적일 가능성이 있고, "
            "복잡한 변수 관계를 학습할 수 있습니다."
        )

        st.subheader("원래 변수별 중요도")

        importance_df = feature_table(
            pipe,
            algorithm.feature_importances_,
            "중요도"
        )

        # 학교급과 학년의 여러 인코딩 항목을 합산
        grouped_importance = (
            importance_df.groupby(
                "원래 변수", as_index=False
            )["중요도"]
            .sum()
            .sort_values("중요도", ascending=False)
        )

        importance_chart = alt.Chart(
            grouped_importance
        ).mark_bar(
            color="#8BB9F0"
        ).encode(
            x=alt.X(
                "중요도:Q",
                title="변수 중요도"
            ),
            y=alt.Y(
                "원래 변수:N",
                sort="-x",
                title=None,
                axis=alt.Axis(labelLimit=320)
            ),
            tooltip=[
                "원래 변수",
                alt.Tooltip("중요도:Q", format=".4f")
            ]
        ).properties(height=410)

        show_chart(importance_chart)

        st.dataframe(
            grouped_importance.round(4),
            hide_index=True,
            width="stretch"
        )

        top_variable = grouped_importance.iloc[0]

        st.write(
            f"랜덤 포레스트에서 가장 높은 중요도를 "
            f"보인 변수는 **{top_variable['원래 변수']}**입니다."
        )

        st.info(
            "이 중요도는 트리의 분할 과정에서 "
            "불순도를 감소시킨 정도를 바탕으로 계산됩니다. "
            "변수 중요도가 높다고 해서 "
            "원인이라고 해석해서는 안 됩니다."
        )

        st.caption(
            "학교급과 학년처럼 여러 범주로 인코딩된 "
            "변수의 중요도는 합산하여 표시했습니다. "
            "이 방식은 범주 수와 상관된 편향이 있을 수 있으므로 "
            "중요도를 절대적인 영향력으로 보지 않습니다."
        )

    # ======================================
    # 선택 모델 검증 점수
    # ======================================

    st.subheader("④ 선택한 모델의 검증 결과")

    score = result["scores"]

    cols = st.columns(5)

    for col, (name, value) in zip(cols, score.items()):
        col.metric(name, f"{value:.3f}")

    st.write(
        f"**{selected_model}**의 검증 데이터 정확도는 "
        f"{score['정확도']:.3f}, "
        f"F1 점수는 {score['F1']:.3f}입니다."
    )

    st.caption(
        "위 점수는 학습에 사용하지 않은 "
        "검증 데이터에서 계산한 결과입니다."
    )


# ==========================================
# 메뉴 6. 모델 성능 비교 및 선정
# ==========================================

elif menu == "⑥ 모델 성능 비교 및 선정":

    st.header("모델 성능 비교 및 선정")

    trained = train_models(X, Y.astype(int))

    results = trained["results"]
    summary = scores_dataframe(results)

    st.subheader("① 세 모델의 검증 점수")

    st.dataframe(
        summary.round(3),
        hide_index=True,
        width="stretch"
    )

    st.subheader("② 성능 비교 그래프")

    metric_name = st.selectbox(
        "비교할 성능 지표",
        ["정확도", "정밀도", "재현율", "F1", "ROC-AUC"],
        index=3
    )

    show_chart(
        alt.Chart(summary).mark_bar().encode(
            x=alt.X(
                "모델:N",
                sort=MODEL_NAMES
            ),
            y=alt.Y(
                f"{metric_name}:Q",
                scale=alt.Scale(domain=[0, 1]),
                title=metric_name
            ),
            color="모델:N",
            tooltip=[
                "모델",
                alt.Tooltip(
                    f"{metric_name}:Q",
                    format=".3f"
                )
            ]
        ).properties(height=350)
    )

    explanations = {
        "정확도": "전체 응답 중 올바르게 예측한 비율",
        "정밀도": "AI 이용자라고 예측한 사람 중 실제 이용자 비율",
        "재현율": "실제 AI 이용자 중 올바르게 찾아낸 비율",
        "F1": "정밀도와 재현율의 조화평균",
        "ROC-AUC": "분류 기준 전반에서 두 집단을 구별하는 능력"
    }

    st.info(explanations[metric_name])

    st.subheader("③ 모델별 혼동행렬")

    selected = st.selectbox(
        "혼동행렬을 볼 모델",
        MODEL_NAMES
    )

    cm = confusion_matrix(
        trained["valid_y"],
        results[selected]["pred"],
        labels=[0, 1]
    )

    labels = ["AI 미이용", "AI 이용"]
    cells = []

    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            cells.append({
                "실제": actual,
                "예측": predicted,
                "응답자 수": int(cm[i, j])
            })

    cm_df = pd.DataFrame(cells)

    base = alt.Chart(cm_df).encode(
        x=alt.X(
            "예측:N",
            sort=labels
        ),
        y=alt.Y(
            "실제:N",
            sort=labels
        )
    )

    rectangles = base.mark_rect().encode(
        color=alt.Color("응답자 수:Q"),
        tooltip=["실제", "예측", "응답자 수"]
    )

    numbers = base.mark_text(
        fontSize=20,
        color="white"
    ).encode(
        text="응답자 수:Q"
    )

    show_chart(
        (rectangles + numbers).properties(height=290)
    )

    st.write(
        "혼동행렬의 대각선 두 칸은 올바르게 예측한 "
        "응답자 수이고, 나머지는 잘못 예측한 수입니다."
    )

    st.subheader("④ ROC 곡선 비교")

    roc_rows = []

    for name, result in results.items():
        fpr, tpr, _ = roc_curve(
            trained["valid_y"],
            result["prob"]
        )

        for f, t in zip(fpr, tpr):
            roc_rows.append({
                "모델": name,
                "거짓양성률": float(f),
                "참양성률": float(t)
            })

    roc_df = pd.DataFrame(roc_rows)

    show_chart(
        alt.Chart(roc_df).mark_line().encode(
            x=alt.X(
                "거짓양성률:Q",
                scale=alt.Scale(domain=[0, 1])
            ),
            y=alt.Y(
                "참양성률:Q",
                scale=alt.Scale(domain=[0, 1])
            ),
            color="모델:N",
            tooltip=["모델", "거짓양성률", "참양성률"]
        ).properties(height=330)
    )

    st.subheader("⑤ 최종 모델 선정")

    ranking = summary.sort_values(
        ["F1", "ROC-AUC", "정확도"],
        ascending=False
    )

    st.dataframe(
        ranking.round(3),
        hide_index=True,
        width="stretch"
    )

    best_name = trained["best_name"]

    st.success(
        f"검증 데이터의 F1 점수를 주요 기준으로 "
        f"선정한 모델은 **{best_name}**입니다."
    )

    st.write(
        "세 모델의 정확도, 정밀도, 재현율, "
        "F1 점수, ROC-AUC를 함께 비교했습니다. "
        "F1 점수가 같으면 ROC-AUC, "
        "그다음 정확도 순서로 비교했습니다."
    )

    st.subheader("⑥ 선정 모델의 최종 평가")

    st.write(
        "모델 선정 이후, 학습·검증 데이터를 합쳐 "
        "선정한 모델을 다시 학습시켰습니다. "
        "최종 평가 데이터는 모델 선정에 사용하지 않았습니다."
    )

    cols = st.columns(5)

    for col, (name, value) in zip(
        cols,
        trained["final_scores"].items()
    ):
        col.metric(name, f"{value:.3f}")

    final = trained["final_scores"]

    st.write(
        f"선정 모델인 **{best_name}**의 최종 평가 정확도는 "
        f"{final['정확도']:.3f}, "
        f"F1 점수는 {final['F1']:.3f}입니다."
    )

    st.subheader("⑦ 탐구 결과 해석")

    st.write(
        "청소년의 디지털 매체 이용 경험과 학교급, 학년을 "
        "입력으로 사용하여 생성형 AI 이용 여부를 "
        "예측하는 이진 분류 모델을 학습했습니다."
    )

    st.write(
        "서로 다른 방식으로 작동하는 세 모델을 "
        "같은 데이터 조건에서 비교하고, "
        "검증 결과에 따라 최종 모델을 선정했습니다."
    )

    st.write(
        "다만 이 결과는 해당 설문 데이터에 대한 평가입니다. "
        "다른 시기나 다른 청소년 집단에서도 "
        "동일한 성능이 나온다고 단정할 수는 없습니다."
    )

    st.info(
        "상관관계나 변수 중요도가 높더라도 "
        "해당 매체가 생성형 AI 이용의 원인이라는 뜻은 아닙니다. "
        "또한 설문 가중치를 적용하지 않았으므로 "
        "전체 청소년 집단으로 일반화할 때 주의가 필요합니다."
    )
