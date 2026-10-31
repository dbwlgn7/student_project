
from pathlib import Path

import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

st.set_page_config(
    page_title="청소년 AI 이용 분류 탐구",
    page_icon="📊",
    layout="wide"
)

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"

SELECTED_FILE = DATA_DIR / "청소년_AI이용분류_ID포함13열.csv"
ORIGINAL_FILE = DATA_DIR / "청소년_AI이용분류_정제데이터.csv"

KEEP = [1, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 36, 40]


@st.cache_data
def load_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def short_name(column):
    if " - " in column:
        return column.split(" - ", 1)[1].split(") ", 1)[-1]
    return column


def get_columns(df):
    columns = list(df.columns)

    target = next(
        (c for c in columns if "생성형 AI" in c and "이용 경험" in c),
        None
    )

    id_col = next(
        (c for c in columns if "응답자 ID" in c),
        columns[0]
    )

    school = next(
        (c for c in columns if "학교급(기본)" in c),
        None
    )

    grade = next(
        (c for c in columns if "학년별" in c),
        None
    )

    media = [
        c for c in columns
        if c not in [target, id_col, school, grade]
    ]

    return target, id_col, school, grade, media


def reason(number):
    if number == 1:
        return "응답자를 구분하고 결과를 확인하기 위해 유지하였다. 모델 입력에는 사용하지 않는다."

    if number == 12:
        return "생성형 AI 이용 여부는 분류 모델이 예측해야 하는 정답이므로 유지하였다."

    if number in [36, 40]:
        return "학교급과 학년에 따른 차이도 분석하기 위해 유지하였다."

    if number in KEEP:
        return "디지털 매체 이용 경험을 직접 나타내므로 입력 변수로 선정하였다."

    if number == 2:
        return "학교를 구분하기 위한 번호로 디지털 매체 이용 특성을 나타내지 않아 제외하였다."

    if number in [8, 10, 15, 16]:
        return "디지털 매체와 관련이 있지만 이번 탐구에서 분석할 매체의 범위를 제한하기 위해 제외하였다."

    if number in [17, 18]:
        return "종이 매체 이용 경험으로 디지털 매체 이용 특성이라는 탐구 주제와 직접적인 관련성이 낮아 제외하였다."

    if number in [19, 20, 21]:
        return "생성형 AI가 자주 이용하는 매체의 응답에 직접 포함될 수 있어 정답 정보가 입력 변수로 유출되는 것을 방지하기 위해 제외하였다."

    if number in [22, 23, 24, 25, 26, 27, 28, 29]:
        return "개인정보 보호 및 온라인 안전 관련 경험으로 매체별 이용 여부를 직접 나타내지 않아 제외하였다."

    if number in [30, 31, 32, 33]:
        return "디지털 매체 이용 특성보다는 교육 경험과 교육 효과를 나타내는 변수이므로 제외하였다."

    if number in [34, 35]:
        return "개인 및 학교 특성에 해당하므로 디지털 매체 이용 경험 중심의 분석 범위에서 제외하였다."

    if number in [37, 38, 39, 43, 44]:
        return "기본 학교급 변수와 유사한 정보를 포함하므로 분석 변수의 중복을 줄이기 위해 제외하였다."

    if number in [41, 42]:
        return "지역 정보를 나타내며 이번 탐구에서 분석하는 매체 이용 특성에 집중하기 위해 제외하였다."

    if number in [45, 46]:
        return "설문조사의 가중치로 디지털 매체 이용 경험 자체를 나타내지 않아 모델 입력에서 제외하였다."

    return "탐구 주제와의 관련성을 고려하여 제외하였다."


st.sidebar.title("🧭 탐구 메뉴")

menu = st.sidebar.radio(
    "이동할 메뉴",
    [
        "① 탐구 소개",
        "② 전처리 및 근거",
        "③ 데이터 분포 시각화",
        "④ 변수 간 관계",
        "⑤ 분류 모델 평가",
        "⑥ 탐구 결론"
    ]
)

uploaded = st.sidebar.file_uploader(
    "13열 CSV 직접 올리기 (선택)",
    type="csv"
)

if uploaded is not None:
    df = pd.read_csv(uploaded, encoding="utf-8-sig")
elif SELECTED_FILE.exists():
    df = load_csv(str(SELECTED_FILE))
else:
    st.error("13열 CSV를 찾지 못했습니다. GitHub data 폴더를 확인해 주세요.")
    st.stop()

target, id_col, school, grade, media = get_columns(df)

if target is None:
    st.error("생성형 AI 이용 여부 변수를 찾지 못했습니다.")
    st.stop()

# 원본 데이터 자체는 변경하지 않는다.
# 그래프 계산에만 별도 복사본을 사용한다.
X_media = df[media].copy()

for col in media:
    X_media[col] = pd.to_numeric(
        X_media[col], errors="coerce"
    ).map({1: 1, 2: 0})

y = pd.to_numeric(
    df[target], errors="coerce"
).map({1: 1, 2: 0})

st.title("📊 청소년 디지털 매체 이용과 생성형 AI 분류 탐구")
st.caption(
    "청소년의 디지털 매체 이용 특성으로 생성형 AI 이용 여부를 분류할 수 있을까?"
)

# --------------------------------------------------
# 1. 탐구 소개
# --------------------------------------------------

if menu == "① 탐구 소개":

    st.header("탐구 소개")

    st.write(
        "다른 디지털 매체 이용 경험과 학교급, 학년을 "
        "바탕으로 생성형 AI 이용 여부를 분류할 수 있는지 탐구한다."
    )

    a, b, c = st.columns(3)

    a.metric("응답자 수", f"{len(df):,}명")
    b.metric("선택한 변수", len(df.columns))
    c.metric("결측치", int(df.isna().sum().sum()))

    st.subheader("분류 모델 구성")

    st.write("**입력 변수(X)**")
    st.write(", ".join(short_name(c) for c in media))

    st.write("학교급과 학년도 추가 입력 변수로 활용한다.")

    st.write("**타깃 변수(Y)**")
    st.write("생성형 AI 이용 경험")

    st.info(
        "응답자 ID는 식별용으로만 사용하며, "
        "생성형 AI 이용 여부는 모델의 입력에 포함하지 않는다."
    )

# --------------------------------------------------
# 2. 전처리 및 근거
# --------------------------------------------------

elif menu == "② 전처리 및 근거":

    st.header("데이터 전처리 및 근거")

    a, b, c = st.columns(3)

    a.metric("원본 변수", 46)
    b.metric("유지한 변수", 13)
    c.metric("제외한 변수", 33)

    st.write(
        "원본의 46개 변수 중 탐구 주제와 관련된 13개를 "
        "선택하였다. 응답 코드와 행 순서는 변경하지 않았다."
    )

    st.subheader("데이터 품질 검사")

    st.write("결측치:", int(df.isna().sum().sum()))
    st.write("중복 응답자 ID:", int(df[id_col].duplicated().sum()))

    if ORIGINAL_FILE.exists():

        original = load_csv(str(ORIGINAL_FILE))

        records = []

        for i, col in enumerate(original.columns, start=1):
            records.append({
                "번호": i,
                "변수": col,
                "처리": "유지" if i in KEEP else "제외",
                "전처리 근거": reason(i)
            })

        report = pd.DataFrame(records)

        choice = st.selectbox(
            "표시할 항목",
            ["전체", "유지", "제외"]
        )

        if choice != "전체":
            report = report[report["처리"] == choice]

        st.dataframe(report, hide_index=True, width="stretch")

        expected = original.iloc[:, [i - 1 for i in KEEP]]

        if expected.shape == df.shape:
            same = expected.reset_index(drop=True).equals(
                df.reset_index(drop=True)
            )

            if same:
                st.success("원본 데이터와 선택 데이터의 셀 값이 일치합니다.")
            else:
                st.warning("원본과 선택 데이터 사이에 차이가 발견되었습니다.")

    else:
        st.warning("원본 46열 CSV를 찾지 못해 변수별 비교표를 표시할 수 없습니다.")

    st.caption(
        "변수를 제외한 이유는 탐구 범위에 따른 판단이며, "
        "실제 예측력이 없다고 확인된 것은 아니다."
    )

# --------------------------------------------------
# 3. 데이터 분포 시각화
# --------------------------------------------------

elif menu == "③ 데이터 분포 시각화":

    st.header("디지털 매체 이용 분포")

    rates = (X_media.mean() * 100).sort_values(ascending=False)

    rate_df = pd.DataFrame({
        "매체": [short_name(c) for c in rates.index],
        "이용률": rates.values
    })

    st.subheader("매체별 이용률")

    chart = alt.Chart(rate_df).mark_bar().encode(
        x=alt.X(
            "이용률:Q",
            scale=alt.Scale(domain=[0, 100]),
            title="이용률(%)"
        ),
        y=alt.Y("매체:N", sort="-x"),
        tooltip=["매체", "이용률"]
    ).properties(height=400)

    st.altair_chart(chart, width="stretch")

    st.dataframe(rate_df.round(2), hide_index=True)

    st.subheader("매체별 이용 여부 비교")

    rows = []

    for col in media:

        rate = X_media[col].mean() * 100

        rows.append({
            "매체": short_name(col),
            "응답": "이용함",
            "비율": rate
        })

        rows.append({
            "매체": short_name(col),
            "응답": "이용 안함",
            "비율": 100 - rate
        })

    stacked = pd.DataFrame(rows)

    chart2 = alt.Chart(stacked).mark_bar().encode(
        y="매체:N",
        x=alt.X(
            "비율:Q",
            stack="zero",
            scale=alt.Scale(domain=[0, 100])
        ),
        color="응답:N",
        tooltip=["매체", "응답", "비율"]
    ).properties(height=400)

    st.altair_chart(chart2, width="stretch")

    st.subheader("학교급별 응답자 수")

    if school is not None:

        school_data = df[school].astype(str).map({
            "1": "초등학교",
            "2": "중학교",
            "3": "고등학교"
        })

        st.bar_chart(school_data.value_counts())

    st.subheader("학년별 응답자 수")

    if grade is not None:
        st.bar_chart(df[grade].value_counts().sort_index())

# --------------------------------------------------
# 4. 변수 간 관계
# --------------------------------------------------

elif menu == "④ 변수 간 관계":

    st.header("변수 간 상관관계 분석")

    st.write(
        "각 매체의 이용 여부를 0과 1로 변환한 분석용 데이터에서 "
        "상관관계를 계산한다."
    )

    corr = X_media.corr()

    corr_data = corr.stack().reset_index()
    corr_data.columns = ["변수A", "변수B", "상관계수"]

    corr_data["변수A"] = corr_data["변수A"].map(short_name)
    corr_data["변수B"] = corr_data["변수B"].map(short_name)

    heatmap = alt.Chart(corr_data).mark_rect().encode(
        x=alt.X("변수A:N", axis=alt.Axis(labelAngle=-45)),
        y="변수B:N",
        color=alt.Color(
            "상관계수:Q",
            scale=alt.Scale(
                scheme="redblue",
                domain=[-1, 1]
            )
        ),
        tooltip=["변수A", "변수B", "상관계수"]
    ).properties(height=450)

    st.altair_chart(heatmap, width="stretch")

    st.info(
        "상관관계가 높다고 반드시 변수를 삭제하지 않는다. "
        "실제 분류 성능을 비교한 후 판단해야 한다."
    )

# --------------------------------------------------
# 5, 6. 분류 모델 및 결론
# --------------------------------------------------

elif menu in ["⑤ 분류 모델 평가", "⑥ 탐구 결론"]:

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        roc_auc_score,
        roc_curve
    )

    st.header(
        "분류 모델 평가"
        if menu == "⑤ 분류 모델 평가"
        else "탐구 결론"
    )

    if df.isna().any().any() or y.isna().any():
        st.error("결측치 또는 잘못된 타깃 코드가 있습니다.")
        st.stop()

    features = media.copy()

    if school is not None:
        features.append(school)

    if grade is not None:
        features.append(grade)

    X = df[features].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y.astype(int),
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    algorithms = {
        "로지스틱 회귀": LogisticRegression(
            max_iter=1500,
            random_state=42
        ),
        "랜덤 포레스트": RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
    }

    results = {}

    for name, algorithm in algorithms.items():

        preprocessing = ColumnTransformer([
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                features
            )
        ])

        pipeline = Pipeline([
            ("preprocessing", preprocessing),
            ("model", algorithm)
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        results[name] = {
            "정확도": accuracy_score(y_test, predictions),
            "정밀도": precision_score(y_test, predictions),
            "재현율": recall_score(y_test, predictions),
            "F1": f1_score(y_test, predictions),
            "ROC-AUC": roc_auc_score(y_test, probabilities),
            "예측값": predictions,
            "확률": probabilities
        }

    summary = pd.DataFrame([
        {
            "모델": name,
            "정확도": result["정확도"],
            "정밀도": result["정밀도"],
            "재현율": result["재현율"],
            "F1": result["F1"],
            "ROC-AUC": result["ROC-AUC"]
        }
        for name, result in results.items()
    ])

    st.dataframe(summary.round(3), hide_index=True, width="stretch")

    if menu == "⑤ 분류 모델 평가":

        selected_model = st.selectbox(
            "모델 선택",
            list(results.keys())
        )

        result = results[selected_model]

        st.subheader("혼동행렬")

        matrix = confusion_matrix(
            y_test,
            result["예측값"],
            labels=[0, 1]
        )

        labels = ["이용 안함", "이용함"]

        matrix_rows = []

        for i, actual in enumerate(labels):
            for j, predicted in enumerate(labels):
                matrix_rows.append({
                    "실제": actual,
                    "예측": predicted,
                    "건수": int(matrix[i, j])
                })

        matrix_df = pd.DataFrame(matrix_rows)

        matrix_chart = alt.Chart(matrix_df).mark_rect().encode(
            x="예측:N",
            y="실제:N",
            color="건수:Q",
            tooltip=["실제", "예측", "건수"]
        ).properties(height=300)

        st.altair_chart(matrix_chart, width="stretch")

        st.subheader("ROC 곡선")

        roc_rows = []

        for name, result in results.items():

            fpr, tpr, _ = roc_curve(
                y_test,
                result["확률"]
            )

            for x, z in zip(fpr, tpr):
                roc_rows.append({
                    "모델": name,
                    "거짓양성률": float(x),
                    "참양성률": float(z)
                })

        roc_df = pd.DataFrame(roc_rows)

        roc_chart = alt.Chart(roc_df).mark_line().encode(
            x="거짓양성률:Q",
            y="참양성률:Q",
            color="모델:N"
        ).properties(height=350)

        st.altair_chart(roc_chart, width="stretch")

    else:

        best = summary.sort_values(
            "F1", ascending=False
        ).iloc[0]

        st.success(
            f"현재 평가 데이터에서 F1 점수가 더 높은 모델은 "
            f"{best['모델']}이며, F1 점수는 {best['F1']:.3f}이다."
        )

        st.write(
            "이번 탐구에서는 디지털 매체 이용 경험과 "
            "학교급, 학년 정보를 활용해 생성형 AI 이용 여부를 "
            "분류하는 모델을 학습하였다."
        )

        st.write(
            "모델의 정확도와 F1 점수 등을 통해 실제 분류 가능성을 "
            "평가할 수 있었다."
        )

        st.write(
            "다만 이 결과는 해당 조사 데이터에서 얻어진 것이므로 "
            "다른 시기의 청소년에게도 동일한 성능을 보장하지 않는다."
        )

        st.write(
            "**추가 탐구:** 다른 디지털 매체 변수를 추가하거나 "
            "제외하여 분류 성능이 어떻게 변화하는지 비교할 수 있다."
        )
