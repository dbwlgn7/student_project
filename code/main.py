
from pathlib import Path

import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
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
    "TV방송",
    "OTT",
    "인터넷 개인방송",
    "숏폼",
    "포털 플랫폼",
    "SNS",
    "메타버스",
    "온라인 게임",
    "웹툰"
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
        path,
        encoding="utf-8-sig",
        low_memory=False
    )


def show_chart(chart):
    st.altair_chart(chart, width="stretch")


def get_reason(n):
    if n == 1:
        return "응답자 식별 및 중복 확인에 사용하며 모델 입력에서는 제외한다."
    if n == 12:
        return "생성형 AI 이용 여부는 예측할 타깃(Y)이므로 유지한다."
    if n in [36, 40]:
        return "학교급과 학년에 따른 이용 특성을 분석하기 위해 유지한다."
    if n in KEEP:
        return "디지털 매체 이용 경험을 나타내는 입력 변수(X)이므로 유지한다."
    if n == 2:
        return "학교 식별 번호이므로 입력에서 제외한다."
    if n in [8, 10, 15, 16]:
        return "디지털 매체 관련 변수지만 이번 탐구의 분석 범위를 제한하기 위해 제외한다."
    if n in [17, 18]:
        return "종이 매체로 디지털 매체 이용 중심 분석에서 제외한다."
    if n in [19, 20, 21]:
        return "생성형 AI가 응답에 포함될 수 있어 타깃 정보 누출 가능성을 줄이기 위해 제외한다."
    if 22 <= n <= 29:
        return "온라인 안전 및 개인정보 보호 경험으로 매체 이용 여부 중심 분석에서 제외한다."
    if 30 <= n <= 33:
        return "교육 경험과 효과 관련 변수이므로 제외한다."
    if n in [34, 35]:
        return "개인·학교 배경 특성으로 이번 분석 범위에서 제외한다."
    if n in [37, 38, 39, 43, 44]:
        return "학교급·학년과 관련된 추가 정보로 기본 변수만 사용하기 위해 제외한다."
    if n in [41, 42]:
        return "지역 관련 변수로 이번 탐구 범위에서 제외한다."
    if n in [45, 46]:
        return "설문조사 가중치이므로 모델 입력에서는 제외한다."
    return "탐구 목적에 따른 변수 선택 범위에서 제외한다."


def score_model(y_true, pred, prob):
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


def score_table(results):
    rows = []
    for name, item in results.items():
        row = {"모델": name}
        row.update(item["점수"])
        rows.append(row)
    return pd.DataFrame(rows)


def make_models():
    return {
        "로지스틱 회귀": LogisticRegression(
            max_iter=1500,
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


@st.cache_resource(show_spinner="분류 모델 3개를 학습하는 중입니다.")
def train_all_models(X, y):
    # 전체 데이터 60% 학습, 20% 검증, 20% 최종 평가
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

        preprocessing = ColumnTransformer([
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(X.columns)
            )
        ])

        pipe = Pipeline([
            ("preprocessing", preprocessing),
            ("model", algorithm)
        ])

        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_valid)
        prob = pipe.predict_proba(X_valid)[:, 1]

        results[name] = {
            "모델": pipe,
            "예측값": pred,
            "확률": prob,
            "점수": score_model(y_valid, pred, prob)
        }

    # 검증용 데이터의 F1으로 최종 모델을 선정
    best_name = max(
        results,
        key=lambda n: results[n]["점수"]["F1"]
    )

    # 최종 평가 데이터는 모델 선정에 사용하지 않음
    # 선택된 모델만 학습+검증 데이터로 재학습
    final_algorithm = make_models()[best_name]

    final_pipe = Pipeline([
        (
            "preprocessing",
            ColumnTransformer([
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    list(X.columns)
                )
            ])
        ),
        ("model", final_algorithm)
    ])

    final_pipe.fit(X_build, y_build)

    test_pred = final_pipe.predict(X_test)
    test_prob = final_pipe.predict_proba(X_test)[:, 1]

    final_scores = score_model(
        y_test, test_pred, test_prob
    )

    return {
        "results": results,
        "best_name": best_name,
        "final_model": final_pipe,
        "final_scores": final_scores,
        "y_valid": y_valid,
        "y_test": y_test,
        "test_pred": test_pred,
        "test_prob": test_prob,
        "counts": {
            "학습": len(X_train),
            "검증": len(X_valid),
            "최종 평가": len(X_test)
        }
    }


# ==========================================
# 3. 사이드바
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
    df = pd.read_csv(
        uploaded,
        encoding="utf-8-sig",
        low_memory=False
    )
elif SELECTED_FILE.exists():
    df = load_csv(str(SELECTED_FILE))
else:
    st.error("13열 CSV를 찾지 못했습니다.")
    st.stop()

if df.shape[1] != 13:
    st.error(
        f"13개 열이 필요하지만 현재 {df.shape[1]}개입니다."
    )
    st.stop()

# 선택 CSV의 열 순서는 원본 KEEP 순서와 같음
id_col = df.columns[0]
media_columns = list(df.columns[1:8]) + list(df.columns[9:11])
target_col = df.columns[8]
school_col = df.columns[11]
grade_col = df.columns[12]

# 예상 열 구성 확인
if (
    "응답자 ID" not in str(id_col)
    or "생성형 AI" not in str(target_col)
    or "학교급" not in str(school_col)
    or "학년별" not in str(grade_col)
):
    st.error(
        "CSV 열 구성과 순서가 예상한 13열 데이터와 다릅니다."
    )
    st.stop()

X_media = pd.DataFrame(index=df.index)

for name, col in zip(MEDIA_NAMES, media_columns):
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
        "일부 데이터의 응답 코드가 예상 범위를 벗어났습니다. "
        "CSV와 코드북을 확인해 주세요."
    )
    st.stop()

# 실제 모델의 입력 변수
# ID와 Y는 포함하지 않는다.
X = X_media.copy()
X["학교급"] = school_labels
X["학년"] = grade_labels

st.title("📊 청소년 생성형 AI 이용 분류 탐구")

st.caption(
    "탐구 주제: 청소년의 디지털 매체 이용 특성으로 "
    "생성형 AI 이용 여부를 분류할 수 있을까?"
)


# ==========================================
# 메뉴 1. 탐구 소개
# ==========================================

if menu == "① 탐구 소개":

    st.header("탐구 소개")

    a, b, c = st.columns(3)

    a.metric("응답자", f"{len(df):,}명")
    b.metric("선택 변수", len(df.columns))
    c.metric("입력 변수", X.shape[1])

    st.subheader("탐구 질문")

    st.write(
        "청소년의 SNS, 숏폼, OTT, 게임 등 디지털 매체 "
        "이용 경험과 학교급·학년을 바탕으로 "
        "생성형 AI 이용 여부를 분류할 수 있을까?"
    )

    st.subheader("왜 분류인가?")

    methods = pd.DataFrame([
        {
            "학습 유형": "회귀",
            "예측 대상": "연속적인 수치",
            "이번 탐구": "주목적과 다름"
        },
        {
            "학습 유형": "분류",
            "예측 대상": "이용함 / 이용 안함",
            "이번 탐구": "선택"
        },
        {
            "학습 유형": "군집",
            "예측 대상": "정답 없이 유사한 집단 구성",
            "이번 탐구": "주목적과 다름"
        }
    ])

    st.dataframe(methods, hide_index=True, width="stretch")

    st.success(
        "생성형 AI 이용 여부는 이용함과 이용 안함이라는 "
        "두 범주로 이루어져 있으므로 지도학습의 "
        "이진 분류 문제에 해당합니다."
    )

    st.subheader("데이터 속성과 유형")

    types = pd.DataFrame([
        ["응답자 ID", "식별자", "중복 검사", "제외"],
        ["9개 매체 이용 경험", "이진 범주형", "입력 X", "포함"],
        ["학교급", "순서형 범주", "입력 X", "포함"],
        ["학년", "순서형 범주", "입력 X", "포함"],
        ["생성형 AI 이용 여부", "이진 범주형", "타깃 Y", "제외"]
    ], columns=[
        "속성", "데이터 유형", "역할", "모델 입력"
    ])

    st.dataframe(types, hide_index=True, width="stretch")

    st.subheader("탐구 과정")

    st.write(
        "① 변수 선정 및 전처리 → "
        "② 데이터 시각화 → "
        "③ X와 Y의 관계 분석 → "
        "④ 세 분류 모델 학습 → "
        "⑤ 성능 비교와 모델 선정"
    )


# ==========================================
# 메뉴 2. 전처리 및 근거
# ==========================================

elif menu == "② 전처리 및 근거":

    st.header("전처리 과정 및 근거")

    a, b, c = st.columns(3)
    a.metric("원본 열", 46)
    b.metric("선택 열", 13)
    c.metric("제외 열", 33)

    st.write(
        "원본 데이터에서 탐구에 필요한 13개 열을 선택하고, "
        "원래의 행 순서와 응답 숫자 코드를 유지했습니다."
    )

    a, b, c = st.columns(3)
    a.metric("응답자 수", f"{len(df):,}")
    b.metric("결측치", int(df.isna().sum().sum()))
    c.metric("중복 ID", int(df[id_col].duplicated().sum()))

    st.subheader("변수별 선택·제외 근거")

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

        filter_value = st.selectbox(
            "표시 항목",
            ["전체", "유지", "제외"]
        )

        if filter_value != "전체":
            report = report[
                report["처리"] == filter_value
            ]

        st.dataframe(
            report,
            hide_index=True,
            width="stretch"
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
            st.success(
                "선택 데이터의 열 이름·순서·셀 값이 원본과 일치합니다."
            )
        else:
            st.warning(
                "선택 데이터와 원본 사이에 차이가 있습니다."
            )
    else:
        st.warning("원본 46열 CSV를 찾지 못했습니다.")

    st.subheader("모델 학습용 전처리")

    st.write(
        "매체 이용 경험과 타깃의 응답은 분석용 데이터에서만 "
        "이용함=1, 이용 안함=0으로 바꾸었습니다."
    )

    st.write(
        "학교급과 학년은 범주형 변수로 사용하며, "
        "모델 학습 시 원-핫 인코딩을 적용합니다. "
        "이 인코딩은 학습 데이터에서만 학습합니다."
    )

    st.info(
        "응답자 ID와 타깃 Y는 입력 X에 포함하지 않습니다. "
        "원본 CSV는 수정하지 않았습니다."
    )


# ==========================================
# 메뉴 3. 데이터 분포 시각화
# ==========================================

elif menu == "③ 데이터 분포 시각화":

    st.header("데이터 속성별 분포")

    st.subheader("① 매체별 이용률")

    st.caption(
        "이진 범주형 데이터이므로 매체별 비율을 "
        "비교하기 쉬운 가로 막대그래프를 사용합니다."
    )

    rates = (X_media.mean() * 100).sort_values(
        ascending=False
    )

    rate_df = rates.reset_index()
    rate_df.columns = ["매체", "이용률"]

    chart = alt.Chart(rate_df).mark_bar().encode(
        x=alt.X(
            "이용률:Q",
            scale=alt.Scale(domain=[0, 100])
        ),
        y=alt.Y("매체:N", sort="-x"),
        tooltip=["매체", alt.Tooltip("이용률:Q", format=".2f")]
    ).properties(height=380)

    show_chart(chart)

    top = rate_df.iloc[0]
    bottom = rate_df.iloc[-1]

    st.write(
        f"가장 이용률이 높은 매체는 **{top['매체']}** "
        f"({top['이용률']:.1f}%)이고, "
        f"가장 낮은 매체는 **{bottom['매체']}** "
        f"({bottom['이용률']:.1f}%)입니다."
    )

    st.subheader("② 생성형 AI 이용 여부 분포")

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
        f"({Y.mean() * 100:.1f}%)이고, "
        f"미이용자는 {(Y == 0).sum():,}명입니다."
    )

    st.subheader("③ 학교급별 응답자 수")

    school_counts = school_labels.value_counts().reindex(
        list(SCHOOL_LABELS.values()), fill_value=0
    )

    st.bar_chart(school_counts)

    st.subheader("④ 학년별 응답자 수")

    grade_counts = grade_labels.value_counts().reindex(
        list(GRADE_LABELS.values()), fill_value=0
    )

    st.bar_chart(grade_counts)

    st.caption(
        "학년 코드는 초4, 초5, 초6, 중1, 중2, 중3, "
        "고1, 고2, 고3으로 표시합니다."
    )

    st.info(
        "매체별 이용률은 입력 데이터의 분포를 보여줍니다. "
        "매체 이용률이 높다고 해당 변수의 분류 성능이 "
        "반드시 높은 것은 아닙니다."
    )


# ==========================================
# 메뉴 4. X와 Y의 관계 분석
# ==========================================

elif menu == "④ 입력 변수와 타깃의 관계":

    st.header("입력 변수(X)와 타깃(Y)의 관계")

    st.write(
        "각 디지털 매체 이용 경험과 생성형 AI 이용 여부가 "
        "어느 정도 관련되는지 확인합니다."
    )

    st.subheader("① 매체별 상관계수")

    st.caption(
        "두 이진 변수의 관계 방향과 크기를 "
        "비교하기 위해 상관계수 막대그래프를 사용합니다."
    )

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

    corr_chart = alt.Chart(corr_df).mark_bar().encode(
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
    ).properties(height=370)

    show_chart(corr_chart)

    strongest = corr_df.iloc[0]

    st.write(
        f"상관계수 절댓값이 가장 큰 매체는 "
        f"**{strongest['매체']}**이며, "
        f"상관계수는 {strongest['상관계수']:.3f}입니다."
    )

    st.subheader("② AI 이용자와 미이용자 비교")

    st.caption(
        "두 집단의 매체 이용률 차이를 비교하기 위해 "
        "그룹 막대그래프를 사용합니다."
    )

    rows = []
    differences = []

    for col in X_media.columns:

        yes_rate = X_media.loc[Y == 1, col].mean() * 100
        no_rate = X_media.loc[Y == 0, col].mean() * 100

        rows.extend([
            {
                "매체": col,
                "집단": "AI 이용자",
                "이용률": yes_rate
            },
            {
                "매체": col,
                "집단": "AI 미이용자",
                "이용률": no_rate
            }
        ])

        differences.append({
            "매체": col,
            "이용률 차이(%p)": yes_rate - no_rate
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
            tooltip=["매체", "집단", "이용률"]
        ).properties(height=430)
    )

    diff_df = pd.DataFrame(differences)
    diff_df["절댓값"] = diff_df["이용률 차이(%p)"].abs()
    diff_df = diff_df.sort_values(
        "절댓값", ascending=False
    )

    st.dataframe(
        diff_df.drop(columns="절댓값").round(2),
        hide_index=True,
        width="stretch"
    )

    biggest = diff_df.iloc[0]

    st.write(
        f"AI 이용자와 미이용자의 매체 이용률 차이가 "
        f"가장 큰 매체는 **{biggest['매체']}**이며, "
        f"차이는 {abs(biggest['이용률 차이(%p)']):.1f}%p입니다."
    )

    st.subheader("③ 학교급별 AI 이용률")

    school_df = pd.DataFrame({
        "학교급": school_labels,
        "AI 이용": Y
    })

    school_rate = (
        school_df.groupby("학교급")["AI 이용"]
        .mean().mul(100)
        .reindex(list(SCHOOL_LABELS.values()))
        .reset_index(name="AI 이용률")
    )

    show_chart(
        alt.Chart(school_rate).mark_bar().encode(
            x=alt.X(
                "학교급:N",
                sort=list(SCHOOL_LABELS.values())
            ),
            y=alt.Y(
                "AI 이용률:Q",
                scale=alt.Scale(domain=[0, 100])
            ),
            tooltip=["학교급", "AI 이용률"]
        ).properties(height=300)
    )

    st.subheader("④ 학년별 AI 이용률")

    grade_df = pd.DataFrame({
        "학년": grade_labels,
        "AI 이용": Y
    })

    grade_rate = (
        grade_df.groupby("학년")["AI 이용"]
        .mean().mul(100)
        .reindex(list(GRADE_LABELS.values()))
        .reset_index(name="AI 이용률")
    )

    show_chart(
        alt.Chart(grade_rate).mark_bar().encode(
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
        "입력 변수와 타깃 사이의 관련성은 확인했지만 "
        "인과관계나 실제 분류 성능을 증명한 것은 아닙니다. "
        "다음 단계에서 세 분류 모델로 예측 성능을 평가합니다."
    )


# ==========================================
# 메뉴 5. 분류 모델 학습
# ==========================================

elif menu == "⑤ 분류 모델 학습":

    st.header("분류 모델 3개 학습")

    st.write(
        "로지스틱 회귀, 의사결정트리, 랜덤 포레스트를 "
        "같은 데이터 조건에서 학습하여 비교합니다."
    )

    st.subheader("① 모델 선정 이유")

    model_info = pd.DataFrame([
        {
            "모델": "로지스틱 회귀",
            "작동 원리": "입력 변수에 가중치를 적용해 AI 이용 확률 계산",
            "선정 이유": "해석 가능한 기본 비교 모델"
        },
        {
            "모델": "의사결정트리",
            "작동 원리": "매체 이용 여부 등의 조건을 따라 데이터를 분할",
            "선정 이유": "분류 과정을 시각적으로 설명 가능"
        },
        {
            "모델": "랜덤 포레스트",
            "작동 원리": "여러 의사결정트리의 예측을 종합",
            "선정 이유": "복잡한 패턴 학습과 과적합 완화 가능"
        }
    ])

    st.dataframe(
        model_info,
        hide_index=True,
        width="stretch"
    )

    st.subheader("② 모델 학습 방법")

    st.write(
        "전체 데이터의 60%를 학습, 20%를 검증, "
        "20%를 최종 평가에 사용합니다."
    )

    st.write(
        "세 모델 모두 동일한 변수, 동일한 학습 데이터, "
        "동일한 검증 데이터를 사용합니다."
    )

    trained = train_all_models(X, Y.astype(int))

    a, b, c = st.columns(3)
    a.metric("학습 데이터", trained["counts"]["학습"])
    b.metric("검증 데이터", trained["counts"]["검증"])
    c.metric(
        "최종 평가 데이터",
        trained["counts"]["최종 평가"]
    )

    st.info(
        "검증 데이터로 모델을 비교하고, "
        "최종 평가 데이터는 모델 선정 이후 "
        "선택된 모델의 성능을 확인할 때만 사용합니다."
    )

    st.subheader("③ 모델별 작동 원리와 학습 결과")

    selected_model = st.selectbox(
        "살펴볼 모델",
        MODEL_NAMES
    )

    result = trained["results"][selected_model]
    pipe = result["모델"]
    algorithm = pipe.named_steps["model"]

    if selected_model == "로지스틱 회귀":

        st.markdown("### 로지스틱 회귀")

        st.write(
            "**작동 원리:** 입력 변수에 가중치를 적용해 "
            "생성형 AI를 이용할 확률을 계산합니다. "
            "확률이 0.5 이상이면 '이용함', "
            "미만이면 '이용 안함'으로 분류합니다."
        )

        st.write(
            "**선정 이유:** 계산 방식이 비교적 단순하고 "
            "다른 모델과 비교하는 기본 기준으로 적합합니다."
        )

        st.write(
            "**이 모델의 특징:** 학습된 계수를 통해 "
            "각 입력 항목이 예측 확률과 어떤 방향으로 "
            "관련되는지 살펴볼 수 있습니다."
        )

        features = pipe.named_steps[
            "preprocessing"
        ].get_feature_names_out()

        coefficients = algorithm.coef_[0]

        coef_df = pd.DataFrame({
            "인코딩된 항목": features,
            "계수": coefficients
        })

        coef_df["절댓값"] = coef_df["계수"].abs()
        coef_df = coef_df.sort_values(
            "절댓값", ascending=False
        ).head(15)

        show_chart(
            alt.Chart(coef_df).mark_bar().encode(
                x="계수:Q",
                y=alt.Y(
                    "인코딩된 항목:N",
                    sort="-x"
                ),
                tooltip=["인코딩된 항목", "계수"]
            ).properties(height=430)
        )

        st.caption(
            "계수는 인코딩된 항목의 영향 방향을 살펴보는 "
            "참고 자료입니다. 인과관계를 뜻하지 않습니다."
        )

    elif selected_model == "의사결정트리":

        st.markdown("### 의사결정트리")

        st.write(
            "**작동 원리:** SNS 이용 여부, 학년 등 "
            "조건을 기준으로 데이터를 반복해서 나눕니다. "
            "마지막에 도달한 그룹의 이용 비율을 바탕으로 "
            "이용 여부를 예측합니다."
        )

        st.write(
            "**선정 이유:** 어떤 조건으로 데이터를 "
            "분류하는지 직접 살펴볼 수 있어 "
            "설명하기 쉬운 모델입니다."
        )

        st.write(
            "**이 모델의 특징:** 하나의 나무 구조로 "
            "학습하기 때문에 복잡한 패턴을 학습할 수 있지만, "
            "과적합이 발생할 수 있습니다."
        )

        features = pipe.named_steps[
            "preprocessing"
        ].get_feature_names_out()

        tree_description = export_text(
            algorithm,
            feature_names=list(features),
            max_depth=3
        )

        st.subheader("학습된 의사결정트리의 분류 조건")

        st.code(tree_description, language="text")

        st.caption(
            "조건을 읽기 쉽게 하기 위해 트리의 앞부분만 "
            "표시합니다. 전체 학습 깊이는 최대 5입니다."
        )

    else:

        st.markdown("### 랜덤 포레스트")

        st.write(
            "**작동 원리:** 여러 개의 의사결정트리를 "
            "서로 다른 표본과 변수 선택을 활용해 학습합니다. "
            "각 트리의 예측을 종합해 최종 결과를 결정합니다."
        )

        st.write(
            "**선정 이유:** 여러 트리를 결합하여 "
            "단일 의사결정트리보다 과적합을 줄이고 "
            "안정적인 예측을 할 가능성이 있습니다."
        )

        st.write(
            "**이 모델의 특징:** 단일 트리보다 구조가 "
            "복잡하지만, 변수 중요도를 확인할 수 있습니다."
        )

        features = pipe.named_steps[
            "preprocessing"
        ].get_feature_names_out()

        importances = algorithm.feature_importances_

        importance_df = pd.DataFrame({
            "인코딩된 항목": features,
            "중요도": importances
        })

        importance_df = importance_df.sort_values(
            "중요도", ascending=False
        ).head(15)

        show_chart(
            alt.Chart(importance_df).mark_bar().encode(
                x="중요도:Q",
                y=alt.Y(
                    "인코딩된 항목:N",
                    sort="-x"
                ),
                tooltip=["인코딩된 항목", "중요도"]
            ).properties(height=430)
        )

        st.caption(
            "변수 중요도는 학습된 모델에서 해당 항목이 "
            "분할에 기여한 정도입니다. "
            "높다고 해서 인과관계가 있다는 의미는 아닙니다."
        )

    st.subheader("④ 선택한 모델의 검증 결과")

    scores = result["점수"]

    cols = st.columns(5)

    for col, (metric, value) in zip(
        cols, scores.items()
    ):
        col.metric(metric, f"{value:.3f}")

    st.write(
        f"**학습 결과:** {selected_model}의 "
        f"검증 데이터 정확도는 {scores['정확도']:.3f}, "
        f"F1 점수는 {scores['F1']:.3f}입니다."
    )

    st.caption(
        "이 점수는 모델을 비교하기 위한 검증 데이터의 결과입니다. "
        "최종 평가 결과는 ⑥ 메뉴에서 확인합니다."
    )


# ==========================================
# 메뉴 6. 모델 성능 비교 및 선정
# ==========================================

elif menu == "⑥ 모델 성능 비교 및 선정":

    st.header("분류 모델 성능 비교 및 선정")

    trained = train_all_models(X, Y.astype(int))

    results = trained["results"]
    best_name = trained["best_name"]

    summary = score_table(results)

    st.subheader("① 세 모델의 검증 성능 비교")

    st.write(
        "세 모델의 검증 데이터 성능을 동일한 평가 지표로 "
        "비교합니다. 아래는 실제 학습 결과입니다."
    )

    st.dataframe(
        summary.round(3),
        hide_index=True,
        width="stretch"
    )

    st.subheader("② 성능 비교 그래프")

    metric_name = st.selectbox(
        "비교할 성능 지표",
        [
            "정확도",
            "정밀도",
            "재현율",
            "F1",
            "ROC-AUC"
        ],
        index=3
    )

    compare_chart = alt.Chart(summary).mark_bar().encode(
        x=alt.X(
            "모델:N",
            sort=MODEL_NAMES
        ),
        y=alt.Y(
            f"{metric_name}:Q",
            scale=alt.Scale(domain=[0, 1])
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

    show_chart(compare_chart)

    st.subheader("③ 평가 지표의 의미")

    explanations = {
        "정확도": "전체 응답자 중 올바르게 분류한 비율",
        "정밀도": "AI 이용자라고 예측한 사람 중 실제 이용자 비율",
        "재현율": "실제 AI 이용자 중 이용자로 맞힌 비율",
        "F1": "정밀도와 재현율의 균형을 나타내는 점수",
        "ROC-AUC": "다양한 분류 기준에서 양성과 음성을 구별하는 능력"
    }

    st.info(
        f"**{metric_name}:** {explanations[metric_name]}"
    )

    st.subheader("④ 검증 데이터의 혼동행렬")

    selected = st.selectbox(
        "혼동행렬을 확인할 모델",
        MODEL_NAMES
    )

    pred = results[selected]["예측값"]
    valid_y = trained["y_valid"]

    cm = confusion_matrix(
        valid_y,
        pred,
        labels=[0, 1]
    )

    labels = ["이용 안함", "이용함"]

    cm_rows = []

    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            cm_rows.append({
                "실제": actual,
                "예측": predicted,
                "응답자 수": int(cm[i, j])
            })

    cm_df = pd.DataFrame(cm_rows)

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
        color="응답자 수:Q",
        tooltip=["실제", "예측", "응답자 수"]
    )

    numbers = base.mark_text(
        fontSize=20
    ).encode(
        text="응답자 수:Q",
        color=alt.value("white")
    )

    show_chart(
        (rectangles + numbers).properties(height=270)
    )

    st.write(
        "혼동행렬에서 대각선에 해당하는 두 칸은 "
        "올바르게 분류한 응답자 수이며, "
        "나머지 두 칸은 잘못 분류한 응답자 수입니다."
    )

    st.subheader("⑤ 검증 데이터의 ROC 곡선")

    roc_rows = []

    for name, result in results.items():

        fpr, tpr, _ = roc_curve(
            valid_y,
            result["확률"]
        )

        for a, b in zip(fpr, tpr):
            roc_rows.append({
                "모델": name,
                "거짓양성률": float(a),
                "참양성률": float(b)
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

    st.subheader("⑥ 최종 모델 선정")

    ranking = summary.sort_values(
        ["F1", "ROC-AUC", "정확도"],
        ascending=False
    )

    st.dataframe(
        ranking.round(3),
        hide_index=True,
        width="stretch"
    )

    best_row = ranking.iloc[0]

    st.success(
        f"검증 데이터에서 가장 높은 F1 점수를 기록한 모델은 "
        f"**{best_name}**입니다."
    )

    st.write(
        f"이 모델의 검증 정확도는 "
        f"{best_row['정확도']:.3f}, "
        f"F1은 {best_row['F1']:.3f}, "
        f"ROC-AUC는 {best_row['ROC-AUC']:.3f}입니다."
    )

    st.write(
        "이번 탐구에서는 정확도만 비교하지 않고, "
        "AI 이용자를 정확하게 찾는 정밀도와 재현율을 "
        "종합한 F1 점수를 주요 선정 기준으로 사용합니다."
    )

    st.subheader("⑦ 선정된 모델의 최종 평가")

    st.write(
        "검증 데이터로 모델을 선택한 뒤, "
        "선택된 모델을 학습·검증 데이터 전체로 다시 학습하고 "
        "별도로 보관한 최종 평가 데이터에서 성능을 확인합니다."
    )

    final_scores = trained["final_scores"]

    cols = st.columns(5)

    for col, (metric, value) in zip(
        cols, final_scores.items()
    ):
        col.metric(metric, f"{value:.3f}")

    st.write(
        f"**최종 평가 결과:** {best_name}의 "
        f"정확도는 {final_scores['정확도']:.3f}, "
        f"F1 점수는 {final_scores['F1']:.3f}입니다."
    )

    st.subheader("⑧ 탐구 결과 해석")

    st.write(
        "본 탐구에서는 청소년의 디지털 매체 이용 경험과 "
        "학교급, 학년을 이용하여 생성형 AI 이용 여부를 "
        "예측하는 이진 분류 모델을 학습하였습니다."
    )

    st.write(
        "서로 다른 원리로 작동하는 로지스틱 회귀, "
        "의사결정트리, 랜덤 포레스트의 성능을 비교하고 "
        "검증 데이터의 F1 점수를 기준으로 모델을 선정했습니다."
    )

    st.write(
        "다만 이번 결과는 해당 설문 데이터에 대한 평가이며, "
        "다른 조사 시기나 학생 집단에서 동일한 성능이 "
        "나온다고 단정할 수는 없습니다."
    )

    st.info(
        "매체 이용 경험과 생성형 AI 이용 여부 사이의 "
        "관련성이 나타나더라도 인과관계가 증명되는 것은 아닙니다. "
        "또한 설문조사 가중치를 반영하지 않은 분석이므로 "
        "결과를 전체 청소년 집단의 비율로 일반화할 때는 "
        "주의가 필요합니다."
    )
