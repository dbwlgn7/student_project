
from pathlib import Path

import pandas as pd
import altair as alt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
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


# ==============================================
# 1. 기본 설정
# ==============================================

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

MEDIA = {
    3: "TV방송",
    4: "OTT",
    5: "인터넷 개인방송",
    6: "숏폼",
    7: "포털 플랫폼",
    9: "SNS",
    11: "메타버스",
    13: "온라인 게임",
    14: "웹툰"
}

SCHOOL = {
    1: "초등학교",
    2: "중학교",
    3: "고등학교"
}

GRADE = {
    1: "초4",
    2: "초5",
    3: "초6",
    4: "중1",
    5: "중2",
    6: "중3",
    7: "고1",
    8: "고2",
    9: "고3"
}

MODEL_NAMES = [
    "로지스틱 회귀",
    "의사결정트리",
    "랜덤 포레스트"
]


# ==============================================
# 2. 공통 함수
# ==============================================

@st.cache_data
def read_csv(path):
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False
    )


def show_chart(obj):
    st.altair_chart(obj, width="stretch")


def get_reason(n):

    if n == 1:
        return "응답자 식별 및 중복 확인에 사용하며 모델 입력에서는 제외한다."

    if n == 12:
        return "생성형 AI 이용 여부는 예측 대상 Y이므로 유지한다."

    if n in [36, 40]:
        return "학교급과 학년에 따른 차이를 분석하기 위해 유지한다."

    if n in KEEP:
        return "디지털 매체 이용 경험을 나타내는 입력 변수 X이다."

    if n == 2:
        return "학교 식별 번호로 이용 특성을 직접 나타내지 않는다."

    if n in [8, 10, 15, 16]:
        return "관련된 디지털 매체지만 탐구 범위를 제한하기 위해 제외한다."

    if n in [17, 18]:
        return "종이 매체로 디지털 매체 중심 분석에서 제외한다."

    if n in [19, 20, 21]:
        return "생성형 AI 정보가 직접 포함될 수 있어 정답 정보 누출을 방지한다."

    if 22 <= n <= 29:
        return "온라인 안전 및 개인정보 보호 경험에 해당한다."

    if 30 <= n <= 33:
        return "교육 경험과 교육 효과에 관한 변수이다."

    if n in [34, 35]:
        return "개인·학교 배경 특성으로 이번 탐구 범위에서 제외한다."

    if n in [37, 38, 39, 43, 44]:
        return "학교급·학년의 추가 정보로 기본 변수만 사용한다."

    if n in [41, 42]:
        return "지역 관련 정보로 이번 분석에서 제외한다."

    if n in [45, 46]:
        return "설문조사 가중치로 모델 입력에서는 제외한다."

    return "탐구 범위를 고려하여 제외한다."


def create_models():

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


def create_pipeline(model, columns):

    encoder = OneHotEncoder(
        drop="if_binary",
        handle_unknown="ignore"
    )

    preprocessing = ColumnTransformer([
        (
            "category",
            encoder,
            list(columns)
        )
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", model)
    ])


def calculate_scores(y_true, pred, prob):

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

        "ROC-AUC": roc_auc_score(
            y_true, prob
        )
    }


def feature_names(pipe):

    raw_names = pipe.named_steps[
        "preprocessing"
    ].get_feature_names_out()

    return [
        str(name).replace("category__", "")
        for name in raw_names
    ]


def readable_name(name):

    name = str(name).replace("category__", "")

    if name.startswith("학교급_"):
        return "학교급: " + name.split("_", 1)[1]

    if name.startswith("학년_"):
        return "학년: " + name.split("_", 1)[1]

    if name.endswith("_1"):
        return name[:-2] + " 이용함"

    if name.endswith("_0"):
        return name[:-2] + " 이용 안함"

    return name.replace("_", " ")


def group_name(name):

    name = str(name).replace("category__", "")

    if name.startswith("학교급_"):
        return "학교급"

    if name.startswith("학년_"):
        return "학년"

    for item in MEDIA.values():
        if name.startswith(item + "_"):
            return item

    return name


# ==============================================
# 3. 모델 학습
# ==============================================

@st.cache_resource(
    show_spinner="세 분류 모델을 학습하고 있습니다."
)
def train_models(X, y):

    # 학습 60%, 검증 20%, 최종 평가 20%
    X_build, X_test, y_build, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_build,
        y_build,
        test_size=0.25,
        random_state=42,
        stratify=y_build
    )

    results = {}

    for name, model in create_models().items():

        pipe = create_pipeline(model, X.columns)
        pipe.fit(X_train, y_train)

        predictions = pipe.predict(X_valid)
        probabilities = pipe.predict_proba(X_valid)[:, 1]

        results[name] = {
            "pipeline": pipe,
            "pred": predictions,
            "prob": probabilities,
            "scores": calculate_scores(
                y_valid,
                predictions,
                probabilities
            )
        }

    best_name = max(
        results,
        key=lambda name: (
            results[name]["scores"]["F1"],
            results[name]["scores"]["ROC-AUC"],
            results[name]["scores"]["정확도"]
        )
    )

    # 선정한 모델만 학습+검증 데이터로 재학습
    final_pipeline = create_pipeline(
        create_models()[best_name],
        X.columns
    )

    final_pipeline.fit(X_build, y_build)

    final_pred = final_pipeline.predict(X_test)
    final_prob = final_pipeline.predict_proba(X_test)[:, 1]

    final_scores = calculate_scores(
        y_test,
        final_pred,
        final_prob
    )

    return {
        "results": results,
        "best_name": best_name,
        "final_scores": final_scores,
        "valid_y": y_valid,
        "counts": {
            "학습": len(X_train),
            "검증": len(X_valid),
            "평가": len(X_test)
        }
    }


# ==============================================
# 4. 의사결정트리 전용 함수
# ==============================================

def node_information(tree, node):

    values = tree.value[node][0].astype(float)

    total = values.sum()

    ai_ratio = (
        values[1] / total
        if total > 0
        else 0
    )

    sample_count = int(tree.n_node_samples[node])

    prediction = (
        "AI 이용함"
        if ai_ratio >= 0.5
        else "AI 이용 안함"
    )

    return {
        "응답자 수": sample_count,
        "AI 이용률": ai_ratio * 100,
        "예측 결과": prediction
    }


def node_condition(tree, node, names):

    feature_idx = tree.feature[node]

    if feature_idx < 0:
        return None

    name = names[feature_idx]

    threshold = tree.threshold[node]

    if name.startswith("학교급_"):

        value = name.split("_", 1)[1]
        question = f"학교급이 '{value}'인가요?"

    elif name.startswith("학년_"):

        value = name.split("_", 1)[1]
        question = f"학년이 '{value}'인가요?"

    elif name.endswith("_1"):

        question = f"'{name[:-2]}'을 이용하나요?"

    elif name.endswith("_0"):

        question = f"'{name[:-2]}'을 이용하지 않나요?"

    else:

        question = f"'{readable_name(name)}'에 해당하나요?"

    return question, threshold


def get_tree_paths(tree, max_depth=5):

    paths = []

    def visit(node, path, depth):

        paths.append((node, path, depth))

        if depth >= max_depth:
            return

        left = tree.children_left[node]
        right = tree.children_right[node]

        if left >= 0:
            visit(
                left,
                path + " → 아니요",
                depth + 1
            )

        if right >= 0:
            visit(
                right,
                path + " → 예",
                depth + 1
            )

    visit(0, "시작", 0)

    return paths


def show_node_card(title, info):

    with st.container(border=True):

        st.markdown(f"**{title}**")

        a, b = st.columns(2)

        a.metric(
            "학습 응답자 수",
            f"{info['응답자 수']:,}명"
        )

        b.metric(
            "AI 이용률",
            f"{info['AI 이용률']:.1f}%"
        )

        st.write(
            f"현재 예측: **{info['예측 결과']}**"
        )


def show_interactive_tree(pipe):

    model = pipe.named_steps["model"]
    tree = model.tree_
    names = feature_names(pipe)

    st.subheader("실제 학습된 분류 과정 살펴보기")

    # ======================================
    # 이번에 추가한 사용 방법 설명
    # ======================================

    with st.expander(
        "📖 의사결정트리 사용 방법",
        expanded=True
    ):

        st.write(
            "이 모델은 질문에 따라 학생들을 두 집단으로 "
            "나누면서 AI 이용 여부를 예측합니다."
        )

        st.write(
            "**시작(단계 0):** 전체 학습 학생을 대상으로 "
            "첫 번째 질문을 확인합니다."
        )

        st.write(
            "**→ 아니요:** 질문의 조건에 해당하지 않는 "
            "학생들의 분류 경로입니다."
        )

        st.write(
            "**→ 예:** 질문의 조건에 해당하는 "
            "학생들의 분류 경로입니다."
        )

        st.write(
            "**단계 1, 2, 3…:** 숫자가 커질수록 "
            "더 세부적인 조건으로 분류됩니다."
        )

        st.write(
            "**마지막 단계:** 더 이상 나누지 않고 "
            "해당 집단의 AI 이용 여부를 최종 예측합니다."
        )

        st.caption(
            "아래 '확인할 분류 단계'에서 원하는 경로를 "
            "선택하면 해당 학생들의 수와 AI 이용률을 "
            "확인할 수 있습니다."
        )

    # ======================================
    # 실제 단계 선택
    # ======================================

    paths = get_tree_paths(tree)

    options = {
        f"{'　' * depth}{path} (단계 {depth})": node
        for node, path, depth in paths
    }

    selected_path = st.selectbox(
        "확인할 분류 단계",
        list(options.keys())
    )

    node = options[selected_path]

    info = node_information(tree, node)

    st.markdown("#### 현재 단계")

    show_node_card(
        "이 단계에 포함된 학생",
        info
    )

    left_node = tree.children_left[node]
    right_node = tree.children_right[node]

    # 마지막 노드
    if left_node < 0:

        st.success(
            "마지막 분류 단계입니다. "
            f"이 집단의 최종 예측은 "
            f"**{info['예측 결과']}**입니다."
        )

        st.write(
            f"이 집단에 속한 학습 응답자는 "
            f"{info['응답자 수']:,}명이며, "
            f"생성형 AI 이용률은 "
            f"{info['AI 이용률']:.1f}%입니다."
        )

        return

    # 현재 분류 질문
    question, threshold = node_condition(
        tree,
        node,
        names
    )

    st.markdown("#### 이번 단계의 분류 질문")

    st.info(f"**{question}**")

    left_info = node_information(
        tree,
        left_node
    )

    right_info = node_information(
        tree,
        right_node
    )

    a, b = st.columns(2)

    with a:

        show_node_card(
            "아니요 — 조건에 해당하지 않는 집단",
            left_info
        )

    with b:

        show_node_card(
            "예 — 조건에 해당하는 집단",
            right_info
        )

    st.markdown("#### 이 단계에서 알 수 있는 사실")

    difference = (
        right_info["AI 이용률"]
        - left_info["AI 이용률"]
    )

    st.write(
        f"'예' 집단의 AI 이용률은 "
        f"**{right_info['AI 이용률']:.1f}%**, "
        f"'아니요' 집단은 "
        f"**{left_info['AI 이용률']:.1f}%**입니다."
    )

    if difference > 0:

        st.write(
            f"'예' 집단의 이용률이 "
            f"{difference:.1f}%p 더 높습니다."
        )

    elif difference < 0:

        st.write(
            f"'아니요' 집단의 이용률이 "
            f"{abs(difference):.1f}%p 더 높습니다."
        )

    else:

        st.write(
            "두 집단의 AI 이용률이 같습니다."
        )

    st.caption(
        f"실제 모델의 분기 기준값은 "
        f"{threshold:.2f}입니다. "
        "왼쪽은 기준 이하, 오른쪽은 기준 초과입니다."
    )

    st.warning(
        "현재 단계의 예측은 해당 집단의 다수 범주입니다. "
        "다음 분류 단계에서 결과가 달라질 수 있습니다."
    )


# ==============================================
# 5. 사이드바
# ==============================================

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


# ==============================================
# 6. 데이터 불러오기
# ==============================================

if uploaded is not None:

    df = pd.read_csv(
        uploaded,
        encoding="utf-8-sig"
    )

elif SELECTED_FILE.exists():

    df = read_csv(str(SELECTED_FILE))

else:

    st.error("13열 CSV 파일을 찾을 수 없습니다.")
    st.stop()


if df.shape[1] != 13:

    st.error(
        "13개 열로 구성된 CSV 파일이 필요합니다."
    )
    st.stop()


id_col = df.columns[0]
target_col = df.columns[8]
school_col = df.columns[11]
grade_col = df.columns[12]

media_cols = (
    list(df.columns[1:8])
    + list(df.columns[9:11])
)

if (
    "응답자 ID" not in str(id_col)
    or "생성형 AI" not in str(target_col)
    or "학교급" not in str(school_col)
    or "학년별" not in str(grade_col)
):

    st.error(
        "CSV의 열 구성이나 순서가 예상과 다릅니다."
    )
    st.stop()


X_media = pd.DataFrame(index=df.index)

for name, col in zip(MEDIA.values(), media_cols):

    X_media[name] = pd.to_numeric(
        df[col],
        errors="coerce"
    ).map({
        1: 1,
        2: 0
    })


Y = pd.to_numeric(
    df[target_col],
    errors="coerce"
).map({
    1: 1,
    2: 0
})


school_labels = pd.to_numeric(
    df[school_col],
    errors="coerce"
).map(SCHOOL)


grade_labels = pd.to_numeric(
    df[grade_col],
    errors="coerce"
).map(GRADE)


if (
    X_media.isna().any().any()
    or Y.isna().any()
    or school_labels.isna().any()
    or grade_labels.isna().any()
):

    st.error(
        "결측치 또는 예상 범위를 벗어난 코드가 있습니다."
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


# ==============================================
# ① 탐구 소개
# ==============================================

if menu.startswith("①"):

    st.header("탐구 소개")

    a, b, c = st.columns(3)

    a.metric("응답자 수", f"{len(df):,}명")
    b.metric("선택 변수", 13)
    c.metric("모델 입력 변수", 11)

    st.subheader("왜 분류를 선택했을까?")

    methods = pd.DataFrame([
        ["회귀", "연속적인 수치 예측", "선택하지 않음"],
        ["분류", "AI 이용함 / 이용 안함 예측", "선택"],
        ["군집", "정답 없이 비슷한 집단 구성", "선택하지 않음"]
    ], columns=[
        "방법", "목적", "적용"
    ])

    st.dataframe(
        methods,
        hide_index=True,
        width="stretch"
    )

    st.success(
        "생성형 AI 이용 여부는 두 범주 중 하나를 "
        "예측하는 문제이므로 이진 분류가 적합합니다."
    )

    st.subheader("데이터 속성과 유형")

    types = pd.DataFrame([
        ["응답자 ID", "식별자", "중복 확인", "제외"],
        ["디지털 매체 이용 경험 9개", "이진 범주형", "입력 X", "포함"],
        ["학교급", "순서형 범주", "입력 X", "포함"],
        ["학년", "순서형 범주", "입력 X", "포함"],
        ["생성형 AI 이용 여부", "이진 범주형", "타깃 Y", "제외"]
    ], columns=[
        "속성", "데이터 유형", "역할", "모델 입력"
    ])

    st.dataframe(
        types,
        hide_index=True,
        width="stretch"
    )

    st.subheader("비교할 분류 모델")

    st.write(
        "**로지스틱 회귀:** 입력 변수로 이용 확률을 계산하는 모델"
    )

    st.write(
        "**의사결정트리:** 조건에 따라 데이터를 나누는 모델"
    )

    st.write(
        "**랜덤 포레스트:** 여러 의사결정트리의 예측을 종합하는 모델"
    )


# ==============================================
# ② 전처리 및 근거
# ==============================================

elif menu.startswith("②"):

    st.header("데이터 전처리 및 근거")

    a, b, c = st.columns(3)

    a.metric("원본 변수", 46)
    b.metric("선택 변수", 13)
    c.metric("제외 변수", 33)

    a, b, c = st.columns(3)

    a.metric("응답자", f"{len(df):,}")
    b.metric("결측치", int(df.isna().sum().sum()))
    c.metric(
        "중복 ID",
        int(df[id_col].duplicated().sum())
    )

    st.write(
        "원본 46개 열 가운데 탐구에 필요한 13개를 "
        "선택했으며 원본 응답 코드는 변경하지 않았습니다."
    )

    if ORIGINAL_FILE.exists():

        original = read_csv(str(ORIGINAL_FILE))

        report = pd.DataFrame([
            {
                "번호": i,
                "변수명": col,
                "처리": "유지" if i in KEEP else "제외",
                "근거": get_reason(i)
            }
            for i, col in enumerate(
                original.columns,
                start=1
            )
        ])

        choice = st.selectbox(
            "확인할 변수",
            ["전체", "유지", "제외"]
        )

        if choice != "전체":

            report = report[
                report["처리"] == choice
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
            expected.columns.tolist()
            == actual.columns.tolist()
            and expected.astype("string")
            .fillna("__NA__")
            .equals(
                actual.astype("string")
                .fillna("__NA__")
            )
        )

        if same:

            st.success(
                "원본과 선택 데이터의 열 이름, 순서, 값이 일치합니다."
            )

        else:

            st.warning(
                "원본과 선택 데이터가 일치하지 않습니다."
            )

    else:

        st.warning(
            "원본 46열 CSV 파일을 찾지 못했습니다."
        )

    st.subheader("모델 학습용 전처리")

    st.write(
        "매체 이용 여부는 분석용 복사본에서 "
        "이용함=1, 이용 안함=0으로 변환했습니다."
    )

    st.write(
        "학교급과 학년을 포함한 범주형 입력 변수는 "
        "원-핫 인코딩을 적용합니다."
    )

    st.info(
        "응답자 ID와 정답 Y는 모델 입력 X에 포함하지 않습니다. "
        "인코딩은 학습 데이터에서만 학습합니다."
    )


# ==============================================
# ③ 데이터 분포 시각화
# ==============================================

elif menu.startswith("③"):

    st.header("데이터 분포 시각화")

    st.subheader("① 디지털 매체별 이용률")

    st.caption(
        "이진 범주형 변수의 이용률을 비교하기 위해 "
        "가로 막대그래프를 사용합니다."
    )

    rates = (
        X_media.mean() * 100
    ).sort_values(ascending=False)

    rate_df = rates.reset_index()

    rate_df.columns = [
        "매체",
        "이용률"
    ]

    show_chart(
        alt.Chart(rate_df).mark_bar().encode(
            x=alt.X(
                "이용률:Q",
                scale=alt.Scale(domain=[0, 100])
            ),
            y=alt.Y(
                "매체:N",
                sort="-x"
            ),
            tooltip=["매체", "이용률"]
        ).properties(height=380)
    )

    highest = rate_df.iloc[0]
    lowest = rate_df.iloc[-1]

    st.write(
        f"이용률이 가장 높은 매체는 "
        f"**{highest['매체']}** "
        f"({highest['이용률']:.1f}%)이며, "
        f"가장 낮은 매체는 "
        f"**{lowest['매체']}** "
        f"({lowest['이용률']:.1f}%)입니다."
    )

    st.subheader("② 생성형 AI 이용 여부")

    target_df = pd.DataFrame({
        "이용 여부": [
            "이용함",
            "이용 안함"
        ],
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
        f"생성형 AI 이용자는 "
        f"{int((Y == 1).sum()):,}명이며, "
        f"전체의 {Y.mean() * 100:.1f}%입니다."
    )

    st.subheader("③ 학교급별 응답자 수")

    school_counts = (
        school_labels.value_counts()
        .reindex(
            list(SCHOOL.values()),
            fill_value=0
        )
    )

    st.bar_chart(school_counts)

    st.subheader("④ 학년별 응답자 수")

    grade_counts = (
        grade_labels.value_counts()
        .reindex(
            list(GRADE.values()),
            fill_value=0
        )
    )

    st.bar_chart(grade_counts)

    st.caption(
        "학년은 초4, 초5, 초6, 중1, 중2, 중3, "
        "고1, 고2, 고3 순서로 표시합니다."
    )

    st.info(
        "매체별 이용률이 높다고 해서 "
        "분류 모델에서 반드시 중요한 것은 아닙니다."
    )


# ==============================================
# ④ 입력 변수와 타깃의 관계
# ==============================================

elif menu.startswith("④"):

    st.header("입력 변수와 타깃의 관계")

    st.write(
        "디지털 매체 이용 경험이 생성형 AI 이용 여부와 "
        "어떤 관련성을 나타내는지 살펴봅니다."
    )

    st.subheader("① 매체별 상관계수")

    corr_rows = []

    for col in X_media.columns:

        value = X_media[col].corr(Y)

        corr_rows.append({
            "매체": col,
            "상관계수": value,
            "절댓값": abs(value)
        })

    corr_df = pd.DataFrame(
        corr_rows
    ).sort_values(
        "절댓값",
        ascending=False
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
            tooltip=[
                "매체",
                "상관계수"
            ]
        ).properties(height=370)
    )

    top_corr = corr_df.iloc[0]

    st.write(
        f"상관계수의 절댓값이 가장 큰 매체는 "
        f"**{top_corr['매체']}**이며, "
        f"상관계수는 {top_corr['상관계수']:.3f}입니다."
    )

    st.subheader("② AI 이용자와 미이용자의 매체 이용률")

    rows = []
    differences = []

    for col in X_media.columns:

        yes_rate = (
            X_media.loc[Y == 1, col].mean() * 100
        )

        no_rate = (
            X_media.loc[Y == 0, col].mean() * 100
        )

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
            tooltip=[
                "매체",
                "집단",
                "이용률"
            ]
        ).properties(height=430)
    )

    diff_df = pd.DataFrame(differences)

    diff_df["절댓값"] = (
        diff_df["이용률 차이(%p)"].abs()
    )

    diff_df = diff_df.sort_values(
        "절댓값",
        ascending=False
    )

    st.dataframe(
        diff_df.drop(
            columns="절댓값"
        ).round(2),
        hide_index=True,
        width="stretch"
    )

    st.subheader("③ 학교급별 AI 이용률")

    school_rates = (
        pd.DataFrame({
            "학교급": school_labels,
            "AI 이용": Y
        })
        .groupby("학교급")["AI 이용"]
        .mean()
        .mul(100)
        .reindex(list(SCHOOL.values()))
        .reset_index(name="AI 이용률")
    )

    show_chart(
        alt.Chart(school_rates).mark_bar().encode(
            x=alt.X(
                "학교급:N",
                sort=list(SCHOOL.values())
            ),
            y=alt.Y(
                "AI 이용률:Q",
                scale=alt.Scale(domain=[0, 100])
            ),
            tooltip=[
                "학교급",
                "AI 이용률"
            ]
        ).properties(height=290)
    )

    st.subheader("④ 학년별 AI 이용률")

    grade_rates = (
        pd.DataFrame({
            "학년": grade_labels,
            "AI 이용": Y
        })
        .groupby("학년")["AI 이용"]
        .mean()
        .mul(100)
        .reindex(list(GRADE.values()))
        .reset_index(name="AI 이용률")
    )

    show_chart(
        alt.Chart(grade_rates).mark_bar().encode(
            x=alt.X(
                "학년:N",
                sort=list(GRADE.values())
            ),
            y=alt.Y(
                "AI 이용률:Q",
                scale=alt.Scale(domain=[0, 100])
            ),
            tooltip=[
                "학년",
                "AI 이용률"
            ]
        ).properties(height=310)
    )

    st.info(
        "상관관계는 변수 사이의 관련성을 나타낼 뿐, "
        "인과관계를 증명하지 않습니다."
    )


# ==============================================
# ⑤ 분류 모델 학습
# ==============================================

elif menu.startswith("⑤"):

    st.header("분류 모델 3개 학습")

    st.write(
        "세 가지 분류 모델을 동일한 데이터로 학습하고 "
        "각 모델의 작동 원리와 결과를 살펴봅니다."
    )

    st.subheader("① 모델의 작동 원리와 선정 이유")

    model_info = pd.DataFrame([
        [
            "로지스틱 회귀",
            "입력 변수로 AI 이용 확률을 계산",
            "기본 비교 모델"
        ],
        [
            "의사결정트리",
            "조건에 따라 데이터를 단계적으로 분할",
            "분류 과정 확인"
        ],
        [
            "랜덤 포레스트",
            "여러 의사결정트리의 예측을 종합",
            "복잡한 관계 학습"
        ]
    ], columns=[
        "모델",
        "작동 원리",
        "선정 이유"
    ])

    st.dataframe(
        model_info,
        hide_index=True,
        width="stretch"
    )

    trained = train_models(
        X,
        Y.astype(int)
    )

    st.subheader("② 학습 데이터 구성")

    a, b, c = st.columns(3)

    a.metric(
        "학습 데이터",
        f"{trained['counts']['학습']:,}명"
    )

    b.metric(
        "검증 데이터",
        f"{trained['counts']['검증']:,}명"
    )

    c.metric(
        "최종 평가 데이터",
        f"{trained['counts']['평가']:,}명"
    )

    st.caption(
        "전체 데이터의 60%를 학습, 20%를 검증, "
        "20%를 최종 평가에 사용합니다."
    )

    st.subheader("③ 모델별 학습 결과 확인")

    selected = st.selectbox(
        "살펴볼 모델",
        MODEL_NAMES
    )

    result = trained["results"][selected]

    pipe = result["pipeline"]

    model = pipe.named_steps["model"]

    # --------------------------------------
    # 로지스틱 회귀
    # --------------------------------------

    if selected == "로지스틱 회귀":

        st.markdown("### 로지스틱 회귀")

        st.write(
            "**작동 원리:** 입력 변수에 가중치를 적용해 "
            "생성형 AI 이용 확률을 계산합니다. "
            "확률이 0.5 이상이면 이용함으로 예측합니다."
        )

        st.write(
            "**선정 이유:** 다른 모델과 성능을 비교할 "
            "기본 기준으로 적합합니다."
        )

        names = feature_names(pipe)

        coef_df = pd.DataFrame({
            "항목": [
                readable_name(n)
                for n in names
            ],
            "원래 변수": [
                group_name(n)
                for n in names
            ],
            "계수": model.coef_[0]
        })

        coef_df["절댓값"] = (
            coef_df["계수"].abs()
        )

        coef_df = coef_df.sort_values(
            "절댓값",
            ascending=False
        )

        st.subheader("학습된 계수")

        show_chart(
            alt.Chart(coef_df).mark_bar().encode(
                x=alt.X(
                    "계수:Q",
                    title="계수"
                ),
                y=alt.Y(
                    "항목:N",
                    sort=list(coef_df["항목"]),
                    axis=alt.Axis(
                        labelLimit=350
                    ),
                    title=None
                ),
                tooltip=[
                    "항목",
                    "원래 변수",
                    alt.Tooltip(
                        "계수:Q",
                        format="+.4f"
                    )
                ]
            ).properties(
                height=max(
                    450,
                    len(coef_df) * 32
                )
            )
        )

        st.dataframe(
            coef_df[
                ["원래 변수", "항목", "계수"]
            ].round(4),
            hide_index=True,
            width="stretch"
        )

        st.info(
            "양수 계수는 AI 이용 확률을 높이는 방향, "
            "음수 계수는 낮추는 방향과 관련됩니다. "
            "인과관계나 변수 전체의 중요도를 의미하지는 않습니다."
        )

    # --------------------------------------
    # 의사결정트리
    # --------------------------------------

    elif selected == "의사결정트리":

        st.markdown("### 의사결정트리")

        st.write(
            "**작동 원리:** 매체 이용 여부, 학교급, 학년 등의 "
            "조건으로 학생들을 단계적으로 나눕니다. "
            "마지막 집단의 AI 이용률을 바탕으로 "
            "이용 여부를 예측합니다."
        )

        st.write(
            "**선정 이유:** 어떤 질문을 거쳐 분류하는지 "
            "직접 확인할 수 있어 설명하기 쉽습니다."
        )

        # 사용 방법 설명을 포함한 탐색 기능
        show_interactive_tree(pipe)

        st.subheader(
            "모델이 가장 먼저 확인하는 변수"
        )

        tree = model.tree_

        names = feature_names(pipe)

        first = tree.feature[0]

        if first >= 0:

            first_name = readable_name(
                names[first]
            )

            st.success(
                f"학습된 의사결정트리가 가장 먼저 "
                f"확인하는 항목은 "
                f"**{first_name}**입니다."
            )

        st.caption(
            "분류 조건과 집단별 수치는 실제 학습된 "
            "의사결정트리의 결과를 사용합니다."
        )

    # --------------------------------------
    # 랜덤 포레스트
    # --------------------------------------

    else:

        st.markdown("### 랜덤 포레스트")

        st.write(
            "**작동 원리:** 여러 의사결정트리를 학습하고 "
            "각 트리의 예측을 종합하여 "
            "생성형 AI 이용 여부를 분류합니다."
        )

        st.write(
            "**선정 이유:** 단일 트리보다 안정적인 "
            "예측이 가능할 수 있고 복잡한 관계도 "
            "학습할 수 있습니다."
        )

        names = feature_names(pipe)

        importance_df = pd.DataFrame({
            "원래 변수": [
                group_name(n)
                for n in names
            ],
            "중요도": model.feature_importances_
        })

        grouped = (
            importance_df.groupby(
                "원래 변수",
                as_index=False
            )["중요도"]
            .sum()
            .sort_values(
                "중요도",
                ascending=False
            )
        )

        st.subheader("변수별 중요도")

        show_chart(
            alt.Chart(grouped).mark_bar().encode(
                x="중요도:Q",
                y=alt.Y(
                    "원래 변수:N",
                    sort="-x",
                    title=None
                ),
                tooltip=[
                    "원래 변수",
                    alt.Tooltip(
                        "중요도:Q",
                        format=".4f"
                    )
                ]
            ).properties(height=390)
        )

        st.dataframe(
            grouped.round(4),
            hide_index=True,
            width="stretch"
        )

        top = grouped.iloc[0]

        st.write(
            f"가장 높은 분할 중요도를 보인 변수는 "
            f"**{top['원래 변수']}**입니다."
        )

        st.info(
            "이 중요도는 모델 내부의 분할 과정에서 "
            "계산된 값으로, 인과관계를 의미하지 않습니다."
        )

    st.subheader("④ 선택한 모델의 검증 결과")

    scores = result["scores"]

    cols = st.columns(5)

    for col, (name, value) in zip(
        cols,
        scores.items()
    ):

        col.metric(
            name,
            f"{value:.3f}"
        )

    st.write(
        f"**{selected}** 모델의 검증 정확도는 "
        f"**{scores['정확도']:.3f}**, "
        f"F1 점수는 **{scores['F1']:.3f}**입니다."
    )


# ==============================================
# ⑥ 모델 성능 비교 및 선정
# ==============================================

elif menu.startswith("⑥"):

    st.header("모델 성능 비교 및 선정")

    trained = train_models(
        X,
        Y.astype(int)
    )

    results = trained["results"]

    summary = pd.DataFrame([
        {
            "모델": name,
            **item["scores"]
        }
        for name, item in results.items()
    ])

    st.subheader("① 세 모델의 검증 성능")

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

    show_chart(
        alt.Chart(summary).mark_bar().encode(
            x=alt.X(
                "모델:N",
                sort=MODEL_NAMES
            ),
            y=alt.Y(
                f"{metric_name}:Q",
                scale=alt.Scale(
                    domain=[0, 1]
                )
            ),
            color="모델:N",
            tooltip=[
                "모델",
                alt.Tooltip(
                    f"{metric_name}:Q",
                    format=".3f"
                )
            ]
        ).properties(height=330)
    )

    explanations = {
        "정확도": "전체 응답 중 올바르게 예측한 비율",
        "정밀도": "AI 이용자로 예측한 사람 중 실제 이용자 비율",
        "재현율": "실제 AI 이용자 중 올바르게 찾은 비율",
        "F1": "정밀도와 재현율의 조화평균",
        "ROC-AUC": "AI 이용자와 미이용자를 구분하는 능력"
    }

    st.info(explanations[metric_name])

    st.subheader("③ 혼동행렬")

    selected_model = st.selectbox(
        "확인할 모델",
        MODEL_NAMES
    )

    cm = confusion_matrix(
        trained["valid_y"],
        results[selected_model]["pred"],
        labels=[0, 1]
    )

    labels = [
        "AI 미이용",
        "AI 이용"
    ]

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
        color="응답자 수:Q",
        tooltip=[
            "실제",
            "예측",
            "응답자 수"
        ]
    )

    numbers = base.mark_text(
        fontSize=20,
        color="white"
    ).encode(
        text="응답자 수:Q"
    )

    show_chart(
        (rectangles + numbers).properties(
            height=280
        )
    )

    st.write(
        "대각선의 두 칸은 올바르게 분류한 응답자 수이고 "
        "나머지 두 칸은 잘못 분류한 응답자 수입니다."
    )

    st.subheader("④ ROC 곡선")

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
                scale=alt.Scale(
                    domain=[0, 1]
                )
            ),
            y=alt.Y(
                "참양성률:Q",
                scale=alt.Scale(
                    domain=[0, 1]
                )
            ),
            color="모델:N",
            tooltip=[
                "모델",
                "거짓양성률",
                "참양성률"
            ]
        ).properties(height=320)
    )

    st.subheader("⑤ 최종 모델 선정")

    ranking = summary.sort_values(
        [
            "F1",
            "ROC-AUC",
            "정확도"
        ],
        ascending=False
    )

    st.dataframe(
        ranking.round(3),
        hide_index=True,
        width="stretch"
    )

    best = trained["best_name"]

    st.success(
        f"검증 데이터에서 F1 점수를 주요 기준으로 "
        f"선정한 모델은 **{best}**입니다."
    )

    st.write(
        "F1 점수가 같으면 ROC-AUC, "
        "그다음 정확도를 기준으로 비교했습니다."
    )

    st.subheader("⑥ 선정 모델의 최종 평가")

    final = trained["final_scores"]

    cols = st.columns(5)

    for col, (name, value) in zip(
        cols,
        final.items()
    ):

        col.metric(
            name,
            f"{value:.3f}"
        )

    st.write(
        f"**{best}** 모델의 최종 평가 정확도는 "
        f"**{final['정확도']:.3f}**, "
        f"F1 점수는 **{final['F1']:.3f}**입니다."
    )

    st.subheader("⑦ 탐구 결과 해석")

    st.write(
        "청소년의 디지털 매체 이용 경험과 학교급, 학년을 "
        "입력 변수로 사용하여 생성형 AI 이용 여부를 "
        "예측하는 세 개의 분류 모델을 학습했습니다."
    )

    st.write(
        "모델별 검증 성능을 비교하여 최종 모델을 선정하고 "
        "별도의 평가 데이터로 최종 성능을 확인했습니다."
    )

    st.info(
        "모델의 결과는 해당 설문 데이터에 대한 분석이며 "
        "매체 이용이 AI 이용의 원인이라는 뜻은 아닙니다. "
        "설문 가중치를 반영하지 않았으므로 "
        "전체 청소년 집단에 일반화할 때 주의해야 합니다."
    )
