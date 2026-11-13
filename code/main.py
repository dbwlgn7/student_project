
from pathlib import Path

import pandas as pd
import altair as alt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)


# =====================================================
# 1. 기본 설정
# =====================================================

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
    1: "초4", 2: "초5", 3: "초6",
    4: "중1", 5: "중2", 6: "중3",
    7: "고1", 8: "고2", 9: "고3"
}

MODEL_NAMES = [
    "로지스틱 회귀",
    "의사결정트리",
    "랜덤 포레스트"
]

METRIC_NAMES = [
    "정확도", "정밀도", "재현율",
    "F1", "ROC-AUC"
]

COLORS = {
    "기준선": "#94A3B8",
    "로지스틱 회귀": "#60A5FA",
    "의사결정트리": "#F59E0B",
    "랜덤 포레스트": "#34D399"
}


# =====================================================
# 2. 디자인
# =====================================================

st.markdown("""
<style>
.block-container {
    max-width: 1120px;
    padding-top: 1.8rem;
    padding-bottom: 4rem;
}
h1, h2, h3 {
    letter-spacing: -0.035em;
}
h1 {
    font-size: 2.05rem !important;
}
h2 {
    margin-top: 1.3rem !important;
}
[data-testid="stMetric"] {
    padding: 0.7rem 0;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem;
}
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)


def section(number, title, description=None):
    st.markdown(f"### {number}. {title}")
    if description:
        st.caption(description)


def show_chart(spec):
    st.altair_chart(spec, width="stretch")


def score_cards(scores, only=None):
    names = only or METRIC_NAMES
    cols = st.columns(len(names))
    for col, name in zip(cols, names):
        col.metric(name, f"{scores[name]:.3f}")


def performance_bar(df, metric):
    order = ["기준선"] + MODEL_NAMES

    graph = alt.Chart(df).mark_bar(
        cornerRadiusEnd=5,
        size=24
    ).encode(
        x=alt.X(
            f"{metric}:Q",
            title=metric,
            scale=alt.Scale(domain=[0, 1])
        ),
        y=alt.Y(
            "모델:N",
            sort=order,
            title=None,
            axis=alt.Axis(labelLimit=220)
        ),
        color=alt.Color(
            "모델:N",
            scale=alt.Scale(
                domain=order,
                range=[COLORS[n] for n in order]
            ),
            legend=None
        ),
        tooltip=[
            "모델:N",
            alt.Tooltip(f"{metric}:Q", format=".3f")
        ]
    ).properties(height=230)

    show_chart(graph)


def confusion_chart(y_true, predictions):
    matrix = confusion_matrix(
        y_true, predictions, labels=[0, 1]
    )

    names = ["AI 미이용", "AI 이용"]
    rows = []

    for i, actual in enumerate(names):
        for j, predicted in enumerate(names):
            rows.append({
                "실제": actual,
                "예측": predicted,
                "인원": int(matrix[i, j])
            })

    df = pd.DataFrame(rows)

    base = alt.Chart(df).encode(
        x=alt.X("예측:N", sort=names, title="예측"),
        y=alt.Y("실제:N", sort=names, title="실제")
    )

    heat = base.mark_rect().encode(
        color=alt.Color("인원:Q", scale=alt.Scale(scheme="blues")),
        tooltip=["실제", "예측", "인원"]
    )

    text = base.mark_text(
        fontSize=20,
        color="#111827"
    ).encode(text="인원:Q")

    show_chart(
        (heat + text).properties(height=280)
    )


# =====================================================
# 3. 데이터 관련 함수
# =====================================================

@st.cache_data
def read_csv(path):
    return pd.read_csv(
        path, encoding="utf-8-sig", low_memory=False
    )


def reason(n):
    if n == 1:
        return "응답자 식별 및 중복 검사에 사용. 모델 입력에서는 제외."
    if n == 12:
        return "생성형 AI 이용 여부는 예측 대상 Y이므로 유지."
    if n in [36, 40]:
        return "학교급과 학년별 차이를 확인하기 위해 유지."
    if n in KEEP:
        return "디지털 매체 이용 경험을 나타내는 입력 X."
    if n == 2:
        return "학교 식별 번호로 이용 행동을 직접 나타내지 않음."
    if n in [8, 10, 15, 16]:
        return "관련된 디지털 매체이나 이번 탐구의 범위에서 제외."
    if n in [17, 18]:
        return "종이 매체로 디지털 매체 중심 분석에서 제외."
    if n in [19, 20, 21]:
        return "AI 이용 정보가 포함될 수 있어 정답 정보 누출을 방지."
    if 22 <= n <= 29:
        return "온라인 안전 및 개인정보 보호 관련 변수."
    if 30 <= n <= 33:
        return "교육 경험과 교육 효과 관련 변수."
    if n in [34, 35]:
        return "개인·학교 배경 특성으로 분석 범위에서 제외."
    if n in [37, 38, 39, 43, 44]:
        return "학교급·학년과 관련된 추가 정보로 이번 범위에서 제외."
    if n in [41, 42]:
        return "지역 관련 변수로 이번 분석 범위에서 제외."
    if n in [45, 46]:
        return "설문조사 가중치로 이번 분류 입력에서 제외."
    return "탐구 범위에 따라 제외."


def feature_names(pipe):
    raw = pipe.named_steps[
        "preprocessing"
    ].get_feature_names_out()

    return [
        str(name).replace("category__", "")
        for name in raw
    ]


def readable_feature(name):
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


def original_feature(name):
    name = str(name).replace("category__", "")

    if name.startswith("학교급_"):
        return "학교급"
    if name.startswith("학년_"):
        return "학년"

    for media in MEDIA.values():
        if name.startswith(media + "_"):
            return media

    return name


# =====================================================
# 4. 분류 모델과 평가 함수
# =====================================================

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


def make_pipeline(algorithm, columns):
    encoder = OneHotEncoder(
        drop="if_binary",
        handle_unknown="ignore"
    )

    preprocessor = ColumnTransformer([
        ("category", encoder, list(columns))
    ])

    return Pipeline([
        ("preprocessing", preprocessor),
        ("model", algorithm)
    ])


def score_model(y_true, predictions, probabilities):
    return {
        "정확도": accuracy_score(y_true, predictions),
        "정밀도": precision_score(
            y_true, predictions, zero_division=0
        ),
        "재현율": recall_score(
            y_true, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_true, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_true, probabilities)
    }


@st.cache_resource(
    show_spinner="기준선과 세 분류 모델을 학습하고 있습니다."
)
def train_all(X, y):

    # 전체 데이터의 20%를 최종 평가용으로 따로 보관
    X_build, X_test, y_build, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # 학습 60%, 검증 20%, 최종 평가 20%
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_build, y_build,
        test_size=0.25,
        stratify=y_build,
        random_state=42
    )

    # 가장 많은 범주만 예측하는 기준선
    baseline = DummyClassifier(
        strategy="most_frequent"
    )
    baseline.fit(X_train, y_train)

    base_pred = baseline.predict(X_valid)
    base_prob = baseline.predict_proba(X_valid)[
        :, list(baseline.classes_).index(1)
    ]

    base_scores = score_model(
        y_valid, base_pred, base_prob
    )

    results = {}

    # 모든 후보 모델에 동일한 데이터를 사용
    for name, algorithm in make_models().items():
        pipe = make_pipeline(algorithm, X.columns)
        pipe.fit(X_train, y_train)

        prediction = pipe.predict(X_valid)
        probability = pipe.predict_proba(X_valid)[:, 1]

        results[name] = {
            "model": pipe,
            "pred": prediction,
            "prob": probability,
            "scores": score_model(
                y_valid, prediction, probability
            )
        }

    # 검증 F1 우선, 동점 시 ROC-AUC와 정확도 사용
    best_name = max(
        results,
        key=lambda name: (
            results[name]["scores"]["F1"],
            results[name]["scores"]["ROC-AUC"],
            results[name]["scores"]["정확도"]
        )
    )

    # 모델 선정 후 학습+검증 데이터를 합쳐 재학습
    final_model = make_pipeline(
        make_models()[best_name],
        X.columns
    )
    final_model.fit(X_build, y_build)

    test_pred = final_model.predict(X_test)
    test_prob = final_model.predict_proba(X_test)[:, 1]
    final_scores = score_model(
        y_test, test_pred, test_prob
    )

    # 최종 평가용 기준선도 같은 평가 데이터 사용
    final_baseline = DummyClassifier(
        strategy="most_frequent"
    )
    final_baseline.fit(X_build, y_build)

    final_base_pred = final_baseline.predict(X_test)
    final_base_prob = final_baseline.predict_proba(X_test)[
        :, list(final_baseline.classes_).index(1)
    ]

    final_base_scores = score_model(
        y_test, final_base_pred, final_base_prob
    )

    return {
        "baseline": {
            "pred": base_pred,
            "scores": base_scores,
            "majority": int(y_train.mode().iloc[0])
        },
        "results": results,
        "best": best_name,
        "valid_y": y_valid,
        "final": final_scores,
        "final_base": final_base_scores,
        "counts": {
            "학습": len(X_train),
            "검증": len(X_valid),
            "평가": len(X_test)
        }
    }


def comparison_table(trained):
    base = trained["baseline"]["scores"]

    rows = [{
        "모델": "기준선",
        **base
    }]

    for name, item in trained["results"].items():
        rows.append({
            "모델": name,
            **item["scores"]
        })

    table = pd.DataFrame(rows)

    table["정확도 개선(%p)"] = (
        table["정확도"] - base["정확도"]
    ) * 100

    table["F1 개선"] = (
        table["F1"] - base["F1"]
    )

    return table


# =====================================================
# 5. 의사결정트리: 한글 단계별 탐색
# =====================================================

def node_info(tree, node):
    values = tree.value[node][0].astype(float)
    total = values.sum()

    ai_rate = values[1] / total if total else 0

    return {
        "인원": int(tree.n_node_samples[node]),
        "AI 이용률": ai_rate * 100,
        "예측": (
            "AI 이용함" if ai_rate >= 0.5
            else "AI 이용 안함"
        )
    }


def tree_question(tree, node, names):
    name = names[tree.feature[node]]

    if name.startswith("학교급_"):
        return "학교급이 " + name.split("_", 1)[1] + "인가요?"
    if name.startswith("학년_"):
        return "학년이 " + name.split("_", 1)[1] + "인가요?"
    if name.endswith("_1"):
        return name[:-2] + "을 이용하나요?"
    if name.endswith("_0"):
        return name[:-2] + "을 이용하지 않나요?"

    return readable_feature(name) + "에 해당하나요?"


def node_card(title, info):
    with st.container(border=True):
        st.write("**" + title + "**")
        a, b = st.columns(2)
        a.metric("학생 수", f"{info['인원']:,}명")
        b.metric("AI 이용률", f"{info['AI 이용률']:.1f}%")
        st.caption("현재 예측: " + info["예측"])


def interactive_tree(pipe):
    tree = pipe.named_steps["model"].tree_
    names = feature_names(pipe)

    with st.expander("의사결정트리 사용 방법", expanded=True):
        st.write(
            "전체 학생에서 시작해 '예' 또는 '아니요' "
            "방향으로 이동하면서 실제 분류 조건을 확인합니다."
        )
        st.caption(
            "학생 수는 학습 데이터 기준이며, "
            "마지막 단계에서 최종 예측 결과가 나옵니다."
        )

    if "tree_node" not in st.session_state:
        st.session_state.tree_node = 0
        st.session_state.tree_history = []

    if st.button("처음부터 보기", key="tree_reset"):
        st.session_state.tree_node = 0
        st.session_state.tree_history = []

    node = st.session_state.tree_node

    if st.session_state.tree_history:
        if st.button("이전 단계로", key="tree_back"):
            st.session_state.tree_node = (
                st.session_state.tree_history.pop()
            )
            st.rerun()

    info = node_info(tree, node)

    st.write(
        f"**현재 단계: {len(st.session_state.tree_history)}단계**"
    )

    node_card("현재 집단", info)

    left = tree.children_left[node]
    right = tree.children_right[node]

    if left < 0:
        st.success(
            "마지막 분류 단계입니다. "
            f"이 집단의 예측 결과는 **{info['예측']}**입니다."
        )
        return

    question = tree_question(tree, node, names)

    st.info("**현재 분류 질문: " + question + "**")

    left_info = node_info(tree, left)
    right_info = node_info(tree, right)

    a, b = st.columns(2)

    with a:
        node_card("아니요 — 해당하지 않음", left_info)
        if st.button(
            "아니요 경로 보기",
            key="tree_no",
            use_container_width=True
        ):
            st.session_state.tree_history.append(node)
            st.session_state.tree_node = left
            st.rerun()

    with b:
        node_card("예 — 해당함", right_info)
        if st.button(
            "예 경로 보기",
            key="tree_yes",
            use_container_width=True
        ):
            st.session_state.tree_history.append(node)
            st.session_state.tree_node = right
            st.rerun()

    difference = (
        right_info["AI 이용률"] - left_info["AI 이용률"]
    )

    higher = (
        "예" if difference > 0
        else "아니요" if difference < 0
        else "두 집단이 동일"
    )

    st.caption(
        f"두 집단의 AI 이용률 차이: {abs(difference):.1f}%p"
        + (
            f" · {higher} 집단이 더 높음"
            if difference != 0 else ""
        )
    )


# =====================================================
# 6. 사이드바 및 데이터 불러오기
# =====================================================

st.sidebar.title("탐구 메뉴")

menu = st.sidebar.radio(
    "분석 단계",
    ["① 질문", "② 데이터 분석", "③ 기준선", "④ 모델 평가"]
)

st.sidebar.caption(
    "탐구 주제: 디지털 매체 이용 특성으로 "
    "생성형 AI 이용 여부를 분류할 수 있을까?"
)

uploaded = st.sidebar.file_uploader(
    "13열 CSV 직접 업로드 (선택)",
    type=["csv"]
)

if uploaded is not None:
    df = pd.read_csv(
        uploaded, encoding="utf-8-sig", low_memory=False
    )
elif SELECTED_FILE.exists():
    df = read_csv(str(SELECTED_FILE))
else:
    st.error("13열 CSV 파일을 찾을 수 없습니다.")
    st.stop()

if df.shape[1] != 13:
    st.error("선택한 13개 열로 구성된 CSV가 필요합니다.")
    st.stop()

id_col = df.columns[0]
target_col = df.columns[8]
school_col = df.columns[11]
grade_col = df.columns[12]

if (
    "응답자 ID" not in str(id_col)
    or "생성형 AI" not in str(target_col)
    or "학교급" not in str(school_col)
    or "학년별" not in str(grade_col)
):
    st.error("CSV 열 이름이나 순서가 예상과 다릅니다.")
    st.stop()

media_cols = list(df.columns[1:8]) + list(df.columns[9:11])

X_media = pd.DataFrame(index=df.index)

for name, col in zip(MEDIA.values(), media_cols):
    X_media[name] = pd.to_numeric(
        df[col], errors="coerce"
    ).map({1: 1, 2: 0})

Y = pd.to_numeric(
    df[target_col], errors="coerce"
).map({1: 1, 2: 0})

school_labels = pd.to_numeric(
    df[school_col], errors="coerce"
).map(SCHOOL)

grade_labels = pd.to_numeric(
    df[grade_col], errors="coerce"
).map(GRADE)

if (
    X_media.isna().any().any()
    or Y.isna().any()
    or school_labels.isna().any()
    or grade_labels.isna().any()
):
    st.error(
        "예상하지 못한 응답 코드나 결측치가 있습니다. "
        "CSV와 코드북을 확인해 주세요."
    )
    st.stop()

X = X_media.copy()
X["학교급"] = school_labels
X["학년"] = grade_labels

st.title("청소년 생성형 AI 이용 분류 탐구")
st.caption(
    "연구 질문 | 청소년의 디지털 매체 이용 특성으로 "
    "생성형 AI 이용 여부를 분류할 수 있을까?"
)


# =====================================================
# 7. 메뉴 ① 질문
# =====================================================

if menu == "① 질문":

    st.header("연구 질문")

    with st.container(border=True):
        st.markdown(
            "### 청소년의 디지털 매체 이용 특성으로 "
            "생성형 AI 이용 여부를 분류할 수 있을까?"
        )
        st.write(
            "디지털 매체 이용 경험, 학교급, 학년을 활용하여 "
            "AI 이용 여부를 예측하는 탐구입니다."
        )

    metric_row({
        "응답자": f"{len(df):,}명",
        "입력 변수": "11개",
        "비교 모델": "3개"
    })

    section("01", "왜 분류를 선택했을까?")

    a, b, c = st.columns(3)

    with a:
        with st.container(border=True):
            st.write("**회귀**")
            st.caption("연속적인 숫자를 예측")
            st.write("우리 주제의 주된 목표와 다름")

    with b:
        with st.container(border=True):
            st.write("**분류 — 이번 탐구**")
            st.caption("정해진 범주를 예측")
            st.write("AI 이용함 / 이용 안함 예측")

    with c:
        with st.container(border=True):
            st.write("**군집**")
            st.caption("정답 없이 비슷한 집단을 묶음")
            st.write("우리 데이터에는 정답 Y가 있음")

    st.success(
        "생성형 AI 이용 여부는 두 범주이므로 "
        "지도학습의 이진 분류가 적합합니다."
    )

    section("02", "어떤 데이터를 사용할까?")

    types = pd.DataFrame([
        ["응답자 ID", "식별자", "중복 검사", "제외"],
        ["디지털 매체 9개", "이진 범주형", "입력 X", "포함"],
        ["학교급", "순서형 범주", "입력 X", "포함"],
        ["학년", "순서형 범주", "입력 X", "포함"],
        ["생성형 AI 이용 여부", "이진 범주형", "정답 Y", "제외"]
    ], columns=["속성", "유형", "역할", "모델 입력"])

    st.dataframe(
        types, hide_index=True, width="stretch"
    )

    section("03", "비교할 세 분류 모델")

    models_info = [
        (
            "로지스틱 회귀",
            "여러 입력을 종합해 AI 이용 확률을 계산",
            "계수를 확인할 수 있는 기본 모델"
        ),
        (
            "의사결정트리",
            "조건을 하나씩 확인해 데이터를 나눔",
            "분류 과정을 설명하기 쉬움"
        ),
        (
            "랜덤 포레스트",
            "여러 의사결정트리의 판단을 종합",
            "복잡한 관계를 학습할 수 있음"
        )
    ]

    for name, principle, why in models_info:
        with st.container(border=True):
            st.write("**" + name + "**")
            st.write(principle)
            st.caption("선정 이유: " + why)


# =====================================================
# 8. 메뉴 ② 데이터 분석
# =====================================================

elif menu == "② 데이터 분석":

    st.header("데이터 분석")
    st.caption(
        "원본과 전처리 결과를 확인하고, "
        "생성형 AI 이용 여부와 입력 변수의 관계를 살펴봅니다."
    )

    tab1, tab2, tab3 = st.tabs([
        "전처리 확인",
        "데이터 분포",
        "AI 이용 여부와의 관계"
    ])

    with tab1:
        section("01", "데이터 전처리")

        metric_row({
            "원본 변수": "46개",
            "선택 변수": "13개",
            "제외 변수": "33개"
        })

        metric_row({
            "응답자": f"{len(df):,}명",
            "결측치": str(int(df.isna().sum().sum())),
            "중복 ID": str(int(df[id_col].duplicated().sum()))
        })

        st.write(
            "탐구와 관련된 변수만 선택했습니다. "
            "응답 값 자체는 수정하지 않았으며, "
            "학습용 데이터에서만 코드를 변환합니다."
        )

        with st.expander(
            "46개 변수의 선택·제외 근거 확인"
        ):
            if ORIGINAL_FILE.exists():
                original = read_csv(str(ORIGINAL_FILE))

                report = pd.DataFrame([
                    {
                        "번호": i,
                        "변수명": col,
                        "처리": "유지" if i in KEEP else "제외",
                        "근거": reason(i)
                    }
                    for i, col in enumerate(
                        original.columns, start=1
                    )
                ])

                st.dataframe(
                    report, hide_index=True, width="stretch"
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
                        "원본과 선택 데이터의 열 이름, "
                        "순서, 셀 값이 일치합니다."
                    )
                else:
                    st.warning(
                        "원본과 선택 데이터에 차이가 있습니다."
                    )
            else:
                st.warning("원본 46열 CSV 파일을 찾지 못했습니다.")

        st.info(
            "모델 입력 X에는 디지털 매체 이용 9개, "
            "학교급, 학년을 사용합니다. "
            "응답자 ID와 정답 Y는 입력에서 제외합니다."
        )

    with tab2:
        section("01", "디지털 매체별 이용률")
        st.caption(
            "이진 범주형 데이터이므로 "
            "매체별 비율을 비교하는 막대그래프를 사용합니다."
        )

        rates = (
            X_media.mean() * 100
        ).sort_values(ascending=False)

        rate_df = rates.reset_index()
        rate_df.columns = ["매체", "이용률"]

        show_chart(
            bar_graph(
                rate_df, "매체", "이용률",
                height=350, percent=True
            )
        )

        high = rate_df.iloc[0]
        low = rate_df.iloc[-1]

        st.write(
            f"이용률이 가장 높은 매체는 "
            f"**{high['매체']}({high['이용률']:.1f}%)**, "
            f"가장 낮은 매체는 "
            f"**{low['매체']}({low['이용률']:.1f}%)**입니다."
        )

        section("02", "생성형 AI 이용 여부")

        yes_count = int(Y.sum())
        no_count = int((Y == 0).sum())

        metric_row({
            "AI 이용함": f"{yes_count:,}명",
            "AI 이용 안함": f"{no_count:,}명",
            "AI 이용률": f"{Y.mean() * 100:.1f}%"
        })

        section("03", "학교급과 학년별 응답자 수")

        a, b = st.columns(2)

        with a:
            school_counts = (
                school_labels.value_counts()
                .reindex(list(SCHOOL.values()), fill_value=0)
            )
            st.write("**학교급별 응답자 수**")
            st.bar_chart(school_counts)

        with b:
            grade_counts = (
                grade_labels.value_counts()
                .reindex(list(GRADE.values()), fill_value=0)
            )
            st.write("**학년별 응답자 수**")
            st.bar_chart(grade_counts)

        st.caption(
            "학년은 원본 코드 1~9 대신 초4~고3으로 표시합니다."
        )

    with tab3:
        section("01", "매체 이용 경험과 AI 이용 여부의 관계")

        st.write(
            "우리의 목표는 매체끼리 비교하는 것이 아니라 "
            "**각 매체(X)가 AI 이용 여부(Y)와 "
            "어떤 관련이 있는지** 확인하는 것입니다."
        )

        rows = []

        for name in X_media.columns:
            r = X_media[name].corr(Y)
            rows.append({
                "매체": name,
                "상관계수": r,
                "절댓값": abs(r)
            })

        corr_df = pd.DataFrame(rows).sort_values(
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
                    sort=list(corr_df["매체"]),
                    title=None
                ),
                tooltip=[
                    "매체",
                    alt.Tooltip("상관계수:Q", format=".3f")
                ]
            ).properties(height=340)
        )

        first = corr_df.iloc[0]

        st.write(
            f"상관계수의 절댓값이 가장 큰 매체는 "
            f"**{first['매체']}**이며, "
            f"상관계수는 **{first['상관계수']:.3f}**입니다."
        )

        section("02", "AI 이용자와 미이용자 비교")

        group_rows = []
        diffs = []

        for name in X_media.columns:
            yes_rate = X_media.loc[Y == 1, name].mean() * 100
            no_rate = X_media.loc[Y == 0, name].mean() * 100

            group_rows.extend([
                {
                    "매체": name,
                    "집단": "AI 이용자",
                    "이용률": yes_rate
                },
                {
                    "매체": name,
                    "집단": "AI 미이용자",
                    "이용률": no_rate
                }
            ])

            diffs.append({
                "매체": name,
                "차이(%p)": yes_rate - no_rate
            })

        grouped = pd.DataFrame(group_rows)

        show_chart(
            alt.Chart(grouped).mark_bar().encode(
                x=alt.X(
                    "이용률:Q",
                    scale=alt.Scale(domain=[0, 100])
                ),
                y=alt.Y("매체:N", title=None),
                yOffset="집단:N",
                color="집단:N",
                tooltip=[
                    "매체", "집단",
                    alt.Tooltip("이용률:Q", format=".2f")
                ]
            ).properties(height=430)
        )

        diffs_df = pd.DataFrame(diffs)
        diffs_df["절댓값"] = diffs_df["차이(%p)"].abs()
        diffs_df = diffs_df.sort_values(
            "절댓값", ascending=False
        )

        largest = diffs_df.iloc[0]

        st.write(
            f"두 집단의 이용률 차이가 가장 큰 매체는 "
            f"**{largest['매체']}**이고, "
            f"차이는 **{abs(largest['차이(%p)']):.1f}%p**입니다."
        )

        section("03", "학교급·학년별 AI 이용률")

        a, b = st.columns(2)

        with a:
            school_rates = (
                pd.DataFrame({
                    "학교급": school_labels,
                    "AI 이용": Y
                })
                .groupby("학교급")["AI 이용"]
                .mean().mul(100)
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
                    tooltip=["학교급", "AI 이용률"]
                ).properties(height=290)
            )

        with b:
            grade_rates = (
                pd.DataFrame({
                    "학년": grade_labels,
                    "AI 이용": Y
                })
                .groupby("학년")["AI 이용"]
                .mean().mul(100)
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
                    tooltip=["학년", "AI 이용률"]
                ).properties(height=290)
            )

        st.info(
            "상관관계나 집단별 차이는 관련성을 나타낼 뿐 "
            "원인과 결과를 증명하지 않습니다."
        )


# =====================================================
# 9. 메뉴 ③ 기준선
# =====================================================

elif menu == "③ 기준선":

    st.header("기준선 모델")
    st.caption(
        "복잡한 분류 모델을 만들기 전, "
        "가장 단순한 예측 방식의 성능을 확인합니다."
    )

    trained = train_all(X, Y.astype(int))
    base = trained["baseline"]
    scores = base["scores"]

    with st.container(border=True):
        st.write("**기준선의 예측 방법**")

        majority = (
            "AI 이용함" if base["majority"] == 1
            else "AI 이용 안함"
        )

        st.markdown(
            f"### 모든 학생을 '{majority}'으로 예측"
        )

        st.write(
            "학습 데이터에서 가장 많은 답을 골라 "
            "모든 학생에게 같은 답을 예측합니다."
        )

        st.caption(
            "사용한 모델: DummyClassifier "
            "(strategy='most_frequent')"
        )

    section("01", "기준선의 실제 검증 성능")

    score_cards(scores)

    st.write(
        f"기준선 정확도는 **{scores['정확도'] * 100:.1f}%**, "
        f"F1 점수는 **{scores['F1']:.3f}**입니다."
    )

    section("02", "혼동행렬로 확인하기")

    confusion_chart(
        trained["valid_y"],
        base["pred"]
    )

    y_valid = trained["valid_y"]
    pred = base["pred"]

    missed_nonusers = int(
        ((y_valid == 0) & (pred == 1)).sum()
    )
    missed_users = int(
        ((y_valid == 1) & (pred == 0)).sum()
    )

    if base["majority"] == 1:
        st.warning(
            f"기준선은 AI 이용자를 모두 이용자로 예측하지만, "
            f"실제 미이용자 {missed_nonusers:,}명도 "
            f"이용자로 잘못 판단했습니다."
        )

        st.write(
            "그래서 재현율이 1.000이어도 "
            "AI 이용자와 미이용자를 잘 구분하는 모델은 아닙니다."
        )
    else:
        st.warning(
            f"기준선은 AI 미이용자로만 예측하기 때문에 "
            f"실제 이용자 {missed_users:,}명을 "
            f"찾아내지 못했습니다."
        )

    with st.expander("평가 지표의 의미"):
        st.write(
            "**정확도:** 전체 학생 중 올바르게 예측한 비율"
        )
        st.write(
            "**정밀도:** AI 이용자로 예측한 학생 중 "
            "실제 이용자인 비율"
        )
        st.write(
            "**재현율:** 실제 AI 이용자를 찾아낸 비율"
        )
        st.write(
            "**F1:** 정밀도와 재현율의 균형을 나타내는 점수"
        )
        st.write(
            "**ROC-AUC:** 두 집단을 구별하는 능력을 "
            "분류 기준 전반에서 평가한 점수"
        )

    st.info(
        "이 기준선보다 더 나은 성능을 보이는지 "
        "④ 모델 평가에서 확인합니다."
    )


# =====================================================
# 10. 메뉴 ④ 모델 평가
# =====================================================

elif menu == "④ 모델 평가":

    st.header("모델 평가")
    st.caption(
        "기준선과 세 분류 모델을 비교하고, "
        "우리 탐구에 적합한 모델을 선정합니다."
    )

    trained = train_all(X, Y.astype(int))
    comparison = comparison_table(trained)

    best = trained["best"]
    best_scores = trained["results"][best]["scores"]
    base_scores = trained["baseline"]["scores"]

    # -------------------------------------------------
    # 1. 최상단 요약
    # -------------------------------------------------

    with st.container(border=True):
        st.write("**검증 결과 요약**")
        st.markdown(f"### 선정 모델: {best}")

        a, b, c = st.columns(3)

        a.metric(
            "선정 모델 정확도",
            f"{best_scores['정확도'] * 100:.1f}%"
        )

        b.metric(
            "선정 모델 F1",
            f"{best_scores['F1']:.3f}"
        )

        c.metric(
            "기준선 대비 정확도",
            f"{(best_scores['정확도'] - base_scores['정확도']) * 100:+.2f}%p"
        )

        st.caption(
            "선정 기준: 검증 F1 점수 우선. "
            "동점이면 ROC-AUC와 정확도를 차례로 비교."
        )

    # -------------------------------------------------
    # 2. 성능 비교 탭
    # -------------------------------------------------

    tab_compare, tab_models, tab_matrix, tab_final = st.tabs([
        "성능 비교",
        "모델별 설명",
        "혼동행렬",
        "최종 평가"
    ])

    with tab_compare:

        section("01", "기준선과 세 모델의 성능")

        st.write(
            "아래는 **같은 검증 데이터**에서 계산한 결과입니다."
        )

        shown = comparison.copy()

        for name in METRIC_NAMES:
            shown[name] = shown[name].map(
                lambda x: f"{x:.3f}"
            )

        shown["정확도 개선(%p)"] = shown[
            "정확도 개선(%p)"
        ].map(lambda x: f"{x:+.2f}")

        shown["F1 개선"] = shown[
            "F1 개선"
        ].map(lambda x: f"{x:+.3f}")

        st.dataframe(
            shown,
            hide_index=True,
            width="stretch"
        )

        a, b = st.columns(2)

        with a:
            st.write("**정확도 비교**")
            performance_bar(comparison, "정확도")

        with b:
            st.write("**F1 점수 비교**")
            performance_bar(comparison, "F1")

        section("02", "기준선보다 얼마나 나아졌을까?")

        for name in MODEL_NAMES:
            result = trained["results"][name]["scores"]

            accuracy_diff = (
                result["정확도"] - base_scores["정확도"]
            ) * 100

            f1_diff = (
                result["F1"] - base_scores["F1"]
            )

            with st.container(border=True):
                st.write("**" + name + "**")

                a, b = st.columns(2)

                a.metric(
                    "정확도 차이",
                    f"{accuracy_diff:+.2f}%p"
                )

                b.metric(
                    "F1 차이",
                    f"{f1_diff:+.3f}"
                )

                acc_text = (
                    "높다" if accuracy_diff > 0
                    else "낮다" if accuracy_diff < 0
                    else "같다"
                )

                f1_text = (
                    "높다" if f1_diff > 0
                    else "낮다" if f1_diff < 0
                    else "같다"
                )

                st.caption(
                    f"기준선보다 정확도는 {acc_text}, "
                    f"F1 점수는 {f1_text}."
                )

        section("03", "이 결과를 어떻게 해석할까?")

        all_better_acc = all(
            trained["results"][name]["scores"]["정확도"]
            > base_scores["정확도"]
            for name in MODEL_NAMES
        )

        if all_better_acc:
            st.success(
                "세 후보 모델 모두 기준선보다 정확도가 높았습니다. "
                "따라서 디지털 매체 이용 경험과 학교급·학년을 "
                "활용한 예측이 단순한 다수 범주 예측보다 "
                "더 좋은 결과를 보였습니다."
            )
        else:
            st.warning(
                "일부 후보 모델의 정확도는 기준선보다 "
                "높지 않았습니다. 모델별 차이를 확인해야 합니다."
            )

        st.write(
            "하지만 정확도가 가장 높은 모델과 "
            "F1 점수가 가장 높은 모델은 다를 수 있습니다. "
            "이번 탐구는 미리 정한 F1 점수를 "
            "우선 기준으로 모델을 선정합니다."
        )

    # -------------------------------------------------
    # 3. 모델별 설명 탭
    # -------------------------------------------------

    with tab_models:

        section("01", "모델의 원리와 학습 결과")

        selected = st.selectbox(
            "자세히 살펴볼 모델",
            MODEL_NAMES
        )

        item = trained["results"][selected]
        pipe = item["model"]
        algorithm = pipe.named_steps["model"]

        if selected == "로지스틱 회귀":

            with st.container(border=True):
                st.write("**로지스틱 회귀**")
                st.write(
                    "입력 변수에 학습된 계수를 적용해 "
                    "AI 이용 확률을 계산합니다."
                )
                st.caption(
                    "선정 이유: 작동 원리를 설명하기 쉽고 "
                    "기본 비교 모델로 적합합니다."
                )

            names = feature_names(pipe)

            coefficient_df = pd.DataFrame({
                "원래 변수": [
                    original_feature(n) for n in names
                ],
                "항목": [
                    readable_feature(n) for n in names
                ],
                "계수": algorithm.coef_[0]
            })

            coefficient_df["절댓값"] = (
                coefficient_df["계수"].abs()
            )

            coefficient_df = coefficient_df.sort_values(
                "절댓값", ascending=False
            )

            st.write("**학습된 계수 상위 항목**")

            top = coefficient_df.head(12)

            show_chart(
                alt.Chart(top).mark_bar().encode(
                    x="계수:Q",
                    y=alt.Y(
                        "항목:N",
                        sort=list(top["항목"]),
                        title=None,
                        axis=alt.Axis(labelLimit=300)
                    ),
                    tooltip=[
                        "원래 변수",
                        "항목",
                        alt.Tooltip("계수:Q", format="+.4f")
                    ]
                ).properties(height=440)
            )

            with st.expander("모든 계수 확인"):
                st.dataframe(
                    coefficient_df[
                        ["원래 변수", "항목", "계수"]
                    ].round(4),
                    hide_index=True,
                    width="stretch"
                )

            st.caption(
                "양수 계수는 AI 이용 확률을 높이는 방향, "
                "음수 계수는 낮추는 방향과 관련됩니다. "
                "계수는 인과관계를 뜻하지 않습니다."
            )

        elif selected == "의사결정트리":

            with st.container(border=True):
                st.write("**의사결정트리**")
                st.write(
                    "학교급, 학년, 매체 이용 여부 등의 "
                    "조건을 단계적으로 확인해 분류합니다."
                )
                st.caption(
                    "선정 이유: 실제 분류 조건을 "
                    "한 단계씩 확인할 수 있습니다."
                )

            interactive_tree(pipe)

        else:

            with st.container(border=True):
                st.write("**랜덤 포레스트**")
                st.write(
                    "여러 의사결정트리를 학습하고 "
                    "각 트리의 판단을 종합해 예측합니다."
                )
                st.caption(
                    "선정 이유: 복잡한 관계를 학습할 수 있고 "
                    "단일 트리보다 안정적일 수 있습니다."
                )

            names = feature_names(pipe)

            importance_df = pd.DataFrame({
                "원래 변수": [
                    original_feature(n) for n in names
                ],
                "중요도": algorithm.feature_importances_
            })

            grouped = (
                importance_df.groupby(
                    "원래 변수", as_index=False
                )["중요도"]
                .sum()
                .sort_values("중요도", ascending=False)
            )

            show_chart(
                bar_graph(
                    grouped, "원래 변수",
                    "중요도", height=360
                )
            )

            st.dataframe(
                grouped.round(4),
                hide_index=True,
                width="stretch"
            )

            st.caption(
                "변수 중요도는 트리 분할 과정의 "
                "불순도 감소를 바탕으로 계산합니다. "
                "원인이나 실제 영향력의 크기를 뜻하지 않습니다."
            )

        section("02", "선택한 모델의 검증 결과")
        score_cards(item["scores"])

        st.write(
            f"**{selected}**의 정확도는 "
            f"**{item['scores']['정확도']:.3f}**, "
            f"F1 점수는 **{item['scores']['F1']:.3f}**입니다."
        )

    # -------------------------------------------------
    # 4. 혼동행렬 탭
    # -------------------------------------------------

    with tab_matrix:

        section("01", "실제와 예측을 비교하기")

        st.write(
            "혼동행렬은 모델이 AI 이용자와 미이용자를 "
            "각각 몇 명이나 올바르게 맞혔는지 보여줍니다."
        )

        selected_cm = st.selectbox(
            "확인할 모델",
            ["기준선"] + MODEL_NAMES
        )

        if selected_cm == "기준선":
            predictions = trained["baseline"]["pred"]
        else:
            predictions = trained["results"][
                selected_cm
            ]["pred"]

        confusion_chart(
            trained["valid_y"],
            predictions
        )

        matrix = confusion_matrix(
            trained["valid_y"],
            predictions,
            labels=[0, 1]
        )

        tn, fp, fn, tp = matrix.ravel()

        a, b = st.columns(2)

        a.metric("올바르게 예측", f"{tn + tp:,}명")
        b.metric("잘못 예측", f"{fp + fn:,}명")

        st.write(
            f"실제 AI 이용자 중 **{tp:,}명**을 "
            f"올바르게 찾아냈고, "
            f"**{fn:,}명**은 놓쳤습니다."
        )

        st.write(
            f"실제 AI 미이용자 중 **{tn:,}명**을 "
            f"올바르게 분류했고, "
            f"**{fp:,}명**은 이용자로 잘못 예측했습니다."
        )

        st.caption(
            "모델의 좋은 성능을 판단하려면 "
            "두 집단을 모두 얼마나 잘 구분하는지 확인해야 합니다."
        )

    # -------------------------------------------------
    # 5. 최종 평가 탭
    # -------------------------------------------------

    with tab_final:

        section("01", "최종 모델 선정 근거")

        ranking = comparison[
            comparison["모델"] != "기준선"
        ].sort_values(
            ["F1", "ROC-AUC", "정확도"],
            ascending=False
        )

        st.dataframe(
            ranking.round(3),
            hide_index=True,
            width="stretch"
        )

        second = ranking.iloc[1]

        best_f1 = best_scores["F1"]
        second_f1 = second["F1"]

        base_acc = base_scores["정확도"]
        base_f1 = base_scores["F1"]

        st.success(
            f"검증 데이터에서 F1 점수가 가장 높은 "
            f"모델은 **{best}**입니다."
        )

        st.write(
            f"이 모델의 F1 점수는 **{best_f1:.3f}**이고, "
            f"두 번째 모델인 **{second['모델']}**보다 "
            f"**{best_f1 - second_f1:.3f}** 높습니다."
        )

        st.write(
            f"기준선과 비교하면 정확도는 "
            f"**{(best_scores['정확도'] - base_acc) * 100:+.2f}%p**, "
            f"F1 점수는 "
            f"**{best_scores['F1'] - base_f1:+.3f}** 차이 납니다."
        )

        characteristics = {
            "로지스틱 회귀":
                "입력 변수의 계수를 해석할 수 있고, "
                "구조가 비교적 단순하다는 장점이 있습니다.",
            "의사결정트리":
                "어떤 조건으로 분류했는지 "
                "단계별로 설명하기 쉽다는 장점이 있습니다.",
            "랜덤 포레스트":
                "여러 결정트리의 판단을 종합하여 "
                "복잡한 관계를 학습할 수 있다는 장점이 있습니다."
        }

        st.write(
            "**선정 모델의 특징:** "
            + characteristics[best]
        )

        st.caption(
            "이 모델의 장점은 선정 근거를 보완하는 설명이며, "
            "최종 선택은 실제 검증 지표를 기준으로 했습니다."
        )

        section("02", "최종 평가 데이터 결과")

        st.write(
            "모델을 선정한 뒤, 학습·검증 데이터로 재학습하고 "
            "그동안 사용하지 않은 평가 데이터로 "
            "최종 성능을 확인했습니다."
        )

        final = trained["final"]
        final_base = trained["final_base"]

        a, b, c = st.columns(3)

        a.metric(
            "선정 모델 정확도",
            f"{final['정확도'] * 100:.2f}%"
        )
        b.metric(
            "기준선 정확도",
            f"{final_base['정확도'] * 100:.2f}%"
        )
        c.metric(
            "정확도 차이",
            f"{(final['정확도'] - final_base['정확도']) * 100:+.2f}%p"
        )

        a, b = st.columns(2)

        a.metric("선정 모델 F1", f"{final['F1']:.3f}")
        b.metric("기준선 F1", f"{final_base['F1']:.3f}")

        with st.expander("최종 평가 전체 지표 보기"):
            st.dataframe(
                pd.DataFrame([
                    {"모델": "기준선", **final_base},
                    {"모델": best, **final}
                ]).round(3),
                hide_index=True,
                width="stretch"
            )

        section("03", "우리의 탐구 질문에 대한 답변")

        acc_diff = final["정확도"] - final_base["정확도"]
        f1_diff = final["F1"] - final_base["F1"]

        if acc_diff > 0 and f1_diff > 0:
            st.success(
                "최종 평가에서 선정 모델은 기준선보다 "
                "정확도와 F1 점수가 모두 높았습니다."
            )
            st.write(
                "따라서 이번 설문 데이터에서는 "
                "디지털 매체 이용 특성과 학교급·학년을 활용하면 "
                "단순한 다수 범주 예측보다 "
                "생성형 AI 이용 여부를 더 잘 분류할 수 "
                "있음을 확인했습니다."
            )

        elif acc_diff > 0:
            st.warning(
                "최종 평가에서 정확도는 기준선보다 높았지만, "
                "F1 점수는 높지 않았습니다."
            )
            st.write(
                "따라서 정확도 측면에서는 개선되었으나, "
                "AI 이용자 분류의 균형까지 개선되었다고 "
                "단정할 수는 없습니다."
            )

        else:
            st.warning(
                "최종 평가에서 선정 모델의 정확도가 "
                "기준선보다 높지 않았습니다."
            )
            st.write(
                "이번 결과만으로 디지털 매체 이용 특성을 활용한 "
                "모델이 단순한 기준선보다 더 잘 분류한다고 "
                "결론 내릴 수는 없습니다."
            )

        st.info(
            "이 결과는 해당 설문 데이터에서의 예측 성능입니다. "
            "매체 이용이 AI 이용의 원인이라는 뜻은 아니며, "
            "다른 학생 집단에서도 같은 성능이 나온다고 "
            "보장하지 않습니다. 조사 가중치도 적용하지 않았습니다."
        )
