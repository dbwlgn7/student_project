
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

CANDIDATES = [
    "로지스틱 회귀",
    "의사결정트리",
    "랜덤 포레스트"
]

METRICS = [
    "정확도", "정밀도", "재현율", "F1", "ROC-AUC"
]


# ==========================================
# 2. 공통 함수
# ==========================================

@st.cache_data
def read_csv(path):
    return pd.read_csv(
        path, encoding="utf-8-sig", low_memory=False
    )


def show_chart(chart):
    st.altair_chart(chart, width="stretch")


def reason(n):
    if n == 1:
        return "응답자 식별 및 중복 검사에 사용. 모델 입력에서 제외."
    if n == 12:
        return "생성형 AI 이용 여부는 예측 대상 Y이므로 유지."
    if n in [36, 40]:
        return "학교급 및 학년별 이용 특성 분석에 사용."
    if n in KEEP:
        return "디지털 매체 이용 경험을 나타내는 입력 변수 X."
    if n == 2:
        return "학교 식별 번호이므로 입력에서 제외."
    if n in [8, 10, 15, 16]:
        return "관련 디지털 매체이나 이번 탐구 범위에서 제외."
    if n in [17, 18]:
        return "종이 매체로 디지털 매체 중심 탐구에서 제외."
    if n in [19, 20, 21]:
        return "AI 이용 정보가 직접 포함될 수 있어 누수 방지를 위해 제외."
    if 22 <= n <= 29:
        return "온라인 안전·개인정보 보호 경험 관련 변수."
    if 30 <= n <= 33:
        return "교육 경험과 효과 관련 변수."
    if n in [34, 35]:
        return "개인 및 학교 배경 특성으로 분석 범위에서 제외."
    if n in [37, 38, 39, 43, 44]:
        return "기본 학교급·학년 변수를 사용하기 위해 제외."
    if n in [41, 42]:
        return "지역 특성으로 분석 범위에서 제외."
    if n in [45, 46]:
        return "설문조사 가중치로 이번 모델 입력에서 제외."
    return "탐구 목적에 따라 제외."


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


def make_pipeline(model, columns):
    encoder = OneHotEncoder(
        drop="if_binary",
        handle_unknown="ignore"
    )

    transformer = ColumnTransformer([
        ("category", encoder, list(columns))
    ])

    return Pipeline([
        ("preprocessing", transformer),
        ("model", model)
    ])


def calculate_scores(y_true, prediction, probability):
    return {
        "정확도": accuracy_score(y_true, prediction),
        "정밀도": precision_score(
            y_true, prediction, zero_division=0
        ),
        "재현율": recall_score(
            y_true, prediction, zero_division=0
        ),
        "F1": f1_score(
            y_true, prediction, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_true, probability)
    }


def feature_names(pipe):
    names = pipe.named_steps[
        "preprocessing"
    ].get_feature_names_out()

    return [
        str(name).replace("category__", "")
        for name in names
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


def original_name(name):
    name = str(name).replace("category__", "")

    if name.startswith("학교급_"):
        return "학교급"

    if name.startswith("학년_"):
        return "학년"

    for media in MEDIA.values():
        if name.startswith(media + "_"):
            return media

    return name


# ==========================================
# 3. 기준선 + 모델 학습
# ==========================================

@st.cache_resource(
    show_spinner="기준선과 세 가지 모델을 학습하고 있습니다."
)
def train_all(X, y):

    # 전체 데이터의 20%는 최종 평가용으로 보관
    X_build, X_test, y_build, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # 나머지 80%를 학습 60%, 검증 20%로 분리
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_build, y_build,
        test_size=0.25,
        stratify=y_build,
        random_state=42
    )

    # 기준선: 학습 데이터에서 가장 많은 정답만 예측
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    base_pred = baseline.predict(X_valid)

    # 기준선의 이용함 예측 확률
    base_prob = baseline.predict_proba(X_valid)[
        :, list(baseline.classes_).index(1)
    ]

    base_scores = calculate_scores(
        y_valid, base_pred, base_prob
    )

    results = {}

    # 세 후보 모델에 같은 학습·검증 데이터 사용
    for name, algorithm in make_models().items():

        pipe = make_pipeline(algorithm, X.columns)
        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_valid)
        prob = pipe.predict_proba(X_valid)[:, 1]

        results[name] = {
            "pipeline": pipe,
            "prediction": pred,
            "probability": prob,
            "scores": calculate_scores(y_valid, pred, prob)
        }

    # 검증 F1을 우선 비교
    # 동점이면 ROC-AUC, 정확도 순으로 선정
    best_name = max(
        results,
        key=lambda name: (
            results[name]["scores"]["F1"],
            results[name]["scores"]["ROC-AUC"],
            results[name]["scores"]["정확도"]
        )
    )

    # 최종 모델을 학습+검증 데이터로 재학습
    final_model = make_pipeline(
        make_models()[best_name],
        X.columns
    )

    final_model.fit(X_build, y_build)

    final_pred = final_model.predict(X_test)
    final_prob = final_model.predict_proba(X_test)[:, 1]

    final_scores = calculate_scores(
        y_test, final_pred, final_prob
    )

    # 최종 평가에서도 기준선을 같은 조건으로 비교
    final_baseline = DummyClassifier(
        strategy="most_frequent"
    )

    final_baseline.fit(X_build, y_build)

    final_base_pred = final_baseline.predict(X_test)
    final_base_prob = final_baseline.predict_proba(X_test)[
        :, list(final_baseline.classes_).index(1)
    ]

    final_base_scores = calculate_scores(
        y_test, final_base_pred, final_base_prob
    )

    majority = int(y_train.mode().iloc[0])

    return {
        "baseline": {
            "model": baseline,
            "prediction": base_pred,
            "probability": base_prob,
            "scores": base_scores,
            "majority": majority
        },
        "results": results,
        "best_name": best_name,
        "final_scores": final_scores,
        "final_base_scores": final_base_scores,
        "valid_y": y_valid,
        "counts": {
            "학습": len(X_train),
            "검증": len(X_valid),
            "최종 평가": len(X_test)
        }
    }


def comparison_table(trained):
    rows = [
        {
            "모델": "기준선",
            **trained["baseline"]["scores"]
        }
    ]

    for name, result in trained["results"].items():
        rows.append({
            "모델": name,
            **result["scores"]
        })

    table = pd.DataFrame(rows)

    baseline_acc = trained["baseline"]["scores"]["정확도"]
    baseline_f1 = trained["baseline"]["scores"]["F1"]

    table["기준선 대비 정확도(%p)"] = (
        table["정확도"] - baseline_acc
    ) * 100

    table["기준선 대비 F1 차이"] = (
        table["F1"] - baseline_f1
    )

    return table


# ==========================================
# 4. 의사결정트리 탐색 함수
# ==========================================

def node_info(tree, node):
    values = tree.value[node][0].astype(float)
    total = values.sum()

    ai_rate = values[1] / total if total else 0

    return {
        "응답자 수": int(tree.n_node_samples[node]),
        "AI 이용률": ai_rate * 100,
        "예측": (
            "AI 이용함"
            if ai_rate >= 0.5
            else "AI 이용 안함"
        )
    }


def tree_question(tree, node, names):
    name = names[tree.feature[node]]
    threshold = tree.threshold[node]

    if name.startswith("학교급_"):
        question = (
            "학교급이 "
            + name.split("_", 1)[1]
            + "인가요?"
        )
    elif name.startswith("학년_"):
        question = (
            "학년이 "
            + name.split("_", 1)[1]
            + "인가요?"
        )
    elif name.endswith("_1"):
        question = name[:-2] + "을 이용하나요?"
    elif name.endswith("_0"):
        question = name[:-2] + "을 이용하지 않나요?"
    else:
        question = readable_name(name) + "에 해당하나요?"

    return question, threshold


def get_tree_paths(tree):
    paths = []

    def visit(node, path, depth):
        paths.append((node, path, depth))

        left = tree.children_left[node]
        right = tree.children_right[node]

        if left >= 0:
            visit(left, path + " → 아니요", depth + 1)
            visit(right, path + " → 예", depth + 1)

    visit(0, "시작", 0)
    return paths


def show_tree_card(title, info):
    with st.container(border=True):
        st.markdown("**" + title + "**")

        a, b = st.columns(2)

        a.metric(
            "학습 응답자 수",
            f"{info['응답자 수']:,}명"
        )

        b.metric(
            "AI 이용률",
            f"{info['AI 이용률']:.1f}%"
        )

        st.write("현재 예측: **" + info["예측"] + "**")


def show_interactive_tree(pipe):
    model = pipe.named_steps["model"]
    tree = model.tree_
    names = feature_names(pipe)

    st.subheader("실제 학습된 분류 과정")

    with st.expander(
        "📖 의사결정트리 사용 방법",
        expanded=True
    ):
        st.write(
            "아래에서 분류 단계를 선택하면 "
            "해당 조건과 학생들의 AI 이용률을 볼 수 있습니다."
        )
        st.write(
            "**시작(단계 0):** 전체 학습 학생에서 시작합니다."
        )
        st.write(
            "**→ 아니요:** 해당 질문의 조건을 만족하지 않는 집단입니다."
        )
        st.write(
            "**→ 예:** 해당 질문의 조건을 만족하는 집단입니다."
        )
        st.write(
            "**단계 1, 2, 3…:** 더 세부적인 조건으로 나뉩니다."
        )
        st.write(
            "**마지막 단계:** 더 이상 나누지 않고 최종 예측합니다."
        )

    paths = get_tree_paths(tree)

    options = {
        path + f" (단계 {depth})": node
        for node, path, depth in paths
    }

    selected_path = st.selectbox(
        "확인할 분류 단계",
        list(options.keys())
    )

    node = options[selected_path]
    info = node_info(tree, node)

    show_tree_card("현재 단계", info)

    left_node = tree.children_left[node]
    right_node = tree.children_right[node]

    if left_node < 0:
        st.success(
            "마지막 단계입니다. 최종 예측은 "
            + info["예측"] + "입니다."
        )
        return

    question, threshold = tree_question(
        tree, node, names
    )

    st.subheader("현재 분류 질문")
    st.info("**" + question + "**")

    left_info = node_info(tree, left_node)
    right_info = node_info(tree, right_node)

    a, b = st.columns(2)

    with a:
        show_tree_card("아니요", left_info)

    with b:
        show_tree_card("예", right_info)

    difference = (
        right_info["AI 이용률"]
        - left_info["AI 이용률"]
    )

    st.write(
        f"두 집단의 AI 이용률 차이는 "
        f"**{abs(difference):.1f}%p**입니다."
    )

    if difference > 0:
        st.write("'예' 집단의 AI 이용률이 더 높습니다.")
    elif difference < 0:
        st.write("'아니요' 집단의 AI 이용률이 더 높습니다.")
    else:
        st.write("두 집단의 AI 이용률이 같습니다.")

    st.caption(
        "이 비율은 학습 데이터에서 계산되었습니다. "
        "실제 예측 성능은 별도 검증 데이터로 평가합니다."
    )


# ==========================================
# 5. 사이드바: 정확히 4개 메뉴
# ==========================================

st.sidebar.title("🧭 탐구 메뉴")

menu = st.sidebar.radio(
    "메뉴 선택",
    [
        "① 질문",
        "② 데이터 분석",
        "③ 기준선",
        "④ 모델 평가"
    ]
)

uploaded = st.sidebar.file_uploader(
    "13열 CSV 업로드 (선택)",
    type=["csv"]
)


# ==========================================
# 6. 데이터 불러오기
# ==========================================

if uploaded is not None:
    df = pd.read_csv(
        uploaded, encoding="utf-8-sig"
    )
elif SELECTED_FILE.exists():
    df = read_csv(str(SELECTED_FILE))
else:
    st.error("13열 CSV 파일을 찾지 못했습니다.")
    st.stop()

if df.shape[1] != 13:
    st.error(
        f"13열 CSV가 필요하지만 현재 {df.shape[1]}열입니다."
    )
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
    st.error(
        "CSV의 열 이름이나 순서가 예상과 다릅니다."
    )
    st.stop()

media_cols = (
    list(df.columns[1:8])
    + list(df.columns[9:11])
)

X_media = pd.DataFrame(index=df.index)

for name, column in zip(MEDIA.values(), media_cols):
    X_media[name] = pd.to_numeric(
        df[column], errors="coerce"
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
        "일부 응답 코드가 예상 범위를 벗어났습니다. "
        "CSV와 코드북을 확인해 주세요."
    )
    st.stop()

# ID와 정답 Y는 모델 입력에서 제외
X = X_media.copy()
X["학교급"] = school_labels
X["학년"] = grade_labels

st.title("📊 청소년 생성형 AI 이용 분류 탐구")
st.caption(
    "청소년의 디지털 매체 이용 특성으로 "
    "생성형 AI 이용 여부를 분류할 수 있을까?"
)


# ==========================================
# 메뉴 1. 질문
# ==========================================

if menu == "① 질문":

    st.header("탐구 질문")

    st.write(
        "**청소년의 디지털 매체 이용 특성으로 "
        "생성형 AI 이용 여부를 분류할 수 있을까?**"
    )

    a, b, c = st.columns(3)
    a.metric("응답자 수", f"{len(df):,}명")
    b.metric("선택 변수", 13)
    c.metric("모델 입력 변수", 11)

    st.subheader("① 왜 분류를 선택했을까?")

    methods = pd.DataFrame([
        [
            "회귀",
            "연속적인 숫자를 예측",
            "우리의 정답은 수치가 아닌 범주"
        ],
        [
            "분류",
            "정해진 범주를 예측",
            "이용함 / 이용 안함 예측에 적합"
        ],
        [
            "군집",
            "정답 없이 유사한 집단 구성",
            "이미 이용 여부라는 정답이 존재"
        ]
    ], columns=[
        "방법", "목적", "우리 주제와의 관계"
    ])

    st.dataframe(
        methods, hide_index=True, width="stretch"
    )

    st.success(
        "타깃 변수가 '이용함'과 '이용 안함' "
        "두 범주이므로 지도학습의 이진 분류를 선택했습니다."
    )

    st.subheader("② 데이터 속성과 유형")

    types = pd.DataFrame([
        ["응답자 ID", "식별자", "중복 검사", "제외"],
        ["9개 디지털 매체", "이진 범주형", "입력 X", "포함"],
        ["학교급", "순서형 범주", "입력 X", "포함"],
        ["학년", "순서형 범주", "입력 X", "포함"],
        ["생성형 AI 이용 여부", "이진 범주형", "정답 Y", "제외"]
    ], columns=[
        "속성", "데이터 유형", "역할", "모델 입력"
    ])

    st.dataframe(
        types, hide_index=True, width="stretch"
    )

    st.subheader("③ 비교할 모델 후보")

    candidates = pd.DataFrame([
        [
            "로지스틱 회귀",
            "이용 확률을 계산",
            "기본 비교 모델로 적합"
        ],
        [
            "의사결정트리",
            "질문을 통해 단계적으로 분류",
            "분류 과정을 설명하기 쉬움"
        ],
        [
            "랜덤 포레스트",
            "여러 결정트리의 예측을 종합",
            "복잡한 관계를 학습할 수 있음"
        ]
    ], columns=[
        "후보 모델", "작동 원리", "선정 이유"
    ])

    st.dataframe(
        candidates, hide_index=True, width="stretch"
    )

    st.subheader("④ 탐구 진행 과정")

    st.write(
        "질문 설정 → 데이터 전처리 및 관계 분석 → "
        "기준선 성능 확인 → 모델 3개 학습 → "
        "기준선과 성능 비교 → 최종 모델 선정"
    )


# ==========================================
# 메뉴 2. 데이터 분석
# ==========================================

elif menu == "② 데이터 분석":

    st.header("데이터 분석")

    st.subheader("① 데이터 전처리")

    a, b, c = st.columns(3)
    a.metric("원본 변수", 46)
    b.metric("선택 변수", 13)
    c.metric("제외 변수", 33)

    a, b, c = st.columns(3)
    a.metric("응답자 수", f"{len(df):,}")
    b.metric("결측치 수", int(df.isna().sum().sum()))
    c.metric(
        "중복 ID 수",
        int(df[id_col].duplicated().sum())
    )

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

        choice = st.selectbox(
            "변수 선택 근거 확인",
            ["전체", "유지", "제외"]
        )

        if choice != "전체":
            report = report[
                report["처리"] == choice
            ]

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
            st.success(
                "원본과 선택 CSV의 열 이름·순서·셀 값이 일치합니다."
            )
        else:
            st.warning(
                "원본과 선택 CSV 사이에 차이가 있습니다."
            )
    else:
        st.warning(
            "원본 46열 CSV가 없어 전처리 비교표를 표시하지 못했습니다."
        )

    st.info(
        "원본의 응답 코드는 변경하지 않았습니다. "
        "분석용 복사본에서만 이용함=1, 이용 안함=0으로 "
        "변환했습니다. 모델 학습 시 ID와 Y는 X에서 제외합니다."
    )

    st.subheader("② 매체별 이용률")

    st.caption(
        "이진 범주형 변수의 비율을 비교하는 가로 막대그래프"
    )

    rates = (X_media.mean() * 100).sort_values(
        ascending=False
    )

    rate_df = rates.reset_index()
    rate_df.columns = ["매체", "이용률"]

    show_chart(
        alt.Chart(rate_df).mark_bar().encode(
            x=alt.X(
                "이용률:Q",
                scale=alt.Scale(domain=[0, 100]),
                title="이용률(%)"
            ),
            y=alt.Y("매체:N", sort="-x"),
            tooltip=[
                "매체",
                alt.Tooltip("이용률:Q", format=".2f")
            ]
        ).properties(height=370)
    )

    highest = rate_df.iloc[0]
    lowest = rate_df.iloc[-1]

    st.write(
        f"이용률이 가장 높은 매체는 "
        f"**{highest['매체']}**({highest['이용률']:.1f}%), "
        f"가장 낮은 매체는 "
        f"**{lowest['매체']}**({lowest['이용률']:.1f}%)입니다."
    )

    st.subheader("③ 생성형 AI 이용 여부 분포")

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
        ).properties(height=270)
    )

    st.write(
        f"전체 응답자 중 AI 이용 경험이 있는 학생은 "
        f"{int((Y == 1).sum()):,}명, "
        f"비율은 {Y.mean() * 100:.1f}%입니다."
    )

    st.subheader("④ 학교급·학년별 응답자 분포")

    a, b = st.columns(2)

    with a:
        st.write("**학교급별 응답자 수**")
        st.bar_chart(
            school_labels.value_counts().reindex(
                list(SCHOOL.values()), fill_value=0
            )
        )

    with b:
        st.write("**학년별 응답자 수**")
        st.bar_chart(
            grade_labels.value_counts().reindex(
                list(GRADE.values()), fill_value=0
            )
        )

    st.subheader("⑤ 입력 변수 X와 타깃 Y의 관계")

    corr_rows = []

    for name in X_media.columns:
        value = X_media[name].corr(Y)
        corr_rows.append({
            "매체": name,
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
        ).properties(height=370)
    )

    if not corr_df.empty:
        top_corr = corr_df.iloc[0]
        st.write(
            f"AI 이용 여부와 상관계수 절댓값이 "
            f"가장 큰 매체는 **{top_corr['매체']}**이며 "
            f"상관계수는 {top_corr['상관계수']:.3f}입니다."
        )

    st.subheader("⑥ AI 이용자·미이용자 매체 이용률 비교")

    rows = []
    differences = []

    for name in X_media.columns:
        yes_rate = X_media.loc[Y == 1, name].mean() * 100
        no_rate = X_media.loc[Y == 0, name].mean() * 100

        rows.extend([
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

        differences.append({
            "매체": name,
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
    diff_df["차이 절댓값"] = diff_df[
        "이용률 차이(%p)"
    ].abs()

    diff_df = diff_df.sort_values(
        "차이 절댓값", ascending=False
    )

    st.dataframe(
        diff_df.drop(columns="차이 절댓값").round(2),
        hide_index=True,
        width="stretch"
    )

    st.subheader("⑦ 학교급·학년과 AI 이용 여부")

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

    a, b = st.columns(2)

    with a:
        st.write("**학교급별 AI 이용률**")
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
            ).properties(height=300)
        )

    with b:
        st.write("**학년별 AI 이용률**")
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
            ).properties(height=300)
        )

    st.caption(
        "학년 코드는 초4~고3으로 표시합니다."
    )

    st.info(
        "분포와 상관관계는 탐색적 분석 결과입니다. "
        "실제 분류 성능은 검증 데이터에서 평가해야 합니다."
    )


# ==========================================
# 메뉴 3. 기준선
# ==========================================

elif menu == "③ 기준선":

    st.header("기준선(Baseline) 모델")

    st.subheader("① 기준선이란?")

    st.write(
        "기준선은 복잡한 머신러닝 모델을 사용하지 않아도 "
        "얻을 수 있는 기본적인 예측 성능입니다."
    )

    st.write(
        "우리 탐구에서는 학습 데이터에서 "
        "가장 많이 나타나는 AI 이용 여부를 "
        "모든 학생에게 똑같이 예측하는 방법을 사용합니다."
    )

    st.info(
        "기준선 모델: DummyClassifier(strategy='most_frequent')"
    )

    trained = train_all(X, Y.astype(int))
    baseline = trained["baseline"]

    majority = baseline["majority"]

    label = (
        "AI 이용함" if majority == 1
        else "AI 이용 안함"
    )

    st.subheader("② 기준선은 무엇을 예측할까?")

    st.success(
        f"학습 데이터에서 가장 많은 범주는 "
        f"**{label}**입니다. "
        f"따라서 기준선은 모든 학생을 "
        f"**{label}**으로 예측합니다."
    )

    st.write(
        "기준선은 SNS, 게임, 학년 등의 입력 특징을 "
        "활용하지 않고 가장 많은 범주만 예측합니다."
    )

    st.subheader("③ 기준선 검증 성능")

    scores = baseline["scores"]

    cols = st.columns(5)

    for col, (name, value) in zip(
        cols, scores.items()
    ):
        col.metric(name, f"{value:.3f}")

    st.write(
        f"기준선의 검증 정확도는 "
        f"**{scores['정확도'] * 100:.2f}%**이고 "
        f"F1 점수는 **{scores['F1']:.3f}**입니다."
    )

    st.subheader("④ 기준선 혼동행렬")

    matrix = confusion_matrix(
        trained["valid_y"],
        baseline["prediction"],
        labels=[0, 1]
    )

    labels = ["AI 미이용", "AI 이용"]

    cells = []

    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            cells.append({
                "실제": actual,
                "예측": predicted,
                "응답자 수": int(matrix[i, j])
            })

    matrix_df = pd.DataFrame(cells)

    base = alt.Chart(matrix_df).encode(
        x=alt.X("예측:N", sort=labels),
        y=alt.Y("실제:N", sort=labels)
    )

    heat = base.mark_rect().encode(
        color="응답자 수:Q",
        tooltip=["실제", "예측", "응답자 수"]
    )

    numbers = base.mark_text(
        fontSize=18
    ).encode(text="응답자 수:Q")

    show_chart(
        (heat + numbers).properties(height=280)
    )

    st.subheader("⑤ 기준선의 의미")

    st.write(
        "기준선보다 좋은 분류 모델을 만든다면 "
        "단순히 다수 범주만 예측하는 것보다 "
        "디지털 매체 이용 특성을 활용한 예측이 "
        "도움이 된다는 근거를 얻을 수 있습니다."
    )

    st.warning(
        "기준선은 모든 학생을 한 범주로 예측하므로 "
        "다른 범주의 학생을 제대로 구분할 수 없습니다. "
        "따라서 정확도뿐 아니라 F1 점수와 혼동행렬도 "
        "함께 살펴봐야 합니다."
    )


# ==========================================
# 메뉴 4. 모델 평가
# ==========================================

elif menu == "④ 모델 평가":

    st.header("분류 모델 학습 및 평가")

    trained = train_all(X, Y.astype(int))
    results = trained["results"]
    baseline = trained["baseline"]

    st.subheader("① 세 모델의 작동 원리")

    explanations = pd.DataFrame([
        [
            "로지스틱 회귀",
            "입력 변수에 가중치를 적용해 이용 확률 계산",
            "해석이 비교적 쉬운 기본 모델"
        ],
        [
            "의사결정트리",
            "질문을 통해 데이터를 단계적으로 분할",
            "분류 조건을 직접 확인 가능"
        ],
        [
            "랜덤 포레스트",
            "여러 의사결정트리의 예측 결과 종합",
            "복잡한 관계를 학습할 수 있음"
        ]
    ], columns=[
        "모델", "작동 원리", "선정 이유"
    ])

    st.dataframe(
        explanations, hide_index=True, width="stretch"
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
        f"{trained['counts']['최종 평가']:,}명"
    )

    st.write(
        "전체 데이터를 학습 60%, 검증 20%, "
        "최종 평가 20%로 나누었습니다."
    )

    st.info(
        "모든 후보 모델은 같은 학습·검증 데이터를 "
        "사용합니다. 최종 평가 데이터는 "
        "모델을 선정할 때 사용하지 않습니다."
    )

    st.subheader("③ 모델별 학습 과정 살펴보기")

    selected = st.selectbox(
        "살펴볼 모델",
        CANDIDATES
    )

    result = results[selected]
    pipe = result["pipeline"]
    model = pipe.named_steps["model"]

    if selected == "로지스틱 회귀":

        st.markdown("### 로지스틱 회귀")

        st.write(
            "입력 변수의 가중치를 합쳐 AI 이용 확률을 "
            "계산합니다. 확률이 0.5 이상이면 "
            "AI 이용함으로 예측합니다."
        )

        names = feature_names(pipe)

        coef_df = pd.DataFrame({
            "원래 변수": [
                original_name(n) for n in names
            ],
            "항목": [
                readable_name(n) for n in names
            ],
            "계수": model.coef_[0]
        })

        coef_df["절댓값"] = coef_df["계수"].abs()
        coef_df = coef_df.sort_values(
            "절댓값", ascending=False
        )

        show_chart(
            alt.Chart(coef_df).mark_bar().encode(
                x="계수:Q",
                y=alt.Y(
                    "항목:N",
                    sort=list(coef_df["항목"]),
                    axis=alt.Axis(labelLimit=350),
                    title=None
                ),
                tooltip=[
                    "원래 변수",
                    "항목",
                    alt.Tooltip("계수:Q", format="+.4f")
                ]
            ).properties(
                height=max(430, len(coef_df) * 30)
            )
        )

        st.dataframe(
            coef_df[
                ["원래 변수", "항목", "계수"]
            ].round(4),
            hide_index=True,
            width="stretch"
        )

        st.caption(
            "양수 계수는 이용 확률을 높이는 방향, "
            "음수 계수는 낮추는 방향의 관련성을 뜻합니다. "
            "인과관계는 아닙니다."
        )

    elif selected == "의사결정트리":

        st.markdown("### 의사결정트리")

        st.write(
            "학교급과 매체 이용 경험 등의 조건으로 "
            "학생들을 두 집단씩 반복해서 나눕니다."
        )

        st.write(
            "분류 과정을 실제 학습된 트리에서 "
            "한 단계씩 확인할 수 있습니다."
        )

        show_interactive_tree(pipe)

    else:

        st.markdown("### 랜덤 포레스트")

        st.write(
            "여러 의사결정트리를 학습하고 "
            "각 트리의 예측을 종합해 최종 결과를 결정합니다."
        )

        names = feature_names(pipe)

        imp_df = pd.DataFrame({
            "원래 변수": [
                original_name(n) for n in names
            ],
            "중요도": model.feature_importances_
        })

        grouped = (
            imp_df.groupby(
                "원래 변수", as_index=False
            )["중요도"]
            .sum()
            .sort_values("중요도", ascending=False)
        )

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
                    alt.Tooltip("중요도:Q", format=".4f")
                ]
            ).properties(height=390)
        )

        st.dataframe(
            grouped.round(4),
            hide_index=True,
            width="stretch"
        )

        st.caption(
            "중요도는 트리의 분할에 기여한 정도이며 "
            "실제 원인이나 영향력의 크기를 뜻하지 않습니다."
        )

    st.subheader("④ 선택한 모델의 검증 성능")

    cols = st.columns(5)

    for col, (name, value) in zip(
        cols, result["scores"].items()
    ):
        col.metric(name, f"{value:.3f}")

    st.subheader("⑤ 기준선과 세 모델 비교")

    summary = comparison_table(trained)

    st.dataframe(
        summary.round(3),
        hide_index=True,
        width="stretch"
    )

    metric_name = st.selectbox(
        "성능 비교 지표",
        METRICS,
        index=0
    )

    chart_data = summary[["모델", metric_name]]

    show_chart(
        alt.Chart(chart_data).mark_bar().encode(
            x=alt.X(
                "모델:N",
                sort=["기준선"] + CANDIDATES
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
    )

    st.subheader("⑥ 기준선보다 얼마나 나아졌을까?")

    base_acc = baseline["scores"]["정확도"]
    base_f1 = baseline["scores"]["F1"]

    differences = []

    for name, item in results.items():
        acc = item["scores"]["정확도"]
        f1 = item["scores"]["F1"]

        differences.append({
            "모델": name,
            "정확도 차이(%p)": (acc - base_acc) * 100,
            "F1 차이": f1 - base_f1
        })

    differences_df = pd.DataFrame(differences)

    st.dataframe(
        differences_df.round(3),
        hide_index=True,
        width="stretch"
    )

    for _, row in differences_df.iterrows():
        value = row["정확도 차이(%p)"]

        if value > 0:
            st.write(
                f"**{row['모델']}**: 기준선보다 "
                f"정확도가 **{value:.2f}%p 높습니다.**"
            )
        elif value < 0:
            st.write(
                f"**{row['모델']}**: 기준선보다 "
                f"정확도가 **{abs(value):.2f}%p 낮습니다.**"
            )
        else:
            st.write(
                f"**{row['모델']}**: "
                "기준선과 정확도가 같습니다."
            )

    st.subheader("⑦ 모델별 혼동행렬")

    selected_cm = st.selectbox(
        "혼동행렬을 확인할 모델",
        ["기준선"] + CANDIDATES
    )

    if selected_cm == "기준선":
        predictions = baseline["prediction"]
    else:
        predictions = results[selected_cm]["prediction"]

    cm = confusion_matrix(
        trained["valid_y"],
        predictions,
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
        x=alt.X("예측:N", sort=labels),
        y=alt.Y("실제:N", sort=labels)
    )

    heat = base.mark_rect().encode(
        color="응답자 수:Q",
        tooltip=["실제", "예측", "응답자 수"]
    )

    numbers = base.mark_text(
        fontSize=18
    ).encode(text="응답자 수:Q")

    show_chart(
        (heat + numbers).properties(height=290)
    )

    st.write(
        "혼동행렬에서 대각선 두 칸은 올바르게 예측한 "
        "응답자 수이고, 나머지는 잘못 예측한 수입니다."
    )

    st.subheader("⑧ 후보 모델 ROC 곡선")

    roc_rows = []

    for name, item in results.items():
        fpr, tpr, _ = roc_curve(
            trained["valid_y"],
            item["probability"]
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
            tooltip=[
                "모델",
                "거짓양성률",
                "참양성률"
            ]
        ).properties(height=320)
    )

    st.subheader("⑨ 최종 모델 선정")

    ranking = summary[
        summary["모델"] != "기준선"
    ].sort_values(
        ["F1", "ROC-AUC", "정확도"],
        ascending=False
    )

    st.dataframe(
        ranking.round(3),
        hide_index=True,
        width="stretch"
    )

    best = trained["best_name"]
    best_scores = results[best]["scores"]

    st.success(
        f"검증 데이터의 F1 점수를 주요 기준으로 "
        f"선정한 모델은 **{best}**입니다."
    )

    st.write(
        f"선정 모델의 검증 정확도는 "
        f"{best_scores['정확도']:.3f}, "
        f"F1 점수는 {best_scores['F1']:.3f}입니다."
    )

    if best == "로지스틱 회귀":
        st.write(
            "로지스틱 회귀는 비교적 단순하며 "
            "입력 항목의 계수를 살펴볼 수 있습니다."
        )
    elif best == "의사결정트리":
        st.write(
            "의사결정트리는 학생들을 나누는 조건을 "
            "단계별로 설명할 수 있습니다."
        )
    else:
        st.write(
            "랜덤 포레스트는 여러 결정트리의 예측을 "
            "종합하여 결과를 결정합니다."
        )

    st.subheader("⑩ 선정된 모델의 최종 평가")

    final = trained["final_scores"]
    final_base = trained["final_base_scores"]

    a, b, c = st.columns(3)

    a.metric(
        "선정 모델 정확도",
        f"{final['정확도'] * 100:.2f}%"
    )

    b.metric(
        "최종 기준선 정확도",
        f"{final_base['정확도'] * 100:.2f}%"
    )

    c.metric(
        "기준선 대비 차이",
        f"{(final['정확도'] - final_base['정확도']) * 100:+.2f}%p"
    )

    st.dataframe(
        pd.DataFrame([
            {"모델": "기준선", **final_base},
            {"모델": best, **final}
        ]).round(3),
        hide_index=True,
        width="stretch"
    )

    st.subheader("⑪ 탐구 결과 해석")

    improvement = final["정확도"] - final_base["정확도"]

    if improvement > 0:
        st.success(
            "선정한 모델은 최종 평가 데이터에서 "
            "기준선보다 높은 정확도를 보였습니다."
        )
    elif improvement < 0:
        st.warning(
            "선정한 모델은 최종 평가 데이터에서 "
            "기준선보다 정확도가 낮았습니다."
        )
    else:
        st.info(
            "선정한 모델과 기준선의 최종 정확도가 같습니다."
        )

    st.write(
        "이번 탐구에서는 청소년의 디지털 매체 이용 특성과 "
        "학교급·학년을 입력으로 사용하여 "
        "생성형 AI 이용 여부를 분류했습니다."
    )

    st.write(
        "기준선과 세 가지 분류 모델을 동일한 조건에서 "
        "비교하고, 검증 F1 점수를 기준으로 최종 모델을 "
        "선정한 뒤 별도의 평가 데이터로 확인했습니다."
    )

    st.info(
        "이번 결과는 해당 설문 데이터에서의 예측 성능입니다. "
        "매체 이용이 AI 이용의 원인이라는 의미는 아닙니다. "
        "또한 설문 가중치를 반영하지 않았으므로 "
        "전체 청소년 집단에 일반화할 때 주의해야 합니다."
    )
