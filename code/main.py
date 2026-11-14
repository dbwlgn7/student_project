
from pathlib import Path

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
    confusion_matrix
)


# ==========================================
# 1. 기본 설정
# ==========================================

st.set_page_config(
    page_title="청소년 생성형 AI 이용 분류",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    max-width: 1120px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}
h1, h2, h3 {
    letter-spacing: -.03em;
}
[data-testid="stMetricValue"] {
    font-size: 1.6rem;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

FILE = "청소년_AI이용분류_ID포함13열.csv"
ORIGINAL = "청소년_AI이용분류_정제데이터.csv"

KEEP = [1, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 36, 40]

MEDIA = [
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

NAMES = [
    "로지스틱 회귀",
    "의사결정트리",
    "랜덤 포레스트"
]

METRICS = [
    "정확도",
    "정밀도",
    "재현율",
    "F1",
    "ROC-AUC"
]


# ==========================================
# 2. 공통 화면 함수
# ==========================================

@st.cache_data
def read_csv(path):
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False
    )


def title(name, detail=None):
    st.subheader(name)

    if detail:
        st.caption(detail)


def numbers(items):
    cols = st.columns(len(items))

    for col, (name, value) in zip(cols, items.items()):
        col.metric(name, value)


def plot(chart):
    st.altair_chart(
        chart,
        use_container_width=True
    )


def metric_cards(scores):
    numbers({
        name: f"{scores[name]:.3f}"
        for name in METRICS
    })


# ==========================================
# 3. 평가 지표
# ==========================================

def evaluate(y, pred, prob):
    return {
        "정확도": accuracy_score(y, pred),
        "정밀도": precision_score(
            y, pred, zero_division=0
        ),
        "재현율": recall_score(
            y, pred, zero_division=0
        ),
        "F1": f1_score(
            y, pred, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y, prob)
    }


# ==========================================
# 4. 분류 모델
# ==========================================

def models():
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


# ==========================================
# 5. 성능 개선용 전처리
# ==========================================

def features_new(X):
    result = X.drop(columns="학교급").copy()

    result["이용 매체 개수"] = (
        X[MEDIA].sum(axis=1).astype(float)
    )

    return result


def make_model(name, improved=False):

    if improved:
        prep = ColumnTransformer([
            (
                "학년",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                ["학년"]
            ),
            (
                "매체",
                "passthrough",
                MEDIA
            ),
            (
                "개수",
                StandardScaler(),
                ["이용 매체 개수"]
            )
        ])

    else:
        prep = ColumnTransformer([
            (
                "범주",
                OneHotEncoder(
                    drop="if_binary",
                    handle_unknown="ignore"
                ),
                MEDIA + ["학교급", "학년"]
            )
        ])

    return Pipeline([
        ("전처리", prep),
        ("모델", models()[name])
    ])


def rank(scores):
    return (
        scores["F1"],
        scores["ROC-AUC"],
        scores["정확도"]
    )


# ==========================================
# 6. 기준선 및 모델 학습
# ==========================================

@st.cache_resource(
    show_spinner="기준선과 분류 모델을 학습하고 있습니다."
)
def learn(X, y):

    build, test = train_test_split(
        X.index,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    tr, valid = train_test_split(
        build,
        test_size=0.25,
        stratify=y.loc[build],
        random_state=42
    )

    # 기준선
    dummy = DummyClassifier(
        strategy="most_frequent"
    ).fit(
        X.loc[tr],
        y.loc[tr]
    )

    pred = dummy.predict(X.loc[valid])

    prob = dummy.predict_proba(
        X.loc[valid]
    )[:, list(dummy.classes_).index(1)]

    baseline = {
        "scores": evaluate(
            y.loc[valid], pred, prob
        ),
        "pred": pred,
        "majority": int(
            dummy.classes_[
                np.argmax(dummy.class_prior_)
            ]
        )
    }

    # 기존 모델과 개선 모델
    results = {}

    for setting, frame in [
        ("기존", X),
        ("개선", features_new(X))
    ]:

        for name in NAMES:

            pipe = make_model(
                name,
                setting == "개선"
            ).fit(
                frame.loc[tr],
                y.loc[tr]
            )

            p = pipe.predict(frame.loc[valid])

            pr = pipe.predict_proba(
                frame.loc[valid]
            )[:, 1]

            results[(setting, name)] = {
                "pipe": pipe,
                "pred": p,
                "scores": evaluate(
                    y.loc[valid], p, pr
                )
            }

    # F1 우선으로 모델 선정
    selected = max(
        results,
        key=lambda key: rank(
            results[key]["scores"]
        )
    )

    setting, name = selected

    frame = (
        features_new(X)
        if setting == "개선"
        else X
    )

    # 최종 평가용 모델
    final = make_model(
        name,
        setting == "개선"
    ).fit(
        frame.loc[build],
        y.loc[build]
    )

    fp = final.predict(frame.loc[test])

    fprob = final.predict_proba(
        frame.loc[test]
    )[:, 1]

    # 최종 평가용 기준선
    db = DummyClassifier(
        strategy="most_frequent"
    ).fit(
        X.loc[build],
        y.loc[build]
    )

    bp = db.predict(X.loc[test])

    bprob = db.predict_proba(
        X.loc[test]
    )[:, list(db.classes_).index(1)]

    return {
        "baseline": baseline,
        "results": results,
        "selected": selected,
        "valid_y": y.loc[valid],
        "final": evaluate(
            y.loc[test], fp, fprob
        ),
        "final_baseline": evaluate(
            y.loc[test], bp, bprob
        ),
        "sizes": (
            len(tr),
            len(valid),
            len(test)
        )
    }


# ==========================================
# 7. 혼동행렬
# ==========================================

def matrix_chart(y, pred):

    matrix = confusion_matrix(
        y,
        pred,
        labels=[0, 1]
    )

    labels = [
        "AI 미이용",
        "AI 이용"
    ]

    data = pd.DataFrame([
        {
            "실제": labels[i],
            "예측": labels[j],
            "학생 수": int(matrix[i, j]),
            "종류": (
                "맞힘" if i == j else "틀림"
            )
        }
        for i in range(2)
        for j in range(2)
    ])

    base = alt.Chart(data).encode(
        x=alt.X(
            "예측:N",
            sort=labels,
            title="모델 예측"
        ),
        y=alt.Y(
            "실제:N",
            sort=labels,
            title="실제 응답"
        )
    )

    tiles = base.mark_rect().encode(
        color=alt.Color(
            "종류:N",
            scale=alt.Scale(
                domain=["맞힘", "틀림"],
                range=["#2878a8", "#d68145"]
            ),
            legend=None
        ),
        tooltip=[
            "실제",
            "예측",
            "학생 수"
        ]
    )

    text = base.mark_text(
        fontSize=19,
        color="white"
    ).encode(
        text="학생 수:Q"
    )

    plot(
        (tiles + text).properties(height=250)
    )

    tn, fp, fn, tp = matrix.ravel()

    numbers({
        "올바른 예측": f"{tn+tp:,}명",
        "잘못된 예측": f"{fp+fn:,}명"
    })

    st.caption(
        f"실제 이용자 중 {tp:,}명 맞힘·"
        f"{fn:,}명 놓침 / "
        f"실제 미이용자 중 {tn:,}명 맞힘·"
        f"{fp:,}명 잘못 이용자로 예측"
    )

    return matrix


# ==========================================
# 8. 변수 이름 정리
# ==========================================

def simple_name(raw):
    name = raw.split("__", 1)[-1]

    for prefix in ["학교급_", "학년_"]:
        if name.startswith(prefix):
            return (
                prefix[:-1]
                + ": "
                + name[len(prefix):]
            )

    if name.endswith("_1"):
        return name[:-2] + " 이용함"

    if name.endswith("_0"):
        return name[:-2] + " 이용 안함"

    return name


# ==========================================
# 9. 의사결정트리 질문
# ==========================================

def tree_question(pipe, index, threshold):

    prep = pipe.named_steps["전처리"]

    raw = prep.get_feature_names_out()[index]

    short = raw.split("__", 1)[-1]

    if raw.startswith("개수__"):

        scaler = prep.named_transformers_["개수"]

        actual = (
            threshold * scaler.scale_[0]
            + scaler.mean_[0]
        )

        return (
            f"사용한 디지털 매체가 "
            f"{int(np.floor(actual + 1e-7))}개 "
            f"이하인가요?",
            "예",
            "아니요"
        )

    if raw.startswith("매체__"):
        return (
            f"{short}을 이용하나요?",
            "아니요",
            "예"
        )

    if short.endswith("_1"):
        return (
            f"{short[:-2]}을 이용하나요?",
            "아니요",
            "예"
        )

    if short.endswith("_0"):
        return (
            f"{short[:-2]}을 이용하지 않나요?",
            "아니요",
            "예"
        )

    if (
        short.startswith("학교급_")
        or short.startswith("학년_")
    ):
        return (
            f"{simple_name(raw)}에 해당하나요?",
            "아니요",
            "예"
        )

    return (
        f"{simple_name(raw)}이(가) "
        f"{threshold:.2f} 이하인가요?",
        "예",
        "아니요"
    )


def tree_explorer(pipe, setting):

    tree = pipe.named_steps["모델"].tree_

    state = "tree_path_" + setting

    if state not in st.session_state:
        st.session_state[state] = [0]

    path = st.session_state[state]

    if path[-1] >= tree.node_count:
        path[:] = [0]

    with st.expander(
        "의사결정트리 사용 방법",
        expanded=True
    ):
        st.write(
            "질문에 맞춰 '예' 또는 '아니요'를 "
            "누르면 다음 분류 단계로 이동합니다. "
            "학생 수와 AI 이용률은 학습 "
            "데이터에서 계산한 값입니다."
        )

    a, b = st.columns(2)

    if a.button(
        "처음부터",
        key="tree_reset_" + setting
    ):
        path[:] = [0]
        st.rerun()

    if b.button(
        "이전 단계",
        key="tree_back_" + setting,
        disabled=len(path) == 1
    ):
        path.pop()
        st.rerun()

    node = path[-1]

    value = tree.value[node][0]

    rate = (
        100 * value[1] / value.sum()
    )

    numbers({
        "분류 단계": str(len(path) - 1),
        "학습 응답자": (
            f"{tree.n_node_samples[node]:,}명"
        ),
        "집단 AI 이용률": f"{rate:.1f}%"
    })

    left = tree.children_left[node]
    right = tree.children_right[node]

    if left < 0:
        st.success(
            "최종 예측: "
            + (
                "AI 이용"
                if rate >= 50
                else "AI 미이용"
            )
        )
        return

    question, left_label, right_label = tree_question(
        pipe,
        tree.feature[node],
        tree.threshold[node]
    )

    st.info(
        "**현재 질문: " + question + "**"
    )

    for col, next_node, label in zip(
        st.columns(2),
        [left, right],
        [left_label, right_label]
    ):

        value = tree.value[next_node][0]

        percent = (
            100 * value[1] / value.sum()
        )

        with col:
            with st.container(border=True):

                st.write("**" + label + "**")

                st.write(
                    f"{tree.n_node_samples[next_node]:,}명 "
                    f"· AI 이용률 {percent:.1f}%"
                )

                if st.button(
                    label + " → 다음 단계",
                    key=(
                        f"tree_{setting}_"
                        f"{node}_{next_node}"
                    ),
                    use_container_width=True
                ):
                    path.append(int(next_node))
                    st.rerun()


# ==========================================
# 10. 열 선택·제외 근거
# ==========================================

def selection_reason(i):

    if i == 1:
        return (
            "응답자 확인용 ID이며 "
            "학습에는 사용하지 않음"
        )

    if i == 12:
        return (
            "생성형 AI 이용 여부: "
            "예측할 정답(Y)"
        )

    if i in [36, 40]:
        return (
            "학교급·학년을 입력(X)으로 사용"
        )

    if i in KEEP:
        return (
            "선택한 디지털 매체 이용 경험(X)"
        )

    if i == 2:
        return (
            "학교 식별 번호라 학습에서 제외"
        )

    if i in [19, 20, 21]:
        return (
            "AI 이용 답변이 포함될 수 있어 "
            "정보 누수 가능성 고려"
        )

    if i in [45, 46]:
        return (
            "설문 가중치: 이번 학습에는 "
            "적용하지 않음"
        )

    return (
        "연구 범위를 매체 이용 여부 "
        "중심으로 제한해 제외"
    )


# ==========================================
# 11. 데이터 불러오기와 검사
# ==========================================

st.sidebar.title("🧭 탐구 메뉴")

menu = st.sidebar.radio(
    "이동할 메뉴",
    [
        "① 질문",
        "② 데이터 분석",
        "③ 기준선",
        "④ 모델 평가",
        "⑤ 탐구 정리"
    ]
)

upload = st.sidebar.file_uploader(
    "13열 CSV 업로드 (선택)",
    type=["csv"]
)

if upload is not None:

    df = pd.read_csv(
        upload,
        encoding="utf-8-sig",
        low_memory=False
    )

elif (DATA / FILE).exists():

    df = read_csv(
        str(DATA / FILE)
    )

else:

    st.error(
        "data 폴더에서 13열 CSV를 "
        "찾지 못했습니다."
    )
    st.stop()


if (
    df.shape[1] != 13
    or "응답자 ID" not in str(df.columns[0])
    or "생성형 AI" not in str(df.columns[8])
    or "학교급" not in str(df.columns[11])
    or "학년별" not in str(df.columns[12])
):

    st.error(
        "CSV의 열 이름이나 순서가 "
        "예상과 다릅니다."
    )
    st.stop()


media_cols = (
    list(df.columns[1:8])
    + list(df.columns[9:11])
)

idcol = df.columns[0]

missing = int(
    df.isna().sum().sum()
)

dup = int(
    df[idcol].duplicated().sum()
)

invalid_details = []

for column, valid in (
    [
        (col, [1, 2])
        for col in media_cols + [df.columns[8]]
    ]
    + [
        (df.columns[11], [1, 2, 3]),
        (df.columns[12], list(range(1, 10)))
    ]
):

    value = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    bad = int(
        (
            df[column].notna()
            & ~value.isin(valid)
        ).sum()
    )

    invalid_details.append({
        "변수": column,
        "허용범위 밖 응답": bad
    })


invalid_df = pd.DataFrame(
    invalid_details
)

invalid = int(
    invalid_df[
        "허용범위 밖 응답"
    ].sum()
)

school_code = pd.to_numeric(
    df.iloc[:, 11],
    errors="coerce"
)

grade_code = pd.to_numeric(
    df.iloc[:, 12],
    errors="coerce"
)

mismatch = int(
    (
        (
            school_code
            != (grade_code - 1) // 3 + 1
        )
        & school_code.notna()
        & grade_code.notna()
    ).sum()
)


if missing or invalid or mismatch:

    st.error(
        "결측치 또는 응답 코드·학교급 오류를 "
        "발견했습니다. 자동 삭제하지 않고 "
        "분석을 중단합니다."
    )

    numbers({
        "결측 셀": missing,
        "허용범위 밖 응답": invalid,
        "학교급·학년 불일치": mismatch
    })

    st.dataframe(invalid_df)
    st.stop()


# ==========================================
# 12. 분석용 데이터 생성
# ==========================================

Xmedia = pd.DataFrame({
    name: pd.to_numeric(
        df[column]
    ).map({
        1: 1,
        2: 0
    })
    for name, column in zip(
        MEDIA,
        media_cols
    )
})

y = pd.to_numeric(
    df.iloc[:, 8]
).map({
    1: 1,
    2: 0
}).astype(int)

school = school_code.map(SCHOOL)

grade = grade_code.map(GRADE)

X = Xmedia.copy()

X["학교급"] = school
X["학년"] = grade


st.title(
    "청소년 생성형 AI 이용 분류 탐구"
)

st.caption(
    "탐구 질문 | 청소년의 디지털 매체 이용 "
    "특성으로 생성형 AI 이용 여부를 "
    "분류할 수 있을까?"
)


# ==========================================
# ① 질문
# ==========================================

if menu == "① 질문":

    with st.container(border=True):

        title("연구 질문")

        st.markdown(
            "### 디지털 매체 이용 특성으로 "
            "생성형 AI 이용 여부를 "
            "분류할 수 있을까?"
        )

        st.write(
            "9개 매체의 이용 경험과 "
            "학교급·학년으로 이용함/이용 안함을 "
            "예측합니다."
        )

    numbers({
        "응답자": f"{len(df):,}명",
        "모델 입력 변수": "11개",
        "예측 대상": "AI 이용 여부"
    })

    title("왜 분류를 선택했을까?")

    methods = pd.DataFrame([
        [
            "회귀",
            "연속적인 수치 예측",
            "이용 여부는 수치가 아니라 범주"
        ],
        [
            "분류",
            "정해진 범주 예측",
            "이용함/미이용 예측에 적합"
        ],
        [
            "군집",
            "정답 없이 비슷한 집단 묶기",
            "이미 정답(Y)이 존재"
        ]
    ], columns=[
        "방법",
        "목적",
        "우리 탐구와의 관계"
    ])

    st.dataframe(
        methods,
        hide_index=True,
        use_container_width=True
    )

    st.success(
        "정답이 두 범주이므로 "
        "지도학습의 이진 분류를 선택했습니다. "
        "AI 이용의 원인을 증명하는 연구는 아닙니다."
    )

    title("데이터 출처와 변수")

    st.write(
        "**2024년 청소년 매체이용 및 "
        "유해환경 실태조사** 자료를 "
        "바탕으로 구성한 CSV를 이용했습니다. "
        "GitHub data 폴더의 원본 자료와 "
        "코드북을 참고합니다."
    )

    types = pd.DataFrame([
        [
            "응답자 ID",
            "식별자",
            "중복 검사·학습에서 제외"
        ],
        [
            "매체 이용 9개",
            "이진 범주형",
            "입력 X"
        ],
        [
            "학교급·학년",
            "순서 있는 범주형",
            "입력 X"
        ],
        [
            "생성형 AI 이용 여부",
            "이진 범주형",
            "정답 Y"
        ]
    ], columns=[
        "자료 속성",
        "유형",
        "역할"
    ])

    st.dataframe(
        types,
        hide_index=True,
        use_container_width=True
    )

    title("비교 모델")

    information = [
        (
            "로지스틱 회귀",
            "변수의 계수로 이용 확률을 계산"
        ),
        (
            "의사결정트리",
            "질문과 조건으로 응답자를 분류"
        ),
        (
            "랜덤 포레스트",
            "여러 트리의 예측을 종합"
        )
    ]

    for name, explanation in information:

        with st.container(border=True):

            st.write(
                "**" + name + "** — "
                + explanation
            )


# ==========================================
# ② 데이터 분석
# ==========================================

elif menu == "② 데이터 분석":

    tab1, tab2, tab3 = st.tabs([
        "기초통계",
        "데이터 전처리",
        "시각화·관계 분석"
    ])

    # --------------------------------------
    # 기초통계
    # --------------------------------------

    with tab1:

        title("전체 데이터와 기초통계")

        numbers({
            "응답자(행)": f"{len(df):,}",
            "선택 변수(열)": 13,
            "결측 셀": missing,
            "중복 ID": dup
        })

        numbers({
            "AI 이용": f"{int(y.sum()):,}명",
            "AI 미이용": (
                f"{int((1-y).sum()):,}명"
            ),
            "AI 이용률": (
                f"{y.mean()*100:.1f}%"
            )
        })

        freq = pd.DataFrame([
            {
                "매체": name,
                "이용자 수": int(
                    Xmedia[name].sum()
                ),
                "미이용자 수": int(
                    (1-Xmedia[name]).sum()
                ),
                "이용률(%)": round(
                    Xmedia[name].mean()*100,
                    2
                )
            }
            for name in MEDIA
        ]).sort_values(
            "이용률(%)",
            ascending=False
        )

        title("매체별 빈도와 비율")

        st.dataframe(
            freq,
            hide_index=True,
            use_container_width=True
        )

        a, b = st.columns(2)

        with a:

            st.write("**학교급별 응답자**")

            st.bar_chart(
                school.value_counts().reindex(
                    list(SCHOOL.values()),
                    fill_value=0
                )
            )

        with b:

            st.write("**학년별 응답자**")

            st.bar_chart(
                grade.value_counts().reindex(
                    list(GRADE.values()),
                    fill_value=0
                )
            )

        st.caption(
            "코드 1~9를 초4~고3으로 바꿔 표시합니다. "
            "범주형 응답 코드의 평균보다 "
            "빈도·비율을 확인합니다."
        )

    # --------------------------------------
    # 데이터 전처리
    # --------------------------------------

    with tab2:

        title("전처리와 검증 근거")

        numbers({
            "원본 열": "46개",
            "선택 열": "13개",
            "학습 입력": "11개"
        })

        numbers({
            "결측 셀": missing,
            "이상 응답 코드": invalid,
            "학년·학교급 불일치": mismatch,
            "중복 ID": dup
        })

        st.write(
            "원본 응답은 바꾸지 않고 "
            "필요한 열만 선택했습니다. "
            "허용 범위 밖의 응답과 "
            "학교급·학년 불일치를 검사했습니다. "
            "응답자 ID가 크다고 이상치로 보지 않습니다."
        )

        with st.expander(
            "코드별 검사 결과"
        ):
            st.dataframe(
                invalid_df,
                hide_index=True,
                use_container_width=True
            )

        if (DATA / ORIGINAL).exists():

            original = read_csv(
                str(DATA / ORIGINAL)
            )

            subset = original.iloc[
                :,
                [i-1 for i in KEEP]
            ].reset_index(drop=True)

            actual = df.reset_index(drop=True)

            identical = (
                subset.columns.tolist()
                == actual.columns.tolist()
                and subset.astype("string")
                .fillna("__NA__")
                .equals(
                    actual.astype("string")
                    .fillna("__NA__")
                )
            )

            if identical:

                st.success(
                    "원본과 선택 CSV의 "
                    "열 이름·순서·값이 일치합니다."
                )

            else:

                st.warning(
                    "원본과 선택 CSV가 완전히 "
                    "일치하지 않습니다. "
                    "다시 확인해야 합니다."
                )

            with st.expander(
                "원본 46열 선택·제외 사유"
            ):

                report = pd.DataFrame([
                    {
                        "번호": i,
                        "원본 변수명": column,
                        "처리": (
                            "유지"
                            if i in KEEP
                            else "제외"
                        ),
                        "사유": selection_reason(i)
                    }
                    for i, column in enumerate(
                        original.columns,
                        1
                    )
                ])

                st.dataframe(
                    report,
                    hide_index=True,
                    use_container_width=True
                )

        else:

            st.warning(
                "원본 46열 CSV가 없어 "
                "전후 비교를 표시할 수 없습니다."
            )

        st.info(
            "학습 시 분석용 복사본에서 "
            "이용함=1, 미이용=0으로 바꾸고 "
            "학교급·학년을 원-핫 인코딩합니다. "
            "정답 Y와 ID는 X에 넣지 않습니다."
        )

    # --------------------------------------
    # 시각화와 관계 분석
    # --------------------------------------

    with tab3:

        title("매체별 이용률")

        rates = pd.DataFrame({
            "매체": MEDIA,
            "이용률(%)": [
                Xmedia[name].mean()*100
                for name in MEDIA
            ]
        }).sort_values(
            "이용률(%)",
            ascending=False
        )

        plot(
            alt.Chart(rates).mark_bar().encode(
                x=alt.X(
                    "이용률(%):Q",
                    scale=alt.Scale(
                        domain=[0, 100]
                    )
                ),
                y=alt.Y(
                    "매체:N",
                    sort="-x",
                    title=None
                ),
                tooltip=[
                    "매체",
                    alt.Tooltip(
                        "이용률(%):Q",
                        format=".1f"
                    )
                ]
            ).properties(height=340)
        )

        st.write(
            f"이용률이 가장 높은 매체는 "
            f"**{rates.iloc[0]['매체']} "
            f"({rates.iloc[0]['이용률(%)']:.1f}%)**입니다."
        )

        title(
            "매체 경험(X)과 AI 이용(Y)의 상관계수"
        )

        corr = pd.DataFrame({
            "매체": MEDIA,
            "상관계수": [
                Xmedia[name].corr(y)
                for name in MEDIA
            ]
        })

        corr["절댓값"] = (
            corr["상관계수"].abs()
        )

        corr = corr.sort_values(
            "절댓값",
            ascending=False
        )

        plot(
            alt.Chart(corr).mark_bar().encode(
                x=alt.X(
                    "상관계수:Q",
                    scale=alt.Scale(
                        domain=[-1, 1]
                    )
                ),
                y=alt.Y(
                    "매체:N",
                    sort=list(corr["매체"]),
                    title=None
                ),
                tooltip=[
                    "매체",
                    alt.Tooltip(
                        "상관계수:Q",
                        format=".3f"
                    )
                ]
            ).properties(height=340)
        )

        rows = []

        for name in MEDIA:

            for label, value in [
                ("AI 이용자", 1),
                ("AI 미이용자", 0)
            ]:

                rows.append({
                    "매체": name,
                    "집단": label,
                    "이용률(%)": (
                        100 * Xmedia.loc[
                            y == value,
                            name
                        ].mean()
                    )
                })

        title(
            "AI 이용자와 미이용자의 "
            "매체 이용률 비교"
        )

        grouped = pd.DataFrame(rows)

        plot(
            alt.Chart(grouped).mark_bar().encode(
                x=alt.X(
                    "이용률(%):Q",
                    scale=alt.Scale(
                        domain=[0, 100]
                    )
                ),
                y="매체:N",
                yOffset="집단:N",
                color="집단:N",
                tooltip=[
                    "매체",
                    "집단",
                    alt.Tooltip(
                        "이용률(%):Q",
                        format=".1f"
                    )
                ]
            ).properties(height=420)
        )

        pivot = grouped.pivot(
            index="매체",
            columns="집단",
            values="이용률(%)"
        )

        differences = (
            pivot["AI 이용자"]
            - pivot["AI 미이용자"]
        ).abs()

        largest = differences.idxmax()

        st.success(
            f"두 집단의 이용률 차이가 "
            f"가장 큰 매체는 **{largest}**로, "
            f"차이는 **{differences.max():.1f}%p**입니다."
        )

        title("학교급·학년별 AI 이용률")

        groups = [
            (
                "학교급",
                school,
                list(SCHOOL.values())
            ),
            (
                "학년",
                grade,
                list(GRADE.values())
            )
        ]

        for label, column, order in groups:

            data = pd.DataFrame({
                label: column,
                "AI 이용": y
            }).groupby(label)[
                "AI 이용"
            ].mean().mul(100).reindex(
                order
            ).reset_index(
                name="AI 이용률(%)"
            )

            st.write("**" + label + "**")

            plot(
                alt.Chart(data).mark_bar().encode(
                    x=alt.X(
                        f"{label}:N",
                        sort=order,
                        axis=alt.Axis(
                            labelAngle=0
                        )
                    ),
                    y=alt.Y(
                        "AI 이용률(%):Q",
                        scale=alt.Scale(
                            domain=[0, 100]
                        )
                    ),
                    tooltip=[
                        label,
                        alt.Tooltip(
                            "AI 이용률(%):Q",
                            format=".1f"
                        )
                    ]
                ).properties(height=260)
            )

        st.info(
            "상관관계는 예측에 관련된 경향을 "
            "보여주지만, AI 이용의 원인을 "
            "증명하지 않습니다."
        )


# ==========================================
# ③ 기준선
# ==========================================

elif menu == "③ 기준선":

    learned = learn(X, y)

    baseline = learned["baseline"]

    label = (
        "AI 이용"
        if baseline["majority"] == 1
        else "AI 미이용"
    )

    with st.container(border=True):

        title("기준선의 예측 방법")

        st.markdown(
            "### 모든 학생을 '"
            + label
            + "'으로 예측"
        )

        st.write(
            "학습 데이터에서 더 많은 범주로 "
            "모든 학생을 예측합니다. "
            "매체 이용 경험이나 학년을 활용하지 않습니다."
        )

    title("기준선 검증 성능")

    metric_cards(
        baseline["scores"]
    )

    title(
        "기준선 혼동행렬",
        "실제 이용 여부와 예측 결과의 "
        "네 가지 경우를 비교합니다."
    )

    matrix = matrix_chart(
        learned["valid_y"],
        baseline["pred"]
    )

    if baseline["majority"] == 1:

        st.warning(
            f"기준선은 실제 미이용자 "
            f"**{matrix[0,1]:,}명**도 "
            f"모두 이용자로 예측했습니다. "
            f"재현율이 높아도 미이용자를 "
            f"구분하지 못합니다."
        )

    else:

        st.warning(
            f"기준선은 실제 이용자 "
            f"**{matrix[1,0]:,}명**을 놓쳤습니다."
        )

    with st.expander(
        "정확도·정밀도·재현율·F1·ROC-AUC 뜻"
    ):

        st.write(
            "**정확도:** 전체 예측 중 맞힌 비율."
        )

        st.write(
            "**정밀도:** 이용자로 예측한 사람 중 "
            "실제 이용자 비율."
        )

        st.write(
            "**재현율:** 실제 이용자를 찾아낸 비율."
        )

        st.write(
            "**F1:** 정밀도와 재현율의 균형."
        )

        st.write(
            "**ROC-AUC:** 두 범주를 구분하는 능력."
        )


# ==========================================
# ④ 모델 평가
# ==========================================

elif menu == "④ 모델 평가":

    learned = learn(X, y)

    results = learned["results"]

    baseline_scores = (
        learned["baseline"]["scores"]
    )

    chosen = learned["selected"]

    chosen_scores = (
        results[chosen]["scores"]
    )

    with st.container(border=True):

        st.write(
            "**검증 데이터에서 선정한 후보**"
        )

        st.markdown(
            f"### {chosen[1]} "
            f"({chosen[0]} 전처리)"
        )

        numbers({
            "선정 모델 F1": (
                f"{chosen_scores['F1']:.3f}"
            ),
            "기준선 F1": (
                f"{baseline_scores['F1']:.3f}"
            ),
            "정확도 차이": (
                f"{(chosen_scores['정확도']-baseline_scores['정확도'])*100:+.2f}%p"
            )
        })

        st.caption(
            "같은 검증 데이터에서 "
            "기준선과 기존·개선 총 6개 후보 비교. "
            "F1, ROC-AUC, 정확도 순으로 선정."
        )

    tab1, tab2, tab3, tab4 = st.tabs([
        "성능 비교",
        "혼동행렬",
        "모델별 원리",
        "성능 개선·최종 선정"
    ])

    # --------------------------------------
    # 성능 비교
    # --------------------------------------

    with tab1:

        title("기준선과 기존 세 모델")

        comparison = pd.DataFrame(
            [
                {
                    "모델": "기준선",
                    **baseline_scores
                }
            ]
            + [
                {
                    "모델": name,
                    **results[
                        ("기존", name)
                    ]["scores"]
                }
                for name in NAMES
            ]
        )

        comparison["기준선 대비 정확도(%p)"] = (
            comparison["정확도"]
            - baseline_scores["정확도"]
        ) * 100

        comparison["기준선 대비 F1"] = (
            comparison["F1"]
            - baseline_scores["F1"]
        )

        st.dataframe(
            comparison.round(3),
            hide_index=True,
            use_container_width=True
        )

        for metric in ["정확도", "F1"]:

            st.write(
                "**" + metric + " 비교**"
            )

            plot(
                alt.Chart(comparison).mark_bar().encode(
                    x=alt.X(
                        f"{metric}:Q",
                        scale=alt.Scale(
                            domain=[0, 1]
                        )
                    ),
                    y=alt.Y(
                        "모델:N",
                        sort=["기준선"] + NAMES,
                        title=None
                    ),
                    color="모델:N",
                    tooltip=[
                        "모델",
                        alt.Tooltip(
                            f"{metric}:Q",
                            format=".3f"
                        )
                    ]
                ).properties(height=220)
            )

        st.info(
            "정확도와 F1이 각각 의미하는 바가 "
            "다르므로 둘 다 기준선과 비교합니다. "
            "F1을 이번 연구의 우선 "
            "선정 기준으로 정했습니다."
        )

    # --------------------------------------
    # 혼동행렬
    # --------------------------------------

    with tab2:

        title(
            "모델을 선택해 혼동행렬 보기"
        )

        options = (
            ["기준선"]
            + [
                f"{setting} · {name}"
                for setting in ["기존", "개선"]
                for name in NAMES
            ]
        )

        selected = st.selectbox(
            "혼동행렬 대상",
            options
        )

        if selected == "기준선":

            pred = learned["baseline"]["pred"]

        else:

            setting, name = selected.split(
                " · ",
                1
            )

            pred = results[
                (setting, name)
            ]["pred"]

        matrix_chart(
            learned["valid_y"],
            pred
        )

        st.caption(
            "파란색=맞힘, 주황색=틀림. "
            "모델의 정확도 하나만으로는 "
            "어느 집단을 놓치는지 알기 어렵습니다."
        )

    # --------------------------------------
    # 모델별 원리
    # --------------------------------------

    with tab3:

        title(
            "모델의 원리와 학습한 결과"
        )

        info = pd.DataFrame([
            [
                "로지스틱 회귀",
                "이용 확률을 계산",
                "계수의 방향 해석 가능"
            ],
            [
                "의사결정트리",
                "조건으로 데이터를 분할",
                "분류 과정을 보기 쉬움"
            ],
            [
                "랜덤 포레스트",
                "여러 트리의 판단을 종합",
                "복잡한 관계 학습 가능"
            ]
        ], columns=[
            "모델",
            "작동 방식",
            "특징"
        ])

        st.dataframe(
            info,
            hide_index=True,
            use_container_width=True
        )

        setting = st.radio(
            "전처리 설정",
            ["기존", "개선"],
            horizontal=True
        )

        name = st.selectbox(
            "자세히 볼 모델",
            NAMES
        )

        pipe = results[
            (setting, name)
        ]["pipe"]

        algorithm = pipe.named_steps["모델"]

        feature_names = (
            pipe.named_steps["전처리"]
            .get_feature_names_out()
        )

        if name == "의사결정트리":

            tree_explorer(
                pipe,
                setting
            )

        elif name == "로지스틱 회귀":

            coefficients = pd.DataFrame({
                "변수 항목": [
                    simple_name(item)
                    for item in feature_names
                ],
                "계수": algorithm.coef_[0]
            })

            coefficients["크기"] = (
                coefficients["계수"].abs()
            )

            coefficients = (
                coefficients.sort_values(
                    "크기",
                    ascending=False
                )
            )

            top = coefficients.head(12)

            plot(
                alt.Chart(top).mark_bar().encode(
                    x="계수:Q",
                    y=alt.Y(
                        "변수 항목:N",
                        sort=list(
                            top["변수 항목"]
                        ),
                        axis=alt.Axis(
                            labelLimit=330
                        ),
                        title=None
                    ),
                    tooltip=[
                        "변수 항목",
                        alt.Tooltip(
                            "계수:Q",
                            format="+.4f"
                        )
                    ]
                ).properties(height=400)
            )

            with st.expander(
                "전체 계수 표 보기"
            ):

                st.dataframe(
                    coefficients[
                        ["변수 항목", "계수"]
                    ].round(4),
                    hide_index=True,
                    use_container_width=True
                )

            st.caption(
                "양수·음수 계수는 다른 조건이 같을 때 "
                "관련 방향을 나타냅니다. "
                "인과관계는 아닙니다."
            )

        else:

            importance = pd.DataFrame({
                "원래 변수": [
                    simple_name(item)
                    .split(": ")[0]
                    .replace(" 이용함", "")
                    .replace(" 이용 안함", "")
                    for item in feature_names
                ],
                "중요도": (
                    algorithm.feature_importances_
                )
            })

            importance = (
                importance.groupby(
                    "원래 변수",
                    as_index=False
                )["중요도"]
                .sum()
                .sort_values(
                    "중요도",
                    ascending=False
                )
            )

            plot(
                alt.Chart(importance).mark_bar().encode(
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
                ).properties(height=340)
            )

            st.caption(
                "중요도는 트리의 분할에 기여한 "
                "정도이지 원인이라는 뜻은 아닙니다."
            )

        title(
            "선택한 모델의 검증 평가 지표"
        )

        metric_cards(
            results[
                (setting, name)
            ]["scores"]
        )

    # --------------------------------------
    # 성능 개선 및 최종 선정
    # --------------------------------------

    with tab4:

        title(
            "성능 개선 실험: 전처리 전후"
        )

        with st.container(border=True):

            st.write(
                "**변경한 점:** 기존에는 매체 9개, "
                "학교급, 학년을 사용했습니다. "
                "개선 모델에서는 학년과 중복되는 "
                "학교급을 빼고, 9개 매체 중 "
                "이용한 매체 개수를 추가했습니다."
            )

            st.caption(
                "원본 행과 정답은 수정하지 않았습니다. "
                "같은 학습·검증 자료와 모델 설정으로 "
                "비교했습니다."
            )

        differences = []

        for name in NAMES:

            old = results[
                ("기존", name)
            ]["scores"]

            new = results[
                ("개선", name)
            ]["scores"]

            differences.append({
                "모델": name,
                "기존 정확도": old["정확도"],
                "개선 정확도": new["정확도"],
                "정확도 차이(%p)": (
                    100 * (
                        new["정확도"]
                        - old["정확도"]
                    )
                ),
                "기존 F1": old["F1"],
                "개선 F1": new["F1"],
                "F1 차이": (
                    new["F1"] - old["F1"]
                )
            })

        differences = pd.DataFrame(
            differences
        )

        st.dataframe(
            differences.round(3),
            hide_index=True,
            use_container_width=True
        )

        long = pd.DataFrame([
            {
                "모델": name,
                "설정": setting,
                "F1": results[
                    (setting, name)
                ]["scores"]["F1"]
            }
            for setting in ["기존", "개선"]
            for name in NAMES
        ])

        plot(
            alt.Chart(long).mark_bar().encode(
                x=alt.X(
                    "모델:N",
                    sort=NAMES,
                    axis=alt.Axis(
                        labelAngle=0
                    )
                ),
                xOffset="설정:N",
                y=alt.Y(
                    "F1:Q",
                    scale=alt.Scale(
                        domain=[0, 1]
                    )
                ),
                color="설정:N",
                tooltip=[
                    "모델",
                    "설정",
                    alt.Tooltip(
                        "F1:Q",
                        format=".3f"
                    )
                ]
            ).properties(height=320)
        )

        better_count = int(
            (differences["F1 차이"] > 0).sum()
        )

        if better_count:

            st.success(
                f"세 모델 중 **{better_count}개**에서 "
                f"변경 후 F1이 높아졌습니다."
            )

        else:

            st.warning(
                "이번 전처리 변경에서 "
                "F1이 높아진 모델이 없습니다. "
                "개선 성공이라고 말하지 않습니다."
            )

        title(
            "최종 모델 선정 및 별도 평가"
        )

        st.write(
            f"검증 F1이 가장 높은 설정은 "
            f"**{chosen[1]} ({chosen[0]})**입니다. "
            f"최종 평가 데이터는 모델 선정에 "
            f"사용하지 않았습니다."
        )

        final = learned["final"]

        final_baseline = (
            learned["final_baseline"]
        )

        final_table = pd.DataFrame([
            {
                "모델": "최종 기준선",
                **final_baseline
            },
            {
                "모델": (
                    f"선정: {chosen[1]} "
                    f"({chosen[0]})"
                ),
                **final
            }
        ])

        st.dataframe(
            final_table.round(3),
            hide_index=True,
            use_container_width=True
        )

        numbers({
            "최종 정확도 차이": (
                f"{100*(final['정확도']-final_baseline['정확도']):+.2f}%p"
            ),
            "최종 F1 차이": (
                f"{final['F1']-final_baseline['F1']:+.3f}"
            )
        })

        st.caption(
            "이 결과는 해당 설문 데이터에서의 "
            "예측 성능입니다. 인과관계를 "
            "입증하지 않으며 설문 가중치도 "
            "적용하지 않았습니다."
        )


# ==========================================
# ⑤ 탐구 정리
# ==========================================

else:

    learned = learn(X, y)

    results = learned["results"]

    chosen = learned["selected"]

    baseline = learned["final_baseline"]

    final = learned["final"]

    tab1, tab2, tab3, tab4 = st.tabs([
        "활동 과정",
        "결과 해석",
        "한계·추가 탐구",
        "느낀 점"
    ])

    # --------------------------------------
    # 활동 과정
    # --------------------------------------

    with tab1:

        title(
            "이번 탐구의 실제 진행 과정"
        )

        st.write(
            "**탐구 질문 설정 → 자료 선택 → "
            "46열에서 13열 선택 → "
            "전처리와 시각화 → 기준선 만들기 → "
            "세 분류 모델 평가 → "
            "전처리 변경 실험 → 최종 평가**"
        )

        title(
            "직접 확인하고 수정한 부분"
        )

        st.write(
            "- 매체끼리의 관계보다 "
            "**매체 이용(X)과 AI 이용 여부(Y)**의 "
            "관계가 탐구에 적절하다고 판단해 "
            "분석 방향을 수정했습니다."
        )

        st.write(
            "- 학년 코드 1~9를 "
            "**초4~고3**으로 표시하도록 요구했습니다."
        )

        st.write(
            "- 그래프에서 변수명이 잘리거나 "
            "트리 그림이 이해하기 어려운 문제를 발견해 "
            "설명 방식을 수정했습니다."
        )

        st.write(
            "- 정확도만 보지 않고 "
            "기준선과 F1을 비교하는 화면을 "
            "추가했습니다."
        )

    # --------------------------------------
    # 결과 해석
    # --------------------------------------

    with tab2:

        title(
            "결과에서 확인한 사실"
        )

        numbers({
            "최종 선택": chosen[1],
            "선택한 전처리": chosen[0],
            "최종 정확도": (
                f"{final['정확도']*100:.1f}%"
            )
        })

        accuracy_difference = (
            100 * (
                final["정확도"]
                - baseline["정확도"]
            )
        )

        f1_difference = (
            final["F1"]
            - baseline["F1"]
        )

        st.write(
            f"별도 최종 평가에서 기준선과 "
            f"비교했을 때 정확도 차이는 "
            f"**{accuracy_difference:+.2f}%p**, "
            f"F1 차이는 **{f1_difference:+.3f}**입니다."
        )

        if (
            accuracy_difference > 0
            and f1_difference > 0
        ):

            st.success(
                "두 지표 모두 기준선보다 높았습니다. "
                "이 설문 데이터에서 매체 이용 특성은 "
                "단순한 다수 범주 예측보다 "
                "더 나은 분류에 도움이 되었습니다."
            )

        elif accuracy_difference > 0:

            st.warning(
                "정확도는 기준선보다 높지만 "
                "F1까지 개선되지는 않았습니다. "
                "모든 지표에서 좋아졌다고 "
                "말할 수 없습니다."
            )

        else:

            st.warning(
                "정확도는 기준선보다 높지 않았습니다. "
                "이번 결과만으로 예측 성능 개선을 "
                "단정할 수 없습니다."
            )

        st.write(
            f"**한계도 함께 해석해야 합니다.** "
            f"최종 정확도가 "
            f"{final['정확도']*100:.1f}%라는 것은 "
            f"모든 학생의 이용 여부를 "
            f"정확히 예측한 것은 아니라는 뜻입니다."
        )

    # --------------------------------------
    # 한계 및 추가 탐구
    # --------------------------------------

    with tab3:

        title("현재 연구의 한계")

        st.write(
            "- 사용한 정보는 "
            "**매체 이용 경험(이용함/미이용)**과 "
            "학교급·학년 중심이어서 "
            "사용 시간, 이용 목적 같은 세부 차이를 "
            "직접 반영하지 못했습니다."
        )

        st.write(
            "- 모델의 정확도만으로 개인의 "
            "AI 이용 여부를 확실히 판단할 수 없습니다."
        )

        st.write(
            "- 설문 응답에서 발견한 관련성은 "
            "**원인과 결과**를 증명하지 않습니다."
        )

        st.write(
            "- 설문 가중치를 적용하지 않았으므로 "
            "전체 청소년 집단으로 결과를 "
            "일반화할 때 주의해야 합니다."
        )

        st.write(
            "- 한 번의 데이터 분할에서 얻은 "
            "성능이므로 다른 분할이나 "
            "다른 연도에서 달라질 수 있습니다."
        )

        title(
            "추가 탐구 아이디어 — 아직 수행하지 않음"
        )

        st.write(
            "**학교급별 성능 평가**, "
            "**다른 연도 자료로 재검증**, "
            "**이용 시간·이용 목적 같은 "
            "추가 변수의 효과 분석**을 "
            "다음 탐구로 고려할 수 있습니다."
        )

        st.caption(
            "현재 완료한 실험이 아니라 "
            "앞으로의 연구 제안입니다."
        )

    # --------------------------------------
    # 느낀 점
    # --------------------------------------

    with tab4:

        title(
            "탐구하면서 느낀 점"
        )

        st.write(
            "원본 데이터에는 우리 주제와 "
            "직접 관련 없는 변수가 많아 "
            "필요한 것만 고르는 과정이 "
            "생각보다 쉽지 않았습니다."
        )

        st.write(
            "처음에는 세 모델 중 하나쯤 "
            "정확도가 80%를 넘을 것으로 "
            "예상했지만, 이전에 확인한 "
            "기존 세 모델의 검증 정확도는 "
            "모두 70% 미만이어서 예상과 달랐습니다."
        )

        st.write(
            "이를 통해 변수를 선택해 "
            "모델을 학습한다고 해서 "
            "반드시 높은 정확도가 나오는 것은 "
            "아니라는 점을 알게 되었습니다."
        )

        st.write(
            "다만 기존 모델은 기준선보다 "
            "정확도가 높았으므로 "
            "**선택한 변수가 전혀 도움이 되지 않았다**고 "
            "해석하는 것도 적절하지 않습니다."
        )

        st.write(
            "실제로 어떤 점이 개선되고 "
            "어떤 한계가 남았는지 "
            "함께 확인하는 것이 중요합니다."
        )

        st.caption(
            "이 글은 직접 이야기한 경험을 "
            "바탕으로 정리한 초안입니다. "
            "본인의 실제 생각과 일치하는지 "
            "확인 후 제출하세요. "
            "70% 미만은 이전에 확인한 "
            "'기존 세 모델' 결과를 뜻하며 "
            "개선 모델의 성능과 구분합니다."
        )
