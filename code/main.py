
from pathlib import Path

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
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)


# ==========================================
# 1. 기본 설정
# ==========================================

st.set_page_config(
    page_title="청소년 AI 이용 분류 탐구",
    page_icon="📊",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent.parent

SELECTED = (
    ROOT / "data" / "청소년_AI이용분류_ID포함13열.csv"
)

ORIGINAL = (
    ROOT / "data" / "청소년_AI이용분류_정제데이터.csv"
)

KEEP = [1, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 36, 40]

MEDIA = [
    "TV방송", "OTT", "인터넷 개인방송",
    "숏폼", "포털 플랫폼", "SNS",
    "메타버스", "온라인 게임", "웹툰"
]

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

MODELS = [
    "로지스틱 회귀",
    "의사결정트리",
    "랜덤 포레스트"
]

METRICS = [
    "정확도", "정밀도", "재현율",
    "F1", "ROC-AUC"
]


# ==========================================
# 2. 디자인
# ==========================================

st.markdown("""
<style>
.block-container {
    max-width: 1120px;
    padding-top: 1.6rem;
    padding-bottom: 4rem;
}

h1, h2, h3 {
    letter-spacing: -.025em;
}

[data-testid="stMetricValue"] {
    font-size: 1.6rem;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. 공통 함수
# ==========================================

@st.cache_data
def read_csv(path):
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False
    )


def heading(title, caption=None):
    st.subheader(title)

    if caption:
        st.caption(caption)


def cards(items):
    cols = st.columns(len(items))

    for col, (label, value) in zip(
        cols, items.items()
    ):
        col.metric(label, value)


def chart(spec):
    st.altair_chart(
        spec,
        use_container_width=True
    )


def bars(df, x, y, order=None, max_y=None):

    y_axis = alt.Y(
        f"{y}:Q",
        scale=(
            alt.Scale(domain=[0, max_y])
            if max_y
            else alt.Scale(zero=True)
        )
    )

    return alt.Chart(df).mark_bar(
        cornerRadiusTopLeft=4,
        cornerRadiusTopRight=4
    ).encode(
        x=alt.X(
            f"{x}:N",
            sort=order,
            axis=alt.Axis(
                labelAngle=0,
                labelLimit=140
            )
        ),
        y=y_axis,
        tooltip=[
            alt.Tooltip(f"{x}:N"),
            alt.Tooltip(f"{y}:Q", format=".3f")
        ]
    ).properties(height=285)


def reason(i):

    if i == 1:
        return (
            "응답자를 식별하고 중복을 확인하되 "
            "학습 입력에서는 제외"
        )

    if i == 12:
        return "예측할 정답인 생성형 AI 이용 여부(Y)"

    if i in (36, 40):
        return "학교급과 학년을 분석하기 위해 유지"

    if i in KEEP:
        return "매체 이용 경험을 나타내는 입력 변수(X)"

    if i == 2:
        return "학교 식별 번호여서 모델 입력에서 제외"

    if i in (19, 20, 21):
        return (
            "응답에 AI 이용 여부가 직접 포함될 수 있어 "
            "정보 누수 방지"
        )

    if i in (45, 46):
        return (
            "설문조사 가중치로 이번 분류 모델에는 "
            "사용하지 않음"
        )

    return (
        "이번 탐구의 매체 이용 여부 중심 "
        "분석 범위에서 제외"
    )


# ==========================================
# 4. 모델 관련 함수
# ==========================================

def score(y, pred, prob):

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


def model_objects():

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
# 5. 성능 개선을 위한 전처리
# ==========================================

def make_features(X):

    # 학년으로 학교급을 알 수 있으므로
    # 개선 모델에서는 학교급을 제외
    improved = X.drop(
        columns=["학교급"]
    ).copy()

    # 9개 디지털 매체 중 이용하는 매체의 개수
    improved["이용 매체 개수"] = (
        X[MEDIA].sum(axis=1).astype(float)
    )

    return improved


def model_pipeline(algorithm, improved):

    if improved:

        # 개선 설정:
        # 학년 인코딩 + 9개 매체 이용 여부
        # + 새로 만든 이용 매체 개수
        prep = ColumnTransformer([
            (
                "category",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                ["학년"]
            ),
            (
                "media",
                "passthrough",
                MEDIA
            ),
            (
                "count",
                StandardScaler(),
                ["이용 매체 개수"]
            )
        ])

    else:

        # 기존 모델과 같은 전처리
        prep = ColumnTransformer([
            (
                "category",
                OneHotEncoder(
                    drop="if_binary",
                    handle_unknown="ignore"
                ),
                MEDIA + ["학교급", "학년"]
            )
        ])

    return Pipeline([
        ("preprocessing", prep),
        ("model", algorithm)
    ])


def rank(scores):
    return (
        scores["F1"],
        scores["ROC-AUC"],
        scores["정확도"]
    )


# ==========================================
# 6. 기준선·기존 모델·개선 모델 학습
# ==========================================

@st.cache_resource(
    show_spinner="기준선, 기존 모델, 개선 모델을 학습하는 중입니다."
)
def train_everything(X, Y):

    X_new = make_features(X)

    ids = X.index.to_numpy()

    # 전체 데이터 20%를 최종 평가용으로 분리
    build, test = train_test_split(
        ids,
        test_size=0.2,
        stratify=Y,
        random_state=42
    )

    # 나머지를 학습 60%, 검증 20%로 분리
    tr, va = train_test_split(
        build,
        test_size=0.25,
        stratify=Y.loc[build],
        random_state=42
    )

    # 기준선
    baseline = DummyClassifier(
        strategy="most_frequent"
    )

    baseline.fit(
        X.loc[tr],
        Y.loc[tr]
    )

    base_pred = baseline.predict(
        X.loc[va]
    )

    base_prob = baseline.predict_proba(
        X.loc[va]
    )[
        :,
        list(baseline.classes_).index(1)
    ]

    base_result = {
        "scores": score(
            Y.loc[va],
            base_pred,
            base_prob
        ),
        "pred": base_pred,
        "majority": int(
            baseline.classes_[
                baseline.class_prior_.argmax()
            ]
        )
    }

    outcomes = {}

    # 같은 분할에서 기존·개선 모델 학습
    for setting, X_used in [
        ("기존", X),
        ("개선", X_new)
    ]:

        for name, alg in model_objects().items():

            pipe = model_pipeline(
                alg,
                setting == "개선"
            )

            pipe.fit(
                X_used.loc[tr],
                Y.loc[tr]
            )

            pred = pipe.predict(
                X_used.loc[va]
            )

            prob = pipe.predict_proba(
                X_used.loc[va]
            )[:, 1]

            outcomes[(setting, name)] = {
                "pipe": pipe,
                "pred": pred,
                "scores": score(
                    Y.loc[va],
                    pred,
                    prob
                )
            }

    # 기존 모델 3개 중 최고
    old_best = max(
        [("기존", n) for n in MODELS],
        key=lambda k: rank(
            outcomes[k]["scores"]
        )
    )

    # 개선 모델 3개 중 최고
    new_best = max(
        [("개선", n) for n in MODELS],
        key=lambda k: rank(
            outcomes[k]["scores"]
        )
    )

    # 기존·개선 총 6개 중 검증 F1 기준 선정
    selected = max(
        outcomes,
        key=lambda k: rank(
            outcomes[k]["scores"]
        )
    )

    setting, name = selected

    X_chosen = (
        X_new
        if setting == "개선"
        else X
    )

    # 선정한 설정으로 학습·검증 데이터를 합쳐 재학습
    final = model_pipeline(
        model_objects()[name],
        setting == "개선"
    )

    final.fit(
        X_chosen.loc[build],
        Y.loc[build]
    )

    pred = final.predict(
        X_chosen.loc[test]
    )

    prob = final.predict_proba(
        X_chosen.loc[test]
    )[:, 1]

    # 최종 평가용 기준선
    final_base = DummyClassifier(
        strategy="most_frequent"
    ).fit(
        X.loc[build],
        Y.loc[build]
    )

    bp = final_base.predict(
        X.loc[test]
    )

    bprob = final_base.predict_proba(
        X.loc[test]
    )[
        :,
        list(final_base.classes_).index(1)
    ]

    return {
        "base": base_result,
        "results": outcomes,
        "old_best": old_best,
        "new_best": new_best,
        "selected": selected,
        "valid_y": Y.loc[va],
        "final": score(
            Y.loc[test],
            pred,
            prob
        ),
        "final_base": score(
            Y.loc[test],
            bp,
            bprob
        ),
        "counts": (
            len(tr),
            len(va),
            len(test)
        )
    }


def score_table(result, which="기존"):

    rows = []

    if which == "기존":
        rows.append({
            "모델": "기준선",
            **result["base"]["scores"]
        })

    for name in MODELS:
        rows.append({
            "모델": name,
            **result["results"][
                (which, name)
            ]["scores"]
        })

    out = pd.DataFrame(rows)

    base = result["base"]["scores"]

    out["기준선 대비 정확도(%p)"] = (
        out["정확도"] - base["정확도"]
    ) * 100

    out["기준선 대비 F1"] = (
        out["F1"] - base["F1"]
    )

    return out


def show_scores(scores):

    cards({
        key: f"{scores[key]:.3f}"
        for key in METRICS
    })


# ==========================================
# 7. 혼동행렬
# ==========================================

def show_confusion(y, pred):

    m = confusion_matrix(
        y,
        pred,
        labels=[0, 1]
    )

    labels = [
        "AI 미이용",
        "AI 이용"
    ]

    df = pd.DataFrame([
        {
            "실제": labels[i],
            "예측": labels[j],
            "인원": int(m[i, j])
        }
        for i in range(2)
        for j in range(2)
    ])

    base = alt.Chart(df).encode(
        x=alt.X(
            "예측:N",
            sort=labels
        ),
        y=alt.Y(
            "실제:N",
            sort=labels
        )
    )

    heat = base.mark_rect().encode(
        color="인원:Q"
    )

    numbers = base.mark_text(
        fontSize=20
    ).encode(
        text="인원:Q"
    )

    chart(
        (heat + numbers).properties(
            height=280
        )
    )

    return m


# ==========================================
# 8. 모델 변수 이름 정리
# ==========================================

def details_of_feature(raw):

    name = raw.split("__", 1)[-1]

    for group in ("학교급", "학년"):
        if name.startswith(group + "_"):
            return (
                group + ": "
                + name.split("_", 1)[1]
            )

    if name.endswith("_1"):
        return name[:-2] + " 이용함"

    return name.replace("_", " ")


# ==========================================
# 9. 의사결정트리 단계별 설명
# ==========================================

def tree_question(raw, threshold):

    name = raw.split("__", 1)[-1]

    if raw.startswith("count__"):
        return (
            f"이용 매체 개수가 "
            f"{int(threshold)}개 이하인가요?",
            "예",
            "아니요"
        )

    if raw.startswith("media__"):
        return (
            f"{name}을 이용하나요?",
            "아니요",
            "예"
        )

    if name.startswith("학교급_"):
        return (
            f"학교급이 "
            f"{name.split('_', 1)[1]}인가요?",
            "아니요",
            "예"
        )

    if name.startswith("학년_"):
        return (
            f"학년이 "
            f"{name.split('_', 1)[1]}인가요?",
            "아니요",
            "예"
        )

    if name.endswith("_1"):
        return (
            f"{name[:-2]}을 이용하나요?",
            "아니요",
            "예"
        )

    if name.endswith("_0"):
        return (
            f"{name[:-2]}을 이용하지 않나요?",
            "아니요",
            "예"
        )

    return (
        f"{details_of_feature(raw)}가 "
        f"{threshold:.2f} 이하인가요?",
        "예",
        "아니요"
    )


def tree_explorer(pipe, setting):

    tree = pipe.named_steps["model"].tree_

    raw_names = pipe.named_steps[
        "preprocessing"
    ].get_feature_names_out()

    key = "tree_steps_" + setting

    if key not in st.session_state:
        st.session_state[key] = [0]

    path = st.session_state[key]

    if path[-1] >= tree.node_count:
        path[:] = [0]

    with st.expander(
        "의사결정트리 탐색 방법",
        expanded=True
    ):
        st.write(
            "질문에서 **예/아니요**를 선택하면 "
            "해당 집단의 다음 조건을 볼 수 있어요. "
            "'이전 단계'로 돌아가거나 "
            "처음부터 다시 볼 수도 있습니다."
        )

    x, y = st.columns(2)

    if x.button(
        "처음부터",
        key="tr_reset"
    ):
        path[:] = [0]
        st.rerun()

    if y.button(
        "이전 단계",
        key="tr_back",
        disabled=len(path) == 1
    ):
        path.pop()
        st.rerun()

    node = path[-1]

    v = tree.value[node][0]

    pct = (
        100 * v[1] / v.sum()
        if v.sum()
        else 0
    )

    cards({
        "현재 단계": f"{len(path)-1}단계",
        "학습 응답자":
            f"{tree.n_node_samples[node]:,}명",
        "AI 이용률": f"{pct:.1f}%"
    })

    left = tree.children_left[node]
    right = tree.children_right[node]

    if left < 0:
        prediction = (
            "AI 이용함"
            if pct >= 50
            else "AI 이용 안함"
        )

        st.success(
            "마지막 분류: " + prediction
        )
        return

    q, left_label, right_label = tree_question(
        raw_names[tree.feature[node]],
        tree.threshold[node]
    )

    st.info(
        "**분류 질문: " + q + "**"
    )

    columns = st.columns(2)

    for col, target, label in [
        (columns[0], left, left_label),
        (columns[1], right, right_label)
    ]:

        value = tree.value[target][0]

        use = (
            100 * value[1] / value.sum()
            if value.sum()
            else 0
        )

        with col:
            with st.container(border=True):

                st.write("**" + label + "**")

                st.write(
                    f"학생 "
                    f"{tree.n_node_samples[target]:,}명"
                    f" · AI 이용률 {use:.1f}%"
                )

                if st.button(
                    label + " → 다음 단계",
                    key="next_" + str(target),
                    use_container_width=True
                ):
                    path.append(int(target))
                    st.rerun()

    st.caption(
        "여기 표시된 비율은 학습 자료의 분포이며, "
        "예측 성능은 별도 검증 자료로 확인합니다."
    )


# ==========================================
# 10. 데이터와 메뉴 불러오기
# ==========================================

st.sidebar.title("🧭 탐구 메뉴")

menu = st.sidebar.radio(
    "이동할 메뉴",
    [
        "① 질문",
        "② 데이터 분석",
        "③ 기준선",
        "④ 모델 평가"
    ]
)

uploaded = st.sidebar.file_uploader(
    "13열 CSV 업로드 (선택)",
    type="csv"
)

if uploaded is not None:
    df = pd.read_csv(
        uploaded,
        encoding="utf-8-sig",
        low_memory=False
    )

elif SELECTED.exists():
    df = read_csv(str(SELECTED))

else:
    st.error(
        "13열 CSV를 찾을 수 없습니다. "
        "data 폴더를 확인하세요."
    )
    st.stop()

if len(df.columns) != 13:
    st.error(
        f"선택한 CSV는 13열이어야 합니다. "
        f"현재 {len(df.columns)}열입니다."
    )
    st.stop()

if (
    "응답자 ID" not in df.columns[0]
    or "생성형 AI" not in df.columns[8]
    or "학교급" not in df.columns[11]
    or "학년별" not in df.columns[12]
):
    st.error(
        "CSV 열의 순서나 이름이 예상과 다릅니다."
    )
    st.stop()

media_cols = (
    list(df.columns[1:8])
    + list(df.columns[9:11])
)

Xmedia = pd.DataFrame({
    name: pd.to_numeric(
        df[col],
        errors="coerce"
    ).map({
        1: 1,
        2: 0
    })
    for name, col in zip(MEDIA, media_cols)
})

Y = pd.to_numeric(
    df.iloc[:, 8],
    errors="coerce"
).map({
    1: 1,
    2: 0
})

sc = pd.to_numeric(
    df.iloc[:, 11],
    errors="coerce"
).map(SCHOOL)

gr = pd.to_numeric(
    df.iloc[:, 12],
    errors="coerce"
).map(GRADE)

if (
    Xmedia.isna().any().any()
    or Y.isna().any()
    or sc.isna().any()
    or gr.isna().any()
):
    st.error(
        "결측값 또는 코드 범위를 벗어난 응답이 있습니다. "
        "CSV와 코드북을 확인하세요."
    )
    st.stop()

X = Xmedia.copy()
X["학교급"] = sc
X["학년"] = gr

st.title("청소년 생성형 AI 이용 분류 탐구")

st.caption(
    "탐구 질문 | 청소년의 디지털 매체 이용 특성으로 "
    "생성형 AI 이용 여부를 분류할 수 있을까?"
)


# ==========================================
# ① 질문
# ==========================================

if menu == "① 질문":

    heading("탐구 질문")

    with st.container(border=True):
        st.markdown(
            "### 청소년의 디지털 매체 이용 특성으로 "
            "생성형 AI 이용 여부를 분류할 수 있을까?"
        )

        st.write(
            "매체 이용 경험 9개, 학교급과 학년을 이용해 "
            "'AI 이용함 / 이용 안함'을 예측합니다."
        )

    cards({
        "응답자": f"{len(df):,}명",
        "입력 변수": "11개",
        "후보 모델": "3개"
    })

    heading("왜 분류를 선택했을까?")

    cols = st.columns(3)

    titles = [
        "회귀",
        "분류 (선택)",
        "군집"
    ]

    descriptions = [
        "연속적인 숫자 예측",
        "이용함·이용 안함 예측",
        "정답 없이 비슷한 집단 묶기"
    ]

    for col, title, content in zip(
        cols,
        titles,
        descriptions
    ):
        with col:
            with st.container(border=True):
                st.write("**" + title + "**")
                st.caption(content)

    st.success(
        "이용 여부가 두 가지 정답으로 나뉘므로 "
        "이진 분류가 적합합니다."
    )

    heading("모델 3개의 작동 방식")

    model_info = pd.DataFrame([
        [
            "로지스틱 회귀",
            "이용 확률 계산",
            "기본 성능과 계수 해석"
        ],
        [
            "의사결정트리",
            "조건에 따라 분류",
            "질문과 분기 과정 확인"
        ],
        [
            "랜덤 포레스트",
            "여러 트리의 예측 종합",
            "복잡한 관계 학습"
        ]
    ], columns=[
        "모델",
        "어떻게 동작하나?",
        "선정 이유"
    ])

    st.dataframe(
        model_info,
        hide_index=True,
        use_container_width=True
    )


# ==========================================
# ② 데이터 분석
# ==========================================

elif menu == "② 데이터 분석":

    tab1, tab2, tab3 = st.tabs([
        "전처리 확인",
        "데이터 분포",
        "AI 이용 여부와의 관계"
    ])

    # 전처리 확인
    with tab1:

        heading("전처리 현황")

        cards({
            "원본 변수": "46개",
            "선택 변수": "13개",
            "제외 변수": "33개"
        })

        cards({
            "응답자": f"{len(df):,}명",
            "선택 CSV 결측치":
                int(df.isna().sum().sum()),
            "중복 ID":
                int(df.iloc[:, 0].duplicated().sum())
        })

        st.write(
            "원본 응답값은 바꾸지 않고, "
            "필요한 13개 열만 선택했습니다. "
            "ID와 정답 Y는 모델 입력에서 제외합니다."
        )

        with st.expander(
            "46개 변수별 선택·제외 이유"
        ):

            if ORIGINAL.exists():

                original = read_csv(
                    str(ORIGINAL)
                )

                report = pd.DataFrame([
                    {
                        "번호": i,
                        "변수": col,
                        "처리": (
                            "유지"
                            if i in KEEP
                            else "제외"
                        ),
                        "이유": reason(i)
                    }
                    for i, col in enumerate(
                        original.columns,
                        1
                    )
                ])

                st.dataframe(
                    report,
                    hide_index=True,
                    use_container_width=True
                )

                a = original.iloc[
                    :,
                    [i - 1 for i in KEEP]
                ].reset_index(drop=True)

                b = df.reset_index(drop=True)

                same = (
                    a.columns.tolist()
                    == b.columns.tolist()
                    and a.astype("string")
                    .fillna("<NA>")
                    .equals(
                        b.astype("string")
                        .fillna("<NA>")
                    )
                )

                if same:
                    st.success(
                        "선택한 13열의 이름·순서·값이 "
                        "원본과 일치합니다."
                    )
                else:
                    st.warning(
                        "원본과 선택 데이터 간 "
                        "차이가 있습니다."
                    )

            else:
                st.warning(
                    "원본 46열 파일이 없어 "
                    "비교를 확인할 수 없습니다."
                )

    # 데이터 분포
    with tab2:

        heading(
            "디지털 매체 이용률",
            "이진 범주형은 막대그래프로 비교합니다."
        )

        rates = (
            Xmedia.mean() * 100
        ).sort_values(
            ascending=False
        ).reset_index()

        rates.columns = [
            "매체",
            "이용률"
        ]

        chart(
            alt.Chart(rates).mark_bar().encode(
                x=alt.X(
                    "이용률:Q",
                    scale=alt.Scale(
                        domain=[0, 100]
                    )
                ),
                y=alt.Y(
                    "매체:N",
                    sort="-x"
                ),
                tooltip=[
                    "매체",
                    "이용률"
                ]
            ).properties(height=340)
        )

        st.write(
            f"가장 이용률이 높은 매체는 "
            f"**{rates.iloc[0]['매체']}**"
            f"({rates.iloc[0]['이용률']:.1f}%)입니다."
        )

        heading("AI 이용 여부")

        cards({
            "AI 이용":
                f"{int(Y.sum()):,}명",
            "AI 미이용":
                f"{int((1-Y).sum()):,}명",
            "AI 이용률":
                f"{Y.mean()*100:.1f}%"
        })

        heading("학교급·학년별 응답자")

        a, b = st.columns(2)

        with a:
            st.bar_chart(
                sc.value_counts().reindex(
                    list(SCHOOL.values()),
                    fill_value=0
                )
            )

        with b:
            st.bar_chart(
                gr.value_counts().reindex(
                    list(GRADE.values()),
                    fill_value=0
                )
            )

        st.caption(
            "학년 코드는 실제 학년명(초4~고3)으로 "
            "표시했습니다."
        )

    # X와 Y의 관계
    with tab3:

        heading("입력 변수(X)와 정답(Y)의 관계")

        rows = pd.DataFrame([
            {
                "매체": m,
                "상관계수": Xmedia[m].corr(Y)
            }
            for m in MEDIA
        ])

        rows["절댓값"] = (
            rows["상관계수"].abs()
        )

        rows = rows.sort_values(
            "절댓값",
            ascending=False
        )

        chart(
            alt.Chart(rows).mark_bar().encode(
                x=alt.X(
                    "상관계수:Q",
                    scale=alt.Scale(
                        domain=[-1, 1]
                    )
                ),
                y=alt.Y(
                    "매체:N",
                    sort=list(rows["매체"])
                ),
                tooltip=[
                    "매체",
                    "상관계수"
                ]
            ).properties(height=340)
        )

        top = rows.iloc[0]

        st.write(
            f"가장 큰 상관계수 절댓값은 "
            f"**{top['매체']}**"
            f"(r={top['상관계수']:.3f})입니다."
        )

        heading(
            "AI 이용자·미이용자의 매체 이용률"
        )

        groups = []

        for m in MEDIA:

            groups.extend([
                {
                    "매체": m,
                    "집단": "AI 이용자",
                    "이용률":
                        100 * Xmedia.loc[
                            Y == 1, m
                        ].mean()
                },
                {
                    "매체": m,
                    "집단": "AI 미이용자",
                    "이용률":
                        100 * Xmedia.loc[
                            Y == 0, m
                        ].mean()
                }
            ])

        chart(
            alt.Chart(
                pd.DataFrame(groups)
            ).mark_bar().encode(
                x=alt.X(
                    "이용률:Q",
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
                    "이용률"
                ]
            ).properties(height=420)
        )

        heading("학교급·학년별 AI 이용률")

        for title, series, order in [
            (
                "학교급",
                sc,
                list(SCHOOL.values())
            ),
            (
                "학년",
                gr,
                list(GRADE.values())
            )
        ]:

            rate = pd.DataFrame({
                title: series,
                "AI 이용": Y
            }).groupby(
                title
            )["AI 이용"].mean().mul(
                100
            ).reindex(
                order
            ).reset_index(
                name="이용률"
            )

            st.write("**" + title + "**")

            chart(
                bars(
                    rate,
                    title,
                    "이용률",
                    order,
                    100
                )
            )

        st.info(
            "상관관계는 인과관계를 증명하지 않으며, "
            "모델 성능은 별도 검증 데이터에서 "
            "확인해야 합니다."
        )


# ==========================================
# ③ 기준선
# ==========================================

elif menu == "③ 기준선":

    trained = train_everything(
        X,
        Y.astype(int)
    )

    base = trained["base"]

    heading("기준선이란?")

    label = (
        "AI 이용"
        if base["majority"] == 1
        else "AI 미이용"
    )

    with st.container(border=True):

        st.markdown(
            "### 모든 학생을 '"
            + label
            + "'으로 예측"
        )

        st.write(
            "학습 데이터에서 가장 많은 범주만 "
            "예측하는 DummyClassifier입니다. "
            "매체 정보는 사용하지 않습니다."
        )

    heading("기준선의 검증 점수")

    show_scores(
        base["scores"]
    )

    heading("실제와 예측 비교")

    cm = show_confusion(
        trained["valid_y"],
        base["pred"]
    )

    if base["majority"] == 1:

        st.warning(
            f"AI 미이용자 {cm[0,1]:,}명도 "
            f"모두 이용자로 잘못 예측했습니다. "
            f"재현율이 1.000이어도 두 범주를 "
            f"잘 구별하는 모델은 아닙니다."
        )

    else:

        st.warning(
            f"실제 AI 이용자 {cm[1,0]:,}명을 "
            f"놓쳤습니다."
        )

    with st.expander("평가 지표 설명"):

        st.write(
            "**정확도:** 전체 예측 중 맞힌 비율"
        )

        st.write(
            "**정밀도:** AI 이용이라고 예측한 사람 중 "
            "실제 이용자인 비율"
        )

        st.write(
            "**재현율:** 실제 AI 이용자를 찾아낸 비율"
        )

        st.write(
            "**F1:** 정밀도와 재현율의 균형"
        )

        st.write(
            "**ROC-AUC:** 두 집단을 구별하는 정도"
        )


# ==========================================
# ④ 모델 평가
# ==========================================

else:

    trained = train_everything(
        X,
        Y.astype(int)
    )

    results = trained["results"]

    old_key = trained["old_best"]
    improved_key = trained["new_best"]
    best_key = trained["selected"]

    st.header("모델 평가")

    with st.container(border=True):

        st.write("**검증 결과 요약**")

        st.markdown(
            f"### 최종 선정 후보: "
            f"{best_key[1]} "
            f"({best_key[0]} 전처리)"
        )

        cards({
            "F1":
                f"{results[best_key]['scores']['F1']:.3f}",

            "기준선 F1":
                f"{trained['base']['scores']['F1']:.3f}",

            "정확도":
                f"{results[best_key]['scores']['정확도']*100:.1f}%"
        })

        st.caption(
            "기준선과 6개 후보를 동일한 검증 데이터에서 "
            "비교하고 F1을 우선 기준으로 선정했습니다."
        )

    t1, t2, t3, t4, t5 = st.tabs([
        "성능 비교",
        "성능 개선",
        "모델별 설명",
        "혼동행렬",
        "최종 평가"
    ])

    # --------------------------------------
    # 탭 1. 기존 모델 성능 비교
    # --------------------------------------

    with t1:

        heading("기존 모델 3개와 기준선 비교")

        original = score_table(
            trained,
            "기존"
        )

        st.dataframe(
            original.round(3),
            hide_index=True,
            use_container_width=True
        )

        a, b = st.columns(2)

        with a:
            st.write("**정확도**")

            chart(
                bars(
                    original,
                    "모델",
                    "정확도",
                    ["기준선"] + MODELS,
                    1
                )
            )

        with b:
            st.write("**F1**")

            chart(
                bars(
                    original,
                    "모델",
                    "F1",
                    ["기준선"] + MODELS,
                    1
                )
            )

        st.info(
            "정확도는 전체 정답을 맞힌 비율이고, "
            "F1은 AI 이용자 분류의 정밀도·재현율 "
            "균형입니다. 이 두 수치를 따로 비교해야 합니다."
        )

    # --------------------------------------
    # 탭 2. 새로 추가한 성능 개선
    # --------------------------------------

    with t2:

        heading(
            "성능 개선 실험",
            "기존 모델과 개선된 전처리 모델을 "
            "같은 검증 데이터에서 비교합니다."
        )

        with st.container(border=True):

            st.write(
                "**데이터 전처리를 이렇게 바꿨습니다.**"
            )

            st.write(
                "① **학교급 제외:** 학년(초4~고3)에서 "
                "학교급을 이미 알 수 있어 중복을 줄였습니다."
            )

            st.write(
                "② **이용 매체 개수 추가:** "
                "9개 매체 중 몇 개를 이용하는지(0~9개)를 "
                "새 입력으로 만들었습니다."
            )

            st.write(
                "③ **원본값·행 유지:** 원본 CSV를 "
                "수정하지 않았고, 정답(Y)이나 ID는 "
                "입력에 사용하지 않았습니다."
            )

            st.caption(
                "이용 매체 개수는 기존 응답에서 계산한 "
                "값이며 새로운 설문조사 정보는 아닙니다. "
                "학습·검증·평가 분할과 모델 설정은 "
                "기존과 동일합니다."
            )

        table = []

        for name in MODELS:

            old = results[
                ("기존", name)
            ]["scores"]

            new = results[
                ("개선", name)
            ]["scores"]

            table.append({
                "모델": name,
                "기존 정확도": old["정확도"],
                "개선 정확도": new["정확도"],
                "정확도 차이(%p)":
                    100 * (
                        new["정확도"]
                        - old["정확도"]
                    ),
                "기존 F1": old["F1"],
                "개선 F1": new["F1"],
                "F1 차이":
                    new["F1"] - old["F1"]
            })

        diff = pd.DataFrame(table)

        heading(
            "기존과 개선 설정의 실제 성능"
        )

        st.dataframe(
            diff.round(3),
            hide_index=True,
            use_container_width=True
        )

        long = pd.DataFrame([
            {
                "모델": name,
                "전처리": setting,
                "F1": results[
                    (setting, name)
                ]["scores"]["F1"]
            }
            for name in MODELS
            for setting in ["기존", "개선"]
        ])

        comparison_chart = alt.Chart(
            long
        ).mark_bar().encode(
            x=alt.X(
                "모델:N",
                sort=MODELS,
                axis=alt.Axis(
                    labelAngle=0
                )
            ),
            xOffset="전처리:N",
            y=alt.Y(
                "F1:Q",
                scale=alt.Scale(
                    domain=[0, 1]
                )
            ),
            color="전처리:N",
            tooltip=[
                "모델",
                "전처리",
                alt.Tooltip(
                    "F1:Q",
                    format=".3f"
                )
            ]
        ).properties(height=330)

        chart(comparison_chart)

        num_better = int(
            (diff["F1 차이"] > 1e-9).sum()
        )

        if num_better:

            st.success(
                f"세 모델 중 {num_better}개에서 "
                f"전처리 변경 후 F1이 높아졌습니다."
            )

        else:

            st.warning(
                "이번 전처리 변경은 세 모델의 F1을 "
                "높이지 못했습니다. "
                "개선을 성공했다고 주장하지 않습니다."
            )

        st.write(
            f"기존 설정의 최고 F1: "
            f"**{old_key[1]} "
            f"{results[old_key]['scores']['F1']:.3f}**"
        )

        st.write(
            f"개선 설정의 최고 F1: "
            f"**{improved_key[1]} "
            f"{results[improved_key]['scores']['F1']:.3f}**"
        )

        delta = (
            results[improved_key]["scores"]["F1"]
            - results[old_key]["scores"]["F1"]
        )

        st.write(
            f"각 설정의 최고 모델끼리 비교한 "
            f"F1 차이: **{delta:+.3f}**"
        )

        if delta <= 0:

            st.info(
                "개선 설정이 기존 최고 모델보다 "
                "좋지 않아, 기존 설정도 "
                "최종 선정 후보로 남겼습니다."
            )

        else:

            st.info(
                "개선 설정의 최고 모델이 검증 F1에서 "
                "앞섰습니다. 최종 평가는 별도 보관한 "
                "데이터로 확인합니다."
            )

        with st.expander(
            "왜 다른 전처리는 하지 않았을까?"
        ):

            st.write(
                "원본 13열에 결측치가 없는 경우 "
                "결측치 대체가 필요하지 않습니다. "
                "또한 두 범주의 비율이 비슷해 "
                "무리한 재표본추출은 진행하지 않았습니다."
            )

            st.write(
                "매체 이용 응답 1·2는 분석 복사본에서 "
                "1·0으로 처리하고 범주형 학년은 "
                "원-핫 인코딩했습니다. "
                "모든 변환은 학습 데이터의 정보만 "
                "사용합니다."
            )

    # --------------------------------------
    # 탭 3. 모델별 설명
    # --------------------------------------

    with t3:

        heading(
            "모델의 작동 원리와 학습 결과"
        )

        setting = st.radio(
            "분석할 전처리",
            ["기존", "개선"],
            horizontal=True
        )

        name = st.selectbox(
            "분류 모델",
            MODELS
        )

        pipe = results[
            (setting, name)
        ]["pipe"]

        alg = pipe.named_steps["model"]

        if name == "로지스틱 회귀":

            st.write(
                "**작동 방식:** 입력 항목에 계수를 "
                "적용해 AI 이용 확률을 계산합니다."
            )

            features = pipe.named_steps[
                "preprocessing"
            ].get_feature_names_out()

            coef = pd.DataFrame({
                "항목": [
                    details_of_feature(n)
                    for n in features
                ],
                "계수": alg.coef_[0]
            })

            coef["크기"] = coef["계수"].abs()

            coef = coef.sort_values(
                "크기",
                ascending=False
            )

            top = coef.head(14)

            chart(
                alt.Chart(top).mark_bar().encode(
                    x="계수:Q",
                    y=alt.Y(
                        "항목:N",
                        sort=list(top["항목"]),
                        axis=alt.Axis(
                            labelLimit=300
                        )
                    ),
                    tooltip=[
                        "항목",
                        alt.Tooltip(
                            "계수:Q",
                            format="+.3f"
                        )
                    ]
                ).properties(height=420)
            )

            with st.expander(
                "모든 계수 보기"
            ):

                st.dataframe(
                    coef[
                        ["항목", "계수"]
                    ].round(4),
                    hide_index=True,
                    use_container_width=True
                )

            st.caption(
                "계수의 양·음은 다른 조건이 같을 때의 "
                "관련 방향이며 인과관계는 아닙니다."
            )

        elif name == "의사결정트리":

            st.write(
                "**작동 방식:** 조건을 따라 데이터를 나누어 "
                "집단별 AI 이용 여부를 예측합니다."
            )

            tree_explorer(
                pipe,
                setting
            )

        else:

            st.write(
                "**작동 방식:** 여러 결정트리의 "
                "예측을 종합합니다."
            )

            features = pipe.named_steps[
                "preprocessing"
            ].get_feature_names_out()

            imp = pd.DataFrame({
                "항목": [
                    details_of_feature(n)
                    for n in features
                ],
                "중요도": alg.feature_importances_
            })

            def group(x):

                if x.startswith("학교급:"):
                    return "학교급"

                if x.startswith("학년:"):
                    return "학년"

                if x.endswith(" 이용함"):
                    return x[:-4]

                return x

            imp["변수"] = imp["항목"].map(group)

            imp = imp.groupby(
                "변수",
                as_index=False
            )["중요도"].sum().sort_values(
                "중요도",
                ascending=False
            )

            chart(
                alt.Chart(imp).mark_bar().encode(
                    x="중요도:Q",
                    y=alt.Y(
                        "변수:N",
                        sort="-x"
                    ),
                    tooltip=[
                        "변수",
                        "중요도"
                    ]
                ).properties(height=350)
            )

            st.caption(
                "트리 분할 중요도는 인과관계가 아닙니다."
            )

        heading("이 모델의 검증 지표")

        show_scores(
            results[
                (setting, name)
            ]["scores"]
        )

    # --------------------------------------
    # 탭 4. 혼동행렬
    # --------------------------------------

    with t4:

        heading(
            "혼동행렬로 실제 결과 확인하기"
        )

        options = ["기준선"] + [
            f"{setting} · {name}"
            for setting in ["기존", "개선"]
            for name in MODELS
        ]

        choice = st.selectbox(
            "평가할 모델",
            options
        )

        if choice == "기준선":

            pred = trained["base"]["pred"]

        else:

            setting, name = choice.split(
                " · ",
                1
            )

            pred = results[
                (setting, name)
            ]["pred"]

        matrix = show_confusion(
            trained["valid_y"],
            pred
        )

        tn, fp, fn, tp = matrix.ravel()

        cards({
            "정확하게 예측": f"{tn+tp:,}명",
            "잘못 예측": f"{fp+fn:,}명"
        })

        st.write(
            f"실제 AI 이용자 중 {tp:,}명을 맞혔고 "
            f"{fn:,}명은 놓쳤습니다. "
            f"AI 미이용자 중 {tn:,}명을 맞혔고 "
            f"{fp:,}명은 이용자로 잘못 예측했습니다."
        )

    # --------------------------------------
    # 탭 5. 최종 평가
    # --------------------------------------

    with t5:

        heading(
            "최종 모델 선정",
            "최종 평가 데이터는 모델 선정에 "
            "사용하지 않았습니다."
        )

        selected_score = results[
            best_key
        ]["scores"]

        base = trained["base"]["scores"]

        st.success(
            f"검증 F1 기준 최종 선정: "
            f"**{best_key[1]} / "
            f"{best_key[0]} 전처리**"
        )

        cards({
            "선정 F1":
                f"{selected_score['F1']:.3f}",
            "기준선 F1":
                f"{base['F1']:.3f}",
            "검증 정확도 개선":
                f"{(selected_score['정확도'] - base['정확도'])*100:+.2f}%p"
        })

        st.write(
            f"검증 F1 차이는 기준선 대비 "
            f"**{selected_score['F1'] - base['F1']:+.3f}**입니다. "
            f"모델의 특징뿐 아니라 F1 점수를 "
            f"우선 기준으로 선정했습니다."
        )

        heading(
            "보관한 평가 데이터에서 재확인"
        )

        final = trained["final"]
        final_base = trained["final_base"]

        final_table = pd.DataFrame([
            {
                "모델": "기준선",
                **final_base
            },
            {
                "모델":
                    f"{best_key[1]} ({best_key[0]})",
                **final
            }
        ])

        st.dataframe(
            final_table.round(3),
            hide_index=True,
            use_container_width=True
        )

        acc_diff = 100 * (
            final["정확도"]
            - final_base["정확도"]
        )

        f1_diff = (
            final["F1"]
            - final_base["F1"]
        )

        cards({
            "최종 정확도 차이":
                f"{acc_diff:+.2f}%p",
            "최종 F1 차이":
                f"{f1_diff:+.3f}"
        })

        if acc_diff > 0 and f1_diff > 0:

            st.success(
                "최종 평가에서도 기준선보다 "
                "정확도와 F1이 모두 높았습니다. "
                "이번 데이터에서 AI 이용 여부를 "
                "더 잘 분류한 결과입니다."
            )

        elif acc_diff > 0 or f1_diff > 0:

            st.warning(
                "일부 지표만 기준선보다 높았습니다. "
                "모든 면에서 성능이 개선되었다고 "
                "말할 수는 없습니다."
            )

        else:

            st.warning(
                "최종 평가에서 기준선보다 "
                "정확도와 F1이 모두 높지는 않았습니다. "
                "데이터의 예측 가능성에 한계가 있습니다."
            )

        st.caption(
            "결과는 이 설문 데이터와 현재 분할에 "
            "대한 것이며, 다른 학생 집단에도 "
            "동일하게 적용되거나 인과관계가 "
            "증명되는 것은 아닙니다. "
            "조사 가중치는 적용하지 않았습니다."
        )
