
from pathlib import Path
import math

import altair as alt
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


# ====================================================
# 1. 기본 설정
# ====================================================

st.set_page_config(
    page_title="청소년 AI 이용 분류 탐구",
    page_icon="📊",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

SELECTED = DATA / "청소년_AI이용분류_ID포함13열.csv"
ORIGINAL = DATA / "청소년_AI이용분류_정제데이터.csv"

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
    "웹툰",
]

SCHOOL = {
    1: "초등학교",
    2: "중학교",
    3: "고등학교",
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
    9: "고3",
}

MODELS = [
    "로지스틱 회귀",
    "의사결정트리",
    "랜덤 포레스트",
]

METRICS = [
    "정확도",
    "정밀도",
    "재현율",
    "F1",
    "ROC-AUC",
]

SOURCE_URL = (
    "https://www.mogef.go.kr/mp/pcd/"
    "mp_pcd_s001d.do?mid=plc502&bbtSn=704853"
)


# ====================================================
# 2. 웹 디자인
# ====================================================

st.markdown(
    """
<style>
.block-container {
    max-width: 1130px;
    padding-top: 1.8rem;
    padding-bottom: 3.5rem;
}

h1, h2, h3 {
    letter-spacing: -.03em;
}

[data-testid="stMetricValue"] {
    font-size: 1.55rem;
}

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ====================================================
# 3. 공통 화면 함수
# ====================================================

def section(title, desc=None):
    st.subheader(title)

    if desc:
        st.caption(desc)


def metrics_view(items):
    cols = st.columns(len(items))

    for col, (name, value) in zip(cols, items.items()):
        with col:
            st.metric(name, value)


def draw(chart):
    st.altair_chart(
        chart,
        use_container_width=True,
    )


@st.cache_data
def load_file(path):
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
    )


# ====================================================
# 4. 모델 평가 함수
# ====================================================

def scores(y, pred, proba):
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
        "ROC-AUC": roc_auc_score(y, proba),
    }


def algorithms():
    return {
        "로지스틱 회귀": LogisticRegression(
            max_iter=2000,
            random_state=42,
        ),
        "의사결정트리": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=30,
            random_state=42,
        ),
        "랜덤 포레스트": RandomForestClassifier(
            n_estimators=120,
            max_depth=10,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        ),
    }


def pipeline(name, improved=False):

    if improved:

        prep = ColumnTransformer([
            (
                "학년",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                ["학년"],
            ),
            (
                "매체",
                "passthrough",
                MEDIA,
            ),
            (
                "개수",
                StandardScaler(),
                ["이용 매체 개수"],
            ),
        ])

    else:

        prep = ColumnTransformer([
            (
                "범주",
                OneHotEncoder(
                    drop="if_binary",
                    handle_unknown="ignore",
                ),
                MEDIA + ["학교급", "학년"],
            ),
        ])

    return Pipeline([
        ("전처리", prep),
        ("모델", algorithms()[name]),
    ])


# ====================================================
# 5. 성능 개선용 데이터 전처리
# ====================================================

def improve_data(X):

    # 학교급은 학년 정보와 중복되므로 제외한다.
    result = X.drop(
        columns=["학교급"]
    ).copy()

    # 기존 매체 이용 여부 9개로 새 변수를 만든다.
    # 생성형 AI 이용 여부(Y)는 사용하지 않는다.
    result["이용 매체 개수"] = (
        X[MEDIA]
        .sum(axis=1)
        .astype(float)
    )

    return result


def ranking(item):
    s = item["scores"]

    return (
        s["F1"],
        s["ROC-AUC"],
        s["정확도"],
    )


# ====================================================
# 6. 기준선 및 분류 모델 학습
# ====================================================

@st.cache_resource(
    show_spinner=(
        "동일한 데이터 분할로 기준선과 "
        "6개 분류 모델을 학습합니다."
    )
)
def train_all(X, y):

    # 학습+검증 80%, 최종 평가 20%
    build_idx, test_idx = train_test_split(
        X.index,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # 학습 60%, 검증 20%, 최종 평가 20%
    train_idx, valid_idx = train_test_split(
        build_idx,
        test_size=0.25,
        stratify=y.loc[build_idx],
        random_state=42,
    )

    improved = improve_data(X)

    # ------------------------------------
    # 기준선
    # ------------------------------------

    dummy = DummyClassifier(
        strategy="most_frequent"
    )

    dummy.fit(
        X.loc[train_idx],
        y.loc[train_idx],
    )

    dummy_pred = dummy.predict(
        X.loc[valid_idx]
    )

    positive_index = list(
        dummy.classes_
    ).index(1)

    dummy_prob = dummy.predict_proba(
        X.loc[valid_idx]
    )[:, positive_index]

    baseline = {
        "pred": dummy_pred,
        "scores": scores(
            y.loc[valid_idx],
            dummy_pred,
            dummy_prob,
        ),
        "label": int(
            dummy.classes_[
                dummy.class_prior_.argmax()
            ]
        ),
    }

    # ------------------------------------
    # 기존 모델 및 개선 모델
    # ------------------------------------

    results = {}

    for setting, frame in [
        ("기존", X),
        ("개선", improved),
    ]:

        for name in MODELS:

            model = pipeline(
                name,
                improved=(setting == "개선"),
            )

            model.fit(
                frame.loc[train_idx],
                y.loc[train_idx],
            )

            pred = model.predict(
                frame.loc[valid_idx]
            )

            prob = model.predict_proba(
                frame.loc[valid_idx]
            )[:, 1]

            results[(setting, name)] = {
                "pipeline": model,
                "pred": pred,
                "scores": scores(
                    y.loc[valid_idx],
                    pred,
                    prob,
                ),
            }

    # ------------------------------------
    # 검증 F1 기준 최종 모델 선정
    # ------------------------------------

    selected = max(
        results,
        key=lambda key: ranking(results[key]),
    )

    chosen_frame = (
        improved
        if selected[0] == "개선"
        else X
    )

    # 학습+검증 데이터로 다시 학습
    chosen_model = pipeline(
        selected[1],
        improved=(selected[0] == "개선"),
    )

    chosen_model.fit(
        chosen_frame.loc[build_idx],
        y.loc[build_idx],
    )

    # 최종 평가 데이터 사용
    test_pred = chosen_model.predict(
        chosen_frame.loc[test_idx]
    )

    test_prob = chosen_model.predict_proba(
        chosen_frame.loc[test_idx]
    )[:, 1]

    # ------------------------------------
    # 최종 평가용 기준선
    # ------------------------------------

    final_dummy = DummyClassifier(
        strategy="most_frequent"
    )

    final_dummy.fit(
        X.loc[build_idx],
        y.loc[build_idx],
    )

    base_test_pred = final_dummy.predict(
        X.loc[test_idx]
    )

    positive_index = list(
        final_dummy.classes_
    ).index(1)

    base_test_prob = final_dummy.predict_proba(
        X.loc[test_idx]
    )[:, positive_index]

    return {
        "baseline": baseline,
        "results": results,
        "selected": selected,
        "valid_y": y.loc[valid_idx],
        "final_scores": scores(
            y.loc[test_idx],
            test_pred,
            test_prob,
        ),
        "final_baseline": scores(
            y.loc[test_idx],
            base_test_pred,
            base_test_prob,
        ),
        "sizes": (
            len(train_idx),
            len(valid_idx),
            len(test_idx),
        ),
    }


def score_table(
    trained,
    mode="기존",
    baseline=True,
):

    rows = []

    if baseline:
        rows.append({
            "모델": "기준선",
            **trained["baseline"]["scores"],
        })

    for name in MODELS:
        rows.append({
            "모델": name,
            **trained["results"][
                (mode, name)
            ]["scores"],
        })

    table = pd.DataFrame(rows)

    base = trained["baseline"]["scores"]

    table["기준선 대비 정확도(%p)"] = (
        table["정확도"] - base["정확도"]
    ) * 100

    table["기준선 대비 F1"] = (
        table["F1"] - base["F1"]
    )

    return table


def show_five(s):
    metrics_view({
        n: f"{s[n]:.3f}"
        for n in METRICS
    })


# ====================================================
# 7. 혼동행렬
# ====================================================

def confusion_display(y, pred):

    matrix = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    )

    labels = [
        "AI 미이용",
        "AI 이용",
    ]

    cells = pd.DataFrame([
        {
            "실제": labels[i],
            "예측": labels[j],
            "학생 수": int(matrix[i, j]),
        }
        for i in range(2)
        for j in range(2)
    ])

    base = alt.Chart(cells).encode(
        x=alt.X(
            "예측:N",
            sort=labels,
        ),
        y=alt.Y(
            "실제:N",
            sort=labels,
        ),
    )

    heat = base.mark_rect().encode(
        color="학생 수:Q",
        tooltip=[
            "실제",
            "예측",
            "학생 수",
        ],
    )

    txt = base.mark_text(
        fontSize=18
    ).encode(
        text="학생 수:Q",
        color=alt.condition(
            alt.datum["학생 수"] > matrix.max() / 2,
            alt.value("white"),
            alt.value("#111827"),
        ),
    )

    draw(
        (heat + txt).properties(height=270)
    )

    tn, fp, fn, tp = matrix.ravel()

    st.caption(
        f"실제 AI 이용자: 정답 {tp:,}명 / "
        f"놓침 {fn:,}명 · "
        f"실제 미이용자: 정답 {tn:,}명 / "
        f"잘못 이용자로 예측 {fp:,}명"
    )

    return matrix


# ====================================================
# 8. 모델 변수 이름
# ====================================================

def pretty_feature(name):

    if "__" in name:
        name = name.split("__", 1)[1]

    if (
        name.startswith("학교급_")
        or name.startswith("학년_")
    ):
        left, right = name.split("_", 1)
        return f"{left}: {right}"

    if name.endswith("_1"):
        return name[:-2] + " 이용함"

    if name.endswith("_0"):
        return name[:-2] + " 이용 안함"

    return name


def original_feature(name):

    label = pretty_feature(name)

    if label.startswith("학교급:"):
        return "학교급"

    if label.startswith("학년:"):
        return "학년"

    for medium in MEDIA:
        if label.startswith(medium):
            return medium

    return label


# ====================================================
# 9. 의사결정트리 질문 만들기
# ====================================================

def branch_question(
    pipe,
    index,
    threshold,
):

    prep = pipe.named_steps["전처리"]

    name = prep.get_feature_names_out()[index]

    short = name.split("__", 1)[-1]

    # 표준화한 이용 매체 개수의 기준을
    # 실제 개수 단위로 다시 변환한다.
    if name.startswith("개수__"):

        scaler = prep.named_transformers_["개수"]

        original_threshold = (
            threshold * scaler.scale_[0]
            + scaler.mean_[0]
        )

        return (
            f"이용 매체가 "
            f"{math.floor(original_threshold + 1e-7)}"
            f"개 이하인가요?",
            "예",
            "아니요",
        )

    if name.startswith("매체__"):
        return (
            f"{short}을 이용하나요?",
            "아니요",
            "예",
        )

    if short.endswith("_1"):
        return (
            f"{short[:-2]}을 이용하나요?",
            "아니요",
            "예",
        )

    if short.endswith("_0"):
        return (
            f"{short[:-2]}을 이용하지 않나요?",
            "아니요",
            "예",
        )

    if (
        short.startswith("학년_")
        or short.startswith("학교급_")
    ):
        label = pretty_feature(name)

        return (
            f"{label}에 해당하나요?",
            "아니요",
            "예",
        )

    return (
        f"{pretty_feature(name)}이(가) "
        f"{threshold:.2f} 이하인가요?",
        "예",
        "아니요",
    )


def explore_tree(pipe, setting):

    tree = pipe.named_steps["모델"].tree_

    state_key = "tree_path_" + setting

    if state_key not in st.session_state:
        st.session_state[state_key] = [0]

    path = st.session_state[state_key]

    if path[-1] >= tree.node_count:
        path[:] = [0]

    with st.expander(
        "의사결정트리 사용 방법",
        expanded=True,
    ):
        st.write(
            "질문에 맞춰 '예' 또는 '아니요'를 "
            "누르면 해당 집단의 다음 분류를 "
            "볼 수 있습니다. 왼쪽은 모델의 "
            "분기 기준 이하, 오른쪽은 초과에 "
            "해당합니다."
        )

    a, b = st.columns(2)

    if a.button(
        "처음부터 보기",
        key="reset_" + setting,
    ):
        path[:] = [0]
        st.rerun()

    if b.button(
        "이전 단계",
        disabled=len(path) == 1,
        key="back_" + setting,
    ):
        path.pop()
        st.rerun()

    node = path[-1]

    counts = tree.value[node][0]

    rate = (
        100 * counts[1] / counts.sum()
        if counts.sum()
        else 0
    )

    metrics_view({
        "현재 단계": str(len(path) - 1),
        "학습 응답자": (
            f"{tree.n_node_samples[node]:,}명"
        ),
        "AI 이용률": f"{rate:.1f}%",
    })

    left = tree.children_left[node]
    right = tree.children_right[node]

    if left == -1:

        result = (
            "AI 이용"
            if rate >= 50
            else "AI 미이용"
        )

        st.success("최종 예측: " + result)
        return

    question, left_label, right_label = branch_question(
        pipe,
        tree.feature[node],
        tree.threshold[node],
    )

    st.info("**현재 질문: " + question + "**")

    columns = st.columns(2)

    branches = [
        (columns[0], left, left_label),
        (columns[1], right, right_label),
    ]

    for col, next_node, text in branches:

        n = tree.value[next_node][0]

        percent = (
            100 * n[1] / n.sum()
            if n.sum()
            else 0
        )

        with col:
            with st.container(border=True):

                st.write("**" + text + "**")

                st.write(
                    f"{tree.n_node_samples[next_node]:,}명 "
                    f"· AI 이용률 {percent:.1f}%"
                )

                if st.button(
                    text + " → 다음 단계",
                    key=(
                        f"{setting}_"
                        f"{node}_{next_node}"
                    ),
                    use_container_width=True,
                ):
                    path.append(int(next_node))
                    st.rerun()

    st.caption(
        "이 비율은 학습 데이터의 집단별 특성이며 "
        "검증 데이터의 정확도가 아닙니다."
    )


# ====================================================
# 10. 열 선택·제외 근거
# ====================================================

def selection_reason(i):

    if i == 1:
        return (
            "응답자 ID는 중복 확인만 하고 "
            "학습에서 제외"
        )

    if i == 12:
        return "생성형 AI 이용 여부는 예측할 정답 Y"

    if i in (36, 40):
        return "학교급과 학년을 예측 입력으로 사용"

    if i in KEEP:
        return "매체 이용 경험을 예측 입력으로 사용"

    if i == 2:
        return "학교 식별 번호이므로 이번 입력에서 제외"

    if i in (8, 10, 15, 16):
        return (
            "디지털 매체이지만 연구 범위를 "
            "9개 매체로 제한해 제외"
        )

    if i in (17, 18):
        return (
            "종이 매체로 이번 디지털 매체 중심 "
            "연구에서 제외"
        )

    if i in (19, 20, 21):
        return (
            "AI 이용 정보가 포함될 수 있어 "
            "정답 누출 가능성을 고려해 제외"
        )

    if 22 <= i <= 29:
        return (
            "온라인 안전·개인정보 경험은 "
            "이번 예측 입력 범위에서 제외"
        )

    if 30 <= i <= 33:
        return (
            "교육 관련 경험으로 탐구 범위를 "
            "제한해 제외"
        )

    if i in (34, 35, 41, 42):
        return (
            "인구·지역 배경 변수로 "
            "이번 연구 범위에서 제외"
        )

    if i in (37, 38, 39, 43, 44):
        return (
            "학교급·학년 관련 추가 정보로 "
            "기본 변수만 선택"
        )

    if i in (45, 46):
        return (
            "설문 가중치로 이번 분류 모델 "
            "입력에 사용하지 않음"
        )

    return "이번 연구의 입력 변수 범위에서 제외"


# ====================================================
# 11. 데이터 품질 검사
# ====================================================

def error_report(df, media_columns):

    diagnostics = []

    binary = list(media_columns) + [
        df.columns[8]
    ]

    # 매체 이용 응답: 1 또는 2
    for name in binary:

        vals = pd.to_numeric(
            df[name],
            errors="coerce",
        )

        bad = (
            df[name].notna()
            & ~vals.isin([1, 2])
        )

        diagnostics.append({
            "변수": name,
            "유효하지 않은 코드 수": int(bad.sum()),
        })

    # 학교급: 1~3
    # 학년: 1~9
    checks = [
        (df.columns[11], [1, 2, 3]),
        (df.columns[12], list(range(1, 10))),
    ]

    for name, allowed in checks:

        vals = pd.to_numeric(
            df[name],
            errors="coerce",
        )

        bad = (
            df[name].notna()
            & ~vals.isin(allowed)
        )

        diagnostics.append({
            "변수": name,
            "유효하지 않은 코드 수": int(bad.sum()),
        })

    school_num = pd.to_numeric(
        df.iloc[:, 11],
        errors="coerce",
    )

    grade_num = pd.to_numeric(
        df.iloc[:, 12],
        errors="coerce",
    )

    inferred_school = (
        (grade_num - 1) // 3
    ) + 1

    mismatch = (
        school_num.notna()
        & grade_num.notna()
        & (school_num != inferred_school)
    )

    return (
        pd.DataFrame(diagnostics),
        int(mismatch.sum()),
    )


# ====================================================
# 12. 메뉴 및 CSV 불러오기
# ====================================================

st.sidebar.title("🧭 탐구 메뉴")

menu = st.sidebar.radio(
    "이동할 메뉴",
    [
        "① 질문",
        "② 데이터 분석",
        "③ 기준선",
        "④ 모델 평가",
    ],
)

st.sidebar.caption(
    "① 문제 정의·느낀 점  "
    "② 기초통계·전처리·시각화  "
    "③ 기준선  "
    "④ 모델 학습·평가·성능 개선"
)

upload = st.sidebar.file_uploader(
    "13열 CSV 선택 업로드",
    type="csv",
)

if upload is not None:

    df = pd.read_csv(
        upload,
        encoding="utf-8-sig",
        low_memory=False,
    )

elif SELECTED.exists():

    df = load_file(str(SELECTED))

else:

    st.error(
        "13열 CSV가 없습니다. "
        "data 폴더를 확인하세요."
    )

    st.stop()


# 열 구조 검사
valid_columns = (
    df.shape[1] == 13
    and "응답자 ID" in str(df.columns[0])
    and "생성형 AI" in str(df.columns[8])
    and "학교급" in str(df.columns[11])
    and "학년별" in str(df.columns[12])
) if df.shape[1] == 13 else False

if not valid_columns:

    st.error(
        "13열 CSV의 열 이름이나 순서가 "
        "예상과 다릅니다."
    )

    st.stop()


media_cols = (
    list(df.columns[1:8])
    + list(df.columns[9:11])
)

code_errors, grade_mismatch = error_report(
    df,
    media_cols,
)

missing = int(df.isna().sum().sum())

duplicate_id = int(
    df.iloc[:, 0].duplicated().sum()
)

invalid = int(
    code_errors[
        "유효하지 않은 코드 수"
    ].sum()
)


# 이상값이 있으면 임의 삭제하지 않음
if missing or invalid or grade_mismatch:

    st.error(
        "결측값·허용 범위 밖 코드·"
        "학교급과 학년 불일치가 발견되었습니다. "
        "데이터를 자동 삭제하지 않고 "
        "분석을 중단합니다."
    )

    metrics_view({
        "결측값": str(missing),
        "유효하지 않은 코드": str(invalid),
        "학교급·학년 불일치": str(grade_mismatch),
    })

    st.dataframe(
        code_errors,
        hide_index=True,
        use_container_width=True,
    )

    st.stop()


# ====================================================
# 13. 분석용 데이터 생성
# ====================================================

# 1=이용함, 2=이용 안함
Xmedia = pd.DataFrame({
    name: pd.to_numeric(
        df[col]
    ).map({
        1: 1,
        2: 0,
    })
    for name, col in zip(MEDIA, media_cols)
})

# 정답 Y
y = pd.to_numeric(
    df.iloc[:, 8]
).map({
    1: 1,
    2: 0,
}).astype(int)

school = pd.to_numeric(
    df.iloc[:, 11]
).map(SCHOOL)

grade = pd.to_numeric(
    df.iloc[:, 12]
).map(GRADE)

# ID 및 정답 Y는 X에 포함하지 않음
X = Xmedia.copy()

X["학교급"] = school
X["학년"] = grade


st.title("청소년 생성형 AI 이용 분류 탐구")

st.caption(
    "연구 질문 | 청소년의 디지털 매체 이용 특성으로 "
    "생성형 AI 이용 여부를 분류할 수 있을까?"
)


# ====================================================
# ① 질문
# ====================================================

if menu == "① 질문":

    tab_question, tab_reflection = st.tabs([
        "문제 정의",
        "활동 과정·느낀 점",
    ])

    # ----------------------------------------------
    # 문제 정의
    # ----------------------------------------------

    with tab_question:

        with st.container(border=True):

            section("연구 질문")

            st.markdown(
                "### 청소년의 디지털 매체 이용 특성으로 "
                "생성형 AI 이용 여부를 분류할 수 있을까?"
            )

            st.write(
                "9개 매체의 이용 경험과 학교급·학년으로 "
                "AI 이용 여부를 예측하는 탐구입니다."
            )

        metrics_view({
            "전체 응답자": f"{len(df):,}명",
            "입력 변수(X)": "11개",
            "정답(Y)": "AI 이용 / 미이용",
        })

        section("데이터 출처")

        st.write(
            "**2024년 청소년 매체이용 및 "
            "유해환경 실태조사**의 응답 자료에서 "
            "탐구에 필요한 13개 열을 선택했습니다. "
            "자료 제공 기관은 여성가족부입니다."
        )

        st.markdown(
            f"[공식 조사 자료 확인]({SOURCE_URL})"
        )

        st.caption(
            "현재 앱은 CSV에 들어 있는 응답자만 "
            "분석하며, 설문 가중치를 반영하지 않습니다."
        )

        section("이 주제를 선택한 이유")

        st.write(
            "처음에는 생성형 AI를 정보교육에 "
            "어떻게 활용할지 관심이 있었지만, "
            "보유한 데이터를 직접 분석할 수 있는 "
            "질문으로 좁혀 '매체 이용 특성으로 "
            "AI 이용 여부를 예측할 수 있을까?'를 "
            "선택했습니다."
        )

        section(
            "회귀·분류·군집 중 왜 분류일까?"
        )

        methods = pd.DataFrame([
            [
                "회귀",
                "연속적인 수치 예측",
                "이번에는 이용 여부라는 범주를 예측",
            ],
            [
                "분류",
                "정해진 범주 예측",
                "이용함·미이용이라는 두 범주가 정답",
            ],
            [
                "군집",
                "정답 없이 유사한 집단 묶기",
                "이번 데이터에는 정답(Y)이 있음",
            ],
        ], columns=[
            "방법",
            "목적",
            "우리 문제와의 관계",
        ])

        st.dataframe(
            methods,
            hide_index=True,
            use_container_width=True,
        )

        st.success(
            "정답이 두 범주이므로 "
            "**이진 분류**가 적합합니다. "
            "AI 이용의 원인을 증명하는 "
            "연구는 아닙니다."
        )

        section("데이터 속성과 역할")

        types = pd.DataFrame([
            [
                "응답자 ID",
                "식별자",
                "중복 확인에만 사용",
            ],
            [
                "매체 이용 경험 9개",
                "이진 범주형",
                "입력 X",
            ],
            [
                "학교급·학년",
                "순서가 있는 범주형",
                "입력 X",
            ],
            [
                "생성형 AI 이용 여부",
                "이진 범주형",
                "정답 Y",
            ],
        ], columns=[
            "변수",
            "데이터 종류",
            "용도",
        ])

        st.dataframe(
            types,
            hide_index=True,
            use_container_width=True,
        )

    # ----------------------------------------------
    # 활동 과정 및 느낀 점
    # ----------------------------------------------

    with tab_reflection:

        section(
            "AI를 활용하면서 직접 확인하고 수정한 내용"
        )

        st.write(
            "**그래프의 의미:** 처음에는 매체끼리의 "
            "상관관계가 표시되어 탐구 질문과 "
            "맞지 않는다고 지적했고, "
            "X와 Y의 관계를 중심으로 바꿨습니다."
        )

        st.write(
            "**쉬운 표현:** 학년 코드 1~9를 "
            "초4~고3으로 바꾸고, 잘린 변수명과 "
            "복잡한 의사결정트리 그림도 "
            "이해하기 쉽게 바꾸도록 요청했습니다."
        )

        st.write(
            "**평가 방법:** 모델 점수만 보는 대신 "
            "기준선과 비교해 실제 개선 정도를 "
            "확인하도록 요구했습니다."
        )

        section("느낀 점과 배운 점")

        st.write(
            "그래프가 많더라도 무엇을 말하는지 "
            "이해할 수 없으면 분석 결과를 "
            "전달하기 어렵다는 점을 확인했습니다. "
            "AI가 작성한 코드도 그대로 사용하지 않고 "
            "화면의 문제를 직접 찾아 수정해야 했습니다."
        )

        st.info(
            "이 내용은 대화에서 확인된 작업을 바탕으로 "
            "정리한 초안입니다. 개인적인 감상은 "
            "본인의 실제 생각과 일치하는지 "
            "확인 후 제출하세요."
        )

        section(
            "이후 더 탐구할 수 있는 것 — 제안"
        )

        st.write(
            "학교급별로 모델을 따로 평가하면 "
            "차이가 있을지, 새로운 연도의 자료에서도 "
            "비슷한 성능이 나올지 추가로 "
            "확인할 수 있습니다. "
            "이는 아직 진행한 연구가 아닌 "
            "**추가 탐구 아이디어**입니다."
        )


# ====================================================
# ② 데이터 분석
# ====================================================

elif menu == "② 데이터 분석":

    tab_stats, tab_prep, tab_visual, tab_relation = st.tabs([
        "기초통계",
        "데이터 전처리",
        "데이터 시각화",
        "X–Y 관계 분석",
    ])

    # ----------------------------------------------
    # 기초통계
    # ----------------------------------------------

    with tab_stats:

        section(
            "기초통계 — 데이터의 전체 모습"
        )

        metrics_view({
            "응답자(행)": f"{len(df):,}",
            "선택 변수(열)": str(df.shape[1]),
            "결측 셀": str(missing),
            "중복 ID": str(duplicate_id),
        })

        yes = int(y.sum())
        no = len(y) - yes

        metrics_view({
            "AI 이용자": f"{yes:,}명",
            "AI 미이용자": f"{no:,}명",
            "AI 이용률": f"{y.mean()*100:.1f}%",
        })

        section("매체별 이용자 수와 비율")

        freq = pd.DataFrame([
            {
                "매체": name,
                "이용자 수": int(Xmedia[name].sum()),
                "미이용자 수": int(
                    (1 - Xmedia[name]).sum()
                ),
                "이용률(%)": round(
                    Xmedia[name].mean() * 100,
                    2,
                ),
            }
            for name in MEDIA
        ]).sort_values(
            "이용률(%)",
            ascending=False,
        )

        st.dataframe(
            freq,
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "설문 응답 1=이용함, "
            "2=이용 안함입니다. "
            "범주형 자료이므로 응답 코드의 평균보다는 "
            "빈도와 비율이 유용합니다."
        )

        a, b = st.columns(2)

        with a:

            st.write("**학교급별 응답자 수**")

            school_counts = (
                school.value_counts()
                .reindex(
                    list(SCHOOL.values()),
                    fill_value=0,
                )
                .rename_axis("학교급")
                .reset_index(name="응답자 수")
            )

            st.dataframe(
                school_counts,
                hide_index=True,
                use_container_width=True,
            )

        with b:

            st.write("**학년별 응답자 수**")

            grade_counts = (
                grade.value_counts()
                .reindex(
                    list(GRADE.values()),
                    fill_value=0,
                )
                .rename_axis("학년")
                .reset_index(name="응답자 수")
            )

            st.dataframe(
                grade_counts,
                hide_index=True,
                use_container_width=True,
            )

    # ----------------------------------------------
    # 데이터 전처리
    # ----------------------------------------------

    with tab_prep:

        section(
            "전처리 결과 및 근거"
        )

        metrics_view({
            "선택 열": "13개",
            "모델 입력 X": "11개",
            "정답 Y": "1개",
        })

        st.write(
            "응답자 ID는 중복 확인에만 사용하고, "
            "정답인 생성형 AI 이용 여부는 "
            "X에 넣지 않았습니다. "
            "행을 임의로 삭제하지 않았습니다."
        )

        section(
            "결측치·이상 코드·일관성 검사"
        )

        metrics_view({
            "결측 셀": str(missing),
            "허용 범위 밖 코드": str(invalid),
            "학교급·학년 불일치": str(grade_mismatch),
            "중복 ID": str(duplicate_id),
        })

        st.dataframe(
            code_errors.rename(
                columns={
                    "유효하지 않은 코드 수":
                        "이상 코드 수"
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "범주형 변수의 이상치는 큰 숫자가 아니라 "
            "허용된 코드 밖의 값으로 검사합니다. "
            "응답자 ID는 숫자가 크더라도 "
            "이상치로 삭제하지 않습니다."
        )

        section(
            "원본 46개 열과 선택한 13개 열의 비교"
        )

        if ORIGINAL.exists():

            original = load_file(
                str(ORIGINAL)
            )

            report = pd.DataFrame([
                {
                    "원본 열 번호": i,
                    "변수명": col,
                    "선택": (
                        "유지"
                        if i in KEEP
                        else "제외"
                    ),
                    "사유": selection_reason(i),
                }
                for i, col in enumerate(
                    original.columns,
                    1,
                )
            ])

            with st.expander(
                "46개 열 선택·제외 결과 보기"
            ):

                st.dataframe(
                    report,
                    hide_index=True,
                    use_container_width=True,
                )

            subset = original.iloc[
                :,
                [i - 1 for i in KEEP]
            ].reset_index(drop=True)

            actual = df.reset_index(drop=True)

            same = (
                subset.columns.tolist()
                == actual.columns.tolist()
                and subset.astype("string")
                .fillna("<NA>")
                .equals(
                    actual.astype("string")
                    .fillna("<NA>")
                )
            )

            if same:

                st.success(
                    "선택 데이터의 열 이름·순서·"
                    "응답값이 원본과 일치합니다."
                )

            else:

                st.warning(
                    "원본과 선택 CSV가 "
                    "완전히 일치하지 않습니다. "
                    "재확인이 필요합니다."
                )

        else:

            st.warning(
                "원본 46열 파일을 찾지 못해 "
                "직접 대조하지 못했습니다."
            )

        section(
            "분류 모델에 넣기 위한 변환"
        )

        st.write(
            "매체 이용 응답은 분석용 복사본에서 "
            "**이용함=1, 미이용=0**으로 바꾸고 "
            "학교급·학년은 **원-핫 인코딩**합니다. "
            "모델 평가 시 인코더는 "
            "학습 자료로만 학습됩니다."
        )

    # ----------------------------------------------
    # 데이터 시각화
    # ----------------------------------------------

    with tab_visual:

        section(
            "매체 이용률 비교",
            "이진 범주형 변수이므로 "
            "가로 막대그래프를 사용합니다."
        )

        rate_df = pd.DataFrame({
            "매체": MEDIA,
            "이용률(%)": [
                Xmedia[m].mean() * 100
                for m in MEDIA
            ],
        })

        rate_df = rate_df.sort_values(
            "이용률(%)",
            ascending=False,
        )

        draw(
            alt.Chart(rate_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "이용률(%):Q",
                    scale=alt.Scale(
                        domain=[0, 100]
                    ),
                ),
                y=alt.Y(
                    "매체:N",
                    sort="-x",
                    title=None,
                ),
                tooltip=[
                    "매체",
                    alt.Tooltip(
                        "이용률(%):Q",
                        format=".1f",
                    ),
                ],
            )
            .properties(height=350)
        )

        top = rate_df.iloc[0]

        st.write(
            f"가장 많이 이용한 매체는 "
            f"**{top['매체']}** "
            f"({top['이용률(%)']:.1f}%)입니다."
        )

        section(
            "학교급과 학년의 분포"
        )

        a, b = st.columns(2)

        with a:

            school_df = (
                school.value_counts()
                .reindex(
                    list(SCHOOL.values()),
                    fill_value=0,
                )
            )

            st.bar_chart(school_df)

        with b:

            grade_df = (
                grade.value_counts()
                .reindex(
                    list(GRADE.values()),
                    fill_value=0,
                )
            )

            st.bar_chart(grade_df)

        st.caption(
            "학년은 숫자 1~9를 실제 학년명 "
            "초4~고3으로 바꾸어 표시했습니다."
        )

        st.write(
            f"가장 응답자가 많은 학교급은 "
            f"**{school_df.idxmax()}** "
            f"({int(school_df.max()):,}명), "
            f"학년은 **{grade_df.idxmax()}** "
            f"({int(grade_df.max()):,}명)입니다."
        )

    # ----------------------------------------------
    # X-Y 관계 분석
    # ----------------------------------------------

    with tab_relation:

        section(
            "매체 이용(X)과 생성형 AI 이용(Y)의 관계"
        )

        st.write(
            "입력 변수끼리의 상관관계가 아니라, "
            "**입력 변수와 우리가 예측하려는 "
            "정답의 관계**를 확인합니다."
        )

        cor = pd.DataFrame({
            "매체": MEDIA,
            "상관계수": [
                Xmedia[m].corr(y)
                for m in MEDIA
            ],
        })

        cor["절댓값"] = (
            cor["상관계수"].abs()
        )

        cor = cor.sort_values(
            "절댓값",
            ascending=False,
        )

        draw(
            alt.Chart(cor)
            .mark_bar()
            .encode(
                x=alt.X(
                    "상관계수:Q",
                    scale=alt.Scale(
                        domain=[-1, 1]
                    ),
                ),
                y=alt.Y(
                    "매체:N",
                    sort=list(cor["매체"]),
                    title=None,
                ),
                tooltip=[
                    "매체",
                    alt.Tooltip(
                        "상관계수:Q",
                        format="+.3f",
                    ),
                ],
            )
            .properties(height=360)
        )

        st.caption(
            "양수는 함께 나타나는 경향, "
            "음수는 반대 경향, 0에 가까우면 "
            "선형 관련성이 약함을 뜻합니다. "
            "인과관계는 아닙니다."
        )

        section(
            "AI 이용자와 미이용자의 매체 이용률"
        )

        comparisons = pd.DataFrame([
            {
                "매체": m,
                "집단": label,
                "이용률(%)": (
                    100 * Xmedia.loc[
                        y == value, m
                    ].mean()
                ),
            }
            for m in MEDIA
            for label, value in [
                ("AI 이용", 1),
                ("AI 미이용", 0),
            ]
        ])

        draw(
            alt.Chart(comparisons)
            .mark_bar()
            .encode(
                x=alt.X(
                    "이용률(%):Q",
                    scale=alt.Scale(
                        domain=[0, 100]
                    ),
                ),
                y=alt.Y(
                    "매체:N",
                    title=None,
                ),
                yOffset="집단:N",
                color="집단:N",
                tooltip=[
                    "매체",
                    "집단",
                    alt.Tooltip(
                        "이용률(%):Q",
                        format=".1f",
                    ),
                ],
            )
            .properties(height=410)
        )

        comparison_pivot = comparisons.pivot(
            index="매체",
            columns="집단",
            values="이용률(%)",
        )

        diffs = (
            comparison_pivot["AI 이용"]
            - comparison_pivot["AI 미이용"]
        ).abs()

        biggest = diffs.idxmax()

        difference = abs(
            comparison_pivot.loc[
                biggest, "AI 이용"
            ]
            - comparison_pivot.loc[
                biggest, "AI 미이용"
            ]
        )

        st.success(
            f"두 집단 간 이용률 차이가 "
            f"가장 큰 매체는 "
            f"**{biggest}**이며 "
            f"차이는 **{difference:.1f}%p**입니다."
        )

        section(
            "학교급·학년별 생성형 AI 이용률"
        )

        groups = [
            (
                "학교급",
                school,
                list(SCHOOL.values()),
            ),
            (
                "학년",
                grade,
                list(GRADE.values()),
            ),
        ]

        for title, series, order in groups:

            rate = (
                pd.DataFrame({
                    title: series,
                    "AI 이용": y,
                })
                .groupby(title)["AI 이용"]
                .mean()
                .mul(100)
                .reindex(order)
                .reset_index(name="AI 이용률(%)")
            )

            st.write("**" + title + "**")

            draw(
                alt.Chart(rate)
                .mark_bar()
                .encode(
                    x=alt.X(
                        f"{title}:N",
                        sort=order,
                        axis=alt.Axis(
                            labelAngle=0
                        ),
                    ),
                    y=alt.Y(
                        "AI 이용률(%):Q",
                        scale=alt.Scale(
                            domain=[0, 100]
                        ),
                    ),
                    tooltip=[
                        title,
                        alt.Tooltip(
                            "AI 이용률(%):Q",
                            format=".1f",
                        ),
                    ],
                )
                .properties(height=255)
            )

        st.info(
            "이 시각화는 예측에 도움이 될 수 있는 "
            "관계를 탐색한 것입니다. "
            "실제 모델이 잘 맞히는지는 "
            "검증 데이터로 평가합니다."
        )


# ====================================================
# ③ 기준선
# ====================================================

elif menu == "③ 기준선":

    trained = train_all(X, y)

    base = trained["baseline"]

    label = (
        "AI 이용"
        if base["label"] == 1
        else "AI 미이용"
    )

    with st.container(border=True):

        section(
            "기준선이 무엇인가요?"
        )

        st.markdown(
            f"### 모든 학생을 "
            f"'{label}'이라고 예측"
        )

        st.write(
            "학습 데이터에서 가장 흔한 정답만 "
            "고르는 단순한 모델입니다. "
            "매체 이용 경험은 고려하지 않습니다."
        )

    section(
        "기준선 검증 성능"
    )

    show_five(
        base["scores"]
    )

    st.caption(
        "정확도: 맞힌 비율 · "
        "정밀도: 이용 예측 중 실제 이용 비율 · "
        "재현율: 실제 이용자를 찾아낸 비율 · "
        "F1: 정밀도와 재현율의 균형"
    )

    section(
        "기준선 혼동행렬"
    )

    matrix = confusion_display(
        trained["valid_y"],
        base["pred"],
    )

    if base["label"] == 1:

        st.warning(
            f"실제 AI 미이용자 "
            f"{matrix[0, 1]:,}명도 모두 "
            f"'AI 이용'이라고 잘못 예측했습니다. "
            f"재현율 1.000만으로 좋은 모델이라고 "
            f"볼 수 없습니다."
        )

    else:

        st.warning(
            f"실제 AI 이용자 "
            f"{matrix[1, 0]:,}명을 "
            f"찾아내지 못했습니다."
        )


# ====================================================
# ④ 모델 평가
# ====================================================

else:

    trained = train_all(X, y)

    r = trained["results"]

    best = trained["selected"]

    best_score = r[best]["scores"]

    base_score = trained["baseline"]["scores"]

    with st.container(border=True):

        st.write(
            "**검증 성능 기준 최종 선정 후보**"
        )

        st.markdown(
            f"### {best[1]} · "
            f"{best[0]} 전처리"
        )

        metrics_view({
            "검증 F1": (
                f"{best_score['F1']:.3f}"
            ),
            "기준선 F1": (
                f"{base_score['F1']:.3f}"
            ),
            "정확도 차이": (
                f"{(best_score['정확도'] - base_score['정확도'])*100:+.2f}%p"
            ),
        })

        st.caption(
            "검증 F1을 우선 비교하고, "
            "같으면 ROC-AUC와 정확도로 "
            "순위를 정했습니다."
        )

    t1, t2, t3, t4 = st.tabs([
        "모델 비교",
        "모델 원리·해석",
        "성능 개선",
        "최종 결론",
    ])

    # ----------------------------------------------
    # 1. 모델 비교
    # ----------------------------------------------

    with t1:

        section(
            "기존 모델 3개와 기준선 비교"
        )

        old = score_table(trained)

        st.dataframe(
            old.round(3),
            hide_index=True,
            use_container_width=True,
        )

        a, b = st.columns(2)

        for col, measure in [
            (a, "정확도"),
            (b, "F1"),
        ]:

            with col:

                st.write(
                    "**" + measure + "**"
                )

                draw(
                    alt.Chart(old)
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            f"{measure}:Q",
                            scale=alt.Scale(
                                domain=[0, 1]
                            ),
                        ),
                        y=alt.Y(
                            "모델:N",
                            sort=(
                                ["기준선"] + MODELS
                            ),
                            title=None,
                        ),
                        color="모델:N",
                        tooltip=[
                            "모델",
                            alt.Tooltip(
                                f"{measure}:Q",
                                format=".3f",
                            ),
                        ],
                    )
                    .properties(height=245)
                )

        st.info(
            "정확도가 가장 높은 모델과 "
            "F1이 가장 높은 모델은 "
            "다를 수 있습니다. "
            "모든 모델을 같은 검증 데이터에서 "
            "비교합니다."
        )

        with st.expander(
            "정확도·정밀도·재현율·"
            "F1·ROC-AUC 설명"
        ):

            st.write(
                "**정확도:** 전체에서 맞힌 비율"
            )

            st.write(
                "**정밀도:** AI 이용이라고 예측한 "
                "사람 중 실제 이용자 비율"
            )

            st.write(
                "**재현율:** 실제 이용자를 "
                "찾아낸 비율"
            )

            st.write(
                "**F1:** 정밀도와 재현율의 조화평균"
            )

            st.write(
                "**ROC-AUC:** 다양한 분류 기준에서 "
                "두 집단을 구분하는 능력"
            )

    # ----------------------------------------------
    # 2. 모델 원리 및 해석
    # ----------------------------------------------

    with t2:

        section(
            "세 가지 분류 모델의 원리와 선정 이유"
        )

        models_info = pd.DataFrame([
            [
                "로지스틱 회귀",
                "확률을 계산해 범주 예측",
                "구조가 단순하고 계수를 해석하기 쉬움",
            ],
            [
                "의사결정트리",
                "조건으로 단계별 분류",
                "질문과 분기 과정을 볼 수 있음",
            ],
            [
                "랜덤 포레스트",
                "여러 결정트리의 판단 결합",
                "복잡한 관계를 학습할 수 있음",
            ],
        ], columns=[
            "모델",
            "작동 원리",
            "선정 이유",
        ])

        st.dataframe(
            models_info,
            hide_index=True,
            use_container_width=True,
        )

        setting = st.radio(
            "전처리 설정",
            ["기존", "개선"],
            horizontal=True,
        )

        name = st.selectbox(
            "자세히 볼 모델",
            MODELS,
        )

        model = r[
            (setting, name)
        ]["pipeline"]

        estimator = model.named_steps["모델"]

        transformed_names = (
            model.named_steps["전처리"]
            .get_feature_names_out()
        )

        # 로지스틱 회귀
        if name == "로지스틱 회귀":

            coef = pd.DataFrame({
                "항목": [
                    pretty_feature(n)
                    for n in transformed_names
                ],
                "계수": estimator.coef_[0],
            })

            coef["절댓값"] = (
                coef["계수"].abs()
            )

            coef = coef.sort_values(
                "절댓값",
                ascending=False,
            )

            top = coef.head(12)

            draw(
                alt.Chart(top)
                .mark_bar()
                .encode(
                    x="계수:Q",
                    y=alt.Y(
                        "항목:N",
                        sort=list(top["항목"]),
                        axis=alt.Axis(
                            labelLimit=340
                        ),
                        title=None,
                    ),
                    tooltip=[
                        "항목",
                        alt.Tooltip(
                            "계수:Q",
                            format="+.4f",
                        ),
                    ],
                )
                .properties(height=400)
            )

            with st.expander(
                "전체 계수 확인"
            ):

                st.dataframe(
                    coef[
                        ["항목", "계수"]
                    ].round(4),
                    hide_index=True,
                    use_container_width=True,
                )

            st.caption(
                "계수는 다른 변수가 같을 때의 "
                "관련 방향이며 원인을 증명하지는 "
                "않습니다. 원-핫 인코딩 항목은 "
                "비교 기준 범주에도 주의해야 합니다."
            )

        # 의사결정트리
        elif name == "의사결정트리":

            explore_tree(
                model,
                setting,
            )

        # 랜덤 포레스트
        else:

            imp = pd.DataFrame({
                "항목": [
                    original_feature(n)
                    for n in transformed_names
                ],
                "중요도": (
                    estimator.feature_importances_
                ),
            })

            imp = (
                imp.groupby(
                    "항목",
                    as_index=False,
                )["중요도"]
                .sum()
                .sort_values(
                    "중요도",
                    ascending=False,
                )
            )

            draw(
                alt.Chart(imp)
                .mark_bar()
                .encode(
                    x="중요도:Q",
                    y=alt.Y(
                        "항목:N",
                        sort="-x",
                        title=None,
                    ),
                    tooltip=[
                        "항목",
                        alt.Tooltip(
                            "중요도:Q",
                            format=".4f",
                        ),
                    ],
                )
                .properties(height=340)
            )

            st.caption(
                "불순도 기반 중요도입니다. "
                "값이 크다고 원인이나 인과적 "
                "영향이 큰 것은 아닙니다."
            )

        section(
            "선택한 모델의 검증 성능"
        )

        show_five(
            r[(setting, name)]["scores"]
        )

    # ----------------------------------------------
    # 3. 성능 개선
    # ----------------------------------------------

    with t3:

        section(
            "데이터 전처리로 성능이 개선되었을까?"
        )

        with st.container(border=True):

            st.write(
                "① **학교급 제거:** 학년(초4~고3)에서 "
                "학교급을 알 수 있어 입력 중복을 줄임"
            )

            st.write(
                "② **이용 매체 개수 생성:** "
                "9개 매체 중 사용한 매체의 수를 추가"
            )

            st.write(
                "③ **같은 조건에서 재학습:** "
                "분할·알고리즘 설정을 바꾸지 않고 비교"
            )

            st.caption(
                "새로운 설문 응답을 만들거나 "
                "기존 행을 삭제하지 않았습니다. "
                "새 변수는 기존 X에서만 계산했습니다."
            )

        comparison_rows = []

        for name in MODELS:

            old_score = r[
                ("기존", name)
            ]["scores"]

            new_score = r[
                ("개선", name)
            ]["scores"]

            comparison_rows.append({
                "모델": name,
                "기존 정확도": (
                    old_score["정확도"]
                ),
                "개선 정확도": (
                    new_score["정확도"]
                ),
                "정확도 차이(%p)": (
                    (
                        new_score["정확도"]
                        - old_score["정확도"]
                    ) * 100
                ),
                "기존 F1": old_score["F1"],
                "개선 F1": new_score["F1"],
                "F1 차이": (
                    new_score["F1"]
                    - old_score["F1"]
                ),
            })

        compare = pd.DataFrame(
            comparison_rows
        )

        st.dataframe(
            compare.round(3),
            hide_index=True,
            use_container_width=True,
        )

        plot = pd.DataFrame([
            {
                "모델": name,
                "설정": setting,
                "F1": r[
                    (setting, name)
                ]["scores"]["F1"],
            }
            for name in MODELS
            for setting in ["기존", "개선"]
        ])

        draw(
            alt.Chart(plot)
            .mark_bar()
            .encode(
                x=alt.X(
                    "모델:N",
                    sort=MODELS,
                    axis=alt.Axis(
                        labelAngle=0
                    ),
                ),
                xOffset="설정:N",
                y=alt.Y(
                    "F1:Q",
                    scale=alt.Scale(
                        domain=[0, 1]
                    ),
                ),
                color="설정:N",
                tooltip=[
                    "모델",
                    "설정",
                    alt.Tooltip(
                        "F1:Q",
                        format=".3f",
                    ),
                ],
            )
            .properties(height=315)
        )

        count_up = int(
            (compare["F1 차이"] > 0).sum()
        )

        if count_up:

            st.success(
                f"세 모델 중 {count_up}개에서 "
                f"전처리 변경 후 F1이 높아졌습니다."
            )

        else:

            st.warning(
                "전처리 변경으로 F1이 높아진 "
                "모델이 없습니다. 개선에 성공했다고 "
                "주장하지 않습니다."
            )

        best_old = max(
            [
                ("기존", n)
                for n in MODELS
            ],
            key=lambda k: ranking(r[k]),
        )

        best_new = max(
            [
                ("개선", n)
                for n in MODELS
            ],
            key=lambda k: ranking(r[k]),
        )

        st.write(
            f"기존 최고: "
            f"**{best_old[1]} "
            f"(F1 {r[best_old]['scores']['F1']:.3f})**"
            f" · 개선 최고: "
            f"**{best_new[1]} "
            f"(F1 {r[best_new]['scores']['F1']:.3f})**"
        )

        st.caption(
            "개선된 결과가 기존 최고 모델보다 "
            "낮을 때는 기존 전처리 설정도 "
            "최종 선정 후보로 남깁니다."
        )

        with st.expander(
            "왜 결측치 대체·무리한 "
            "재표본추출을 하지 않았을까?"
        ):

            st.write(
                f"선택 CSV 결측 셀은 "
                f"**{missing}개**, "
                f"AI 이용 비율은 "
                f"**{y.mean()*100:.1f}%**입니다. "
                "결측치가 없다면 대체가 필요하지 "
                "않습니다. 두 범주의 비율도 "
                "비슷해 이번에는 인위적으로 "
                "수를 맞추지 않았습니다."
            )

    # ----------------------------------------------
    # 4. 최종 결론
    # ----------------------------------------------

    with t4:

        section(
            "검증 지표로 선정한 모델과 최종 평가"
        )

        best_s = r[best]["scores"]

        base_s = trained["baseline"]["scores"]

        st.success(
            f"선정: **{best[1]} · "
            f"{best[0]} 전처리** "
            f"(검증 F1 우선)"
        )

        metrics_view({
            "선정 모델 F1": (
                f"{best_s['F1']:.3f}"
            ),
            "기준선 F1": (
                f"{base_s['F1']:.3f}"
            ),
            "정확도 개선": (
                f"{(best_s['정확도'] - base_s['정확도'])*100:+.2f}%p"
            ),
        })

        st.write(
            "학습 60%·검증 20%·최종 평가 20%로 "
            "분할했습니다. 모델을 선정할 때 "
            "최종 평가 데이터는 사용하지 않았습니다."
        )

        final_table = pd.DataFrame([
            {
                "모델": "최종 평가 기준선",
                **trained["final_baseline"],
            },
            {
                "모델": (
                    f"최종 선정("
                    f"{best[0]}·{best[1]})"
                ),
                **trained["final_scores"],
            },
        ])

        st.dataframe(
            final_table.round(3),
            hide_index=True,
            use_container_width=True,
        )

        final = trained["final_scores"]

        base_final = trained["final_baseline"]

        delta_acc = (
            final["정확도"]
            - base_final["정확도"]
        ) * 100

        delta_f1 = (
            final["F1"]
            - base_final["F1"]
        )

        metrics_view({
            "최종 정확도 차이": (
                f"{delta_acc:+.2f}%p"
            ),
            "최종 F1 차이": (
                f"{delta_f1:+.3f}"
            ),
        })

        if (
            delta_acc > 0
            and delta_f1 > 0
        ):

            st.success(
                "선정 모델은 별도 최종 평가에서도 "
                "기준선보다 정확도와 F1이 "
                "높았습니다."
            )

        elif (
            delta_acc > 0
            or delta_f1 > 0
        ):

            st.warning(
                "일부 지표는 개선되었지만 "
                "모든 지표가 개선된 것은 아닙니다."
            )

        else:

            st.warning(
                "최종 평가에서 기준선보다 "
                "두 지표 모두 좋아지지 않았습니다."
            )

        section(
            "우리 탐구 질문에 대한 해석"
        )

        if delta_acc > 0:

            st.write(
                f"기준선 대비 정확도가 "
                f"{delta_acc:.2f}%p 높았습니다. "
                "따라서 이번 설문 데이터에서는 "
                "입력 변수로 AI 이용 여부를 "
                "단순한 다수 범주 예측보다 "
                "더 잘 맞힐 수 있었습니다."
            )

        else:

            st.write(
                "이번 최종 평가에서는 정확도가 "
                "기준선보다 높지 않아 "
                "입력 변수가 예측에 도움이 "
                "된다고 단정할 수 없습니다."
            )

        st.write(
            f"다만 최종 정확도는 "
            f"**{final['정확도']*100:.1f}%**이므로 "
            "모든 학생을 정확하게 "
            "예측한 것은 아닙니다."
        )

        st.caption(
            "설문 응답을 이용한 예측 분석이지 "
            "AI 이용의 원인을 규명한 결과는 "
            "아닙니다. 설문 가중치를 적용하지 "
            "않았고 다른 시기·집단에서도 "
            "같은 성능이 나온다고 "
            "보장할 수 없습니다."
        )

        with st.expander(
            "포스터에 넣기 좋은 핵심 내용 체크"
        ):

            poster_items = [
                "데이터셋과 출처",
                "탐구 동기",
                "46열→13열 선택과 전처리",
                "모델 후보 3개",
                "기준선과 평가 지표",
                "최종 모델 선정 근거",
                "핵심 발견 1개와 수치",
                "내가 직접 수정한 점",
                "한계와 추가 탐구",
                "실제 느낀 점",
            ]

            for message in poster_items:
                st.write("• " + message)
