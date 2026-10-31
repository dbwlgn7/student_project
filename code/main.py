
from pathlib import Path

import pandas as pd
import altair as alt
import streamlit as st

# ==========================================
# 1. 기본 설정
# ==========================================

st.set_page_config(
    page_title="청소년 디지털 매체 이용 분석",
    page_icon="📊",
    layout="wide"
)

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data"

SELECTED_FILE = DATA_DIR / "청소년_AI이용분류_ID포함13열.csv"
ORIGINAL_FILE = DATA_DIR / "청소년_AI이용분류_정제데이터.csv"

KEEP = [1, 3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 36, 40]

GRADE_LABELS = {
    1: "초4", 2: "초5", 3: "초6",
    4: "중1", 5: "중2", 6: "중3",
    7: "고1", 8: "고2", 9: "고3"
}

# ==========================================
# 2. 함수
# ==========================================

@st.cache_data
def load_csv(path):
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False
    )


def short_name(column):
    if " - " in column:
        return column.split(" - ", 1)[1].split(") ", 1)[-1]
    return column


def get_columns(df):
    cols = list(df.columns)

    target = next(
        (c for c in cols if "생성형 AI" in c and "이용 경험" in c),
        None
    )

    id_col = next(
        (c for c in cols if "응답자 ID" in c),
        cols[0]
    )

    school = next(
        (c for c in cols if "학교급(기본)" in c),
        None
    )

    grade = next(
        (c for c in cols if "학년별" in c),
        None
    )

    media = [
        c for c in cols
        if c not in [target, id_col, school, grade]
    ]

    return target, id_col, school, grade, media


def get_reason(number):
    if number == 1:
        return "응답자를 구분하기 위해 유지하였다. 분류 모델의 입력에는 사용하지 않는다."

    if number == 12:
        return "생성형 AI 이용 여부는 분류 모델이 예측해야 하는 타깃 변수이므로 유지하였다."

    if number in [36, 40]:
        return "학교급과 학년에 따른 디지털 매체 이용 특성을 함께 분석하기 위해 유지하였다."

    if number in KEEP:
        return "청소년의 디지털 매체 이용 경험을 나타내므로 탐구 주제와 직접적인 관련이 있어 유지하였다."

    if number == 2:
        return "학교를 구분하기 위한 번호로 디지털 매체 이용 특성을 나타내지 않아 제외하였다."

    if number in [8, 10, 15, 16]:
        return "디지털 매체 이용과 관련이 있지만 이번 탐구에서 분석할 매체의 범위를 제한하기 위해 제외하였다."

    if number in [17, 18]:
        return "종이신문과 종이책은 디지털 매체가 아니므로 탐구 주제와 직접적인 관련성이 낮아 제외하였다."

    if number in [19, 20, 21]:
        return "자주 이용하는 매체에 생성형 AI가 직접 포함될 수 있어 정답 정보가 유출되는 것을 방지하기 위해 제외하였다."

    if number in [22, 23, 24, 25, 26, 27, 28, 29]:
        return "개인정보 보호 및 온라인 안전 관련 경험으로, 디지털 매체의 이용 여부 자체를 나타내지 않아 제외하였다."

    if number in [30, 31, 32, 33]:
        return "디지털 매체 이용 특성보다는 교육 경험과 교육 효과를 나타내므로 제외하였다."

    if number in [34, 35]:
        return "개인 및 학교 특성으로 디지털 매체 이용 경험 중심의 탐구 범위에서 제외하였다."

    if number in [37, 38, 39, 43, 44]:
        return "기본 학교급 변수와 유사한 정보를 포함하므로 중복 정보를 줄이기 위해 제외하였다."

    if number in [41, 42]:
        return "지역 정보는 매체 이용 여부 자체를 나타내지 않아 이번 분석에서 제외하였다."

    if number in [45, 46]:
        return "조사 가중치로서 매체 이용 경험을 나타내는 입력 변수가 아니므로 제외하였다."

    return "탐구 주제와의 관련성을 고려해 제외하였다."


# ==========================================
# 3. 사이드바 메뉴
# ==========================================

st.sidebar.title("🧭 탐구 메뉴")

menu = st.sidebar.radio(
    "이동할 메뉴",
    [
        "① 탐구 소개",
        "② 전처리 및 근거",
        "③ 데이터 분포 시각화",
        "④ 디지털 매체 이용 특성 간 관계"
    ]
)

uploaded = st.sidebar.file_uploader(
    "13열 CSV 직접 올리기 (선택)",
    type="csv"
)

# ==========================================
# 4. 데이터 불러오기
# ==========================================

if uploaded is not None:
    df = pd.read_csv(
        uploaded,
        encoding="utf-8-sig"
    )

elif SELECTED_FILE.exists():
    df = load_csv(str(SELECTED_FILE))

else:
    st.error(
        "13열 CSV를 찾지 못했습니다. "
        "GitHub data 폴더를 확인해 주세요."
    )
    st.stop()

target, id_col, school, grade, media = get_columns(df)

if target is None:
    st.error("생성형 AI 이용 여부 변수를 찾지 못했습니다.")
    st.stop()

# 원본 응답값은 변경하지 않는다.
# 그래프 계산을 위한 별도 데이터만 생성한다.

X_media = df[media].copy()

for col in media:
    X_media[col] = pd.to_numeric(
        X_media[col],
        errors="coerce"
    ).map({1: 1, 2: 0})

st.title("📊 청소년 디지털 매체 이용과 생성형 AI 분류 탐구")

st.caption(
    "청소년의 디지털 매체 이용 특성으로 "
    "생성형 AI 이용 여부를 분류할 수 있을까?"
)

# ==========================================
# 메뉴 1. 탐구 소개
# ==========================================

if menu == "① 탐구 소개":

    st.header("탐구 소개")

    st.write(
        "본 탐구에서는 청소년의 디지털 매체 이용 경험을 "
        "분석하고, 이러한 특성이 생성형 AI 이용 여부를 "
        "분류하는 데 활용될 수 있는지 알아보고자 한다."
    )

    a, b, c = st.columns(3)

    a.metric("전체 응답자", f"{len(df):,}명")
    b.metric("선택한 변수", len(df.columns))
    c.metric("전체 결측치", int(df.isna().sum().sum()))

    st.subheader("탐구 목표")

    st.write(
        "OTT, 숏폼, SNS, 게임 등 다양한 디지털 매체의 "
        "이용 경험을 확인하고, 이후 분류 알고리즘을 "
        "활용해 생성형 AI 이용 여부를 예측한다."
    )

    st.subheader("입력 변수(X)")

    st.write(", ".join(short_name(c) for c in media))

    st.write("학교급과 학년도 추가 입력 변수로 활용한다.")

    st.subheader("타깃 변수(Y)")

    st.write("생성형 AI 이용 여부")

    st.info(
        "응답자 ID는 식별을 위해 유지하지만 "
        "모델 학습에는 사용하지 않는다. "
        "생성형 AI 이용 여부는 타깃으로만 사용한다."
    )

    st.subheader("탐구 진행 단계")

    st.write(
        "① 변수 선택과 전처리 → "
        "② 데이터 분포 확인 → "
        "③ 변수 간 관계 분석 → "
        "④ 분류 모델 선정 및 평가(추후 진행)"
    )


# ==========================================
# 메뉴 2. 전처리 및 근거
# ==========================================

elif menu == "② 전처리 및 근거":

    st.header("데이터 전처리 및 근거")

    a, b, c = st.columns(3)

    a.metric("원본 변수", 46)
    b.metric("유지한 변수", 13)
    c.metric("제외한 변수", 33)

    st.write(
        "원본 데이터의 46개 변수 중 탐구 주제와 "
        "관련된 13개 변수를 선택하였다. "
        "선택한 열의 응답 코드와 행 순서는 변경하지 않았다."
    )

    st.subheader("데이터 품질 검사")

    a, b, c = st.columns(3)

    a.metric("응답자", f"{len(df):,}명")
    b.metric("결측치", int(df.isna().sum().sum()))
    c.metric("중복 ID", int(df[id_col].duplicated().sum()))

    st.subheader("전체 변수의 전처리 근거")

    if ORIGINAL_FILE.exists():

        original = load_csv(str(ORIGINAL_FILE))

        records = []

        for i, col in enumerate(original.columns, start=1):
            records.append({
                "번호": i,
                "변수명": col,
                "처리": "유지" if i in KEEP else "제외",
                "근거": get_reason(i)
            })

        report = pd.DataFrame(records)

        choice = st.selectbox(
            "표시할 변수",
            ["전체", "유지", "제외"]
        )

        if choice != "전체":
            report = report[report["처리"] == choice]

        st.dataframe(
            report,
            hide_index=True,
            width="stretch"
        )

        st.subheader("전처리 전후 비교")

        expected = original.iloc[
            :, [i - 1 for i in KEEP]
        ].reset_index(drop=True)

        actual = df.reset_index(drop=True)

        same_headers = (
            expected.columns.tolist() == actual.columns.tolist()
        )

        same_values = (
            expected.astype("string").fillna("__NA__")
            .equals(
                actual.astype("string").fillna("__NA__")
            )
        )

        if same_headers and same_values:
            st.success(
                "원본과 전처리 데이터의 열 이름, "
                "열 순서 및 셀 값이 일치합니다."
            )
        else:
            st.warning(
                "원본과 선택 데이터에 차이가 발견되었습니다."
            )

    else:
        st.warning(
            "원본 46열 CSV를 찾지 못했습니다. "
            "GitHub data 폴더의 파일 이름을 확인해 주세요."
        )

    st.info(
        "변수를 제외한 이유는 탐구 주제와의 관련성에 "
        "따른 판단이며, 예측력이 없다고 검증된 것은 아닙니다."
    )


# ==========================================
# 메뉴 3. 데이터 분포 시각화
# ==========================================

elif menu == "③ 데이터 분포 시각화":

    st.header("디지털 매체 이용 분포 분석")

    st.write(
        "생성형 AI 이용 여부를 예측하기 전에 "
        "입력 변수들의 분포를 확인한다."
    )

    st.subheader("매체별 이용 경험 비율")

    rates = (
        X_media.mean() * 100
    ).sort_values(ascending=False)

    rate_df = pd.DataFrame({
        "매체": [short_name(c) for c in rates.index],
        "이용률": rates.values
    })

    chart = alt.Chart(rate_df).mark_bar().encode(
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
    ).properties(height=400)

    st.altair_chart(chart, width="stretch")

    st.dataframe(
        rate_df.round(2),
        hide_index=True,
        width="stretch"
    )

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
        y=alt.Y("매체:N", title=None),
        x=alt.X(
            "비율:Q",
            stack="zero",
            scale=alt.Scale(domain=[0, 100]),
            title="비율(%)"
        ),
        color="응답:N",
        tooltip=[
            "매체",
            "응답",
            alt.Tooltip("비율:Q", format=".2f")
        ]
    ).properties(height=400)

    st.altair_chart(chart2, width="stretch")

    if school is not None:

        st.subheader("학교급별 응답자 수")

        school_data = pd.to_numeric(
            df[school], errors="coerce"
        ).map({
            1: "초등학교",
            2: "중학교",
            3: "고등학교"
        })

        school_counts = school_data.value_counts().reindex(
            ["초등학교", "중학교", "고등학교"],
            fill_value=0
        )

        st.bar_chart(school_counts)

    if grade is not None:

        st.subheader("학년별 응답자 수")

        grade_data = pd.to_numeric(
            df[grade], errors="coerce"
        ).map(GRADE_LABELS)

        grade_counts = grade_data.value_counts().reindex(
            list(GRADE_LABELS.values()),
            fill_value=0
        )

        st.bar_chart(grade_counts)

        st.caption(
            "원본 학년 코드 1~9를 코드북에 따라 "
            "초4~고3으로 변경하여 표시했습니다."
        )

    st.info(
        "이용률이 높거나 낮다는 이유만으로 "
        "해당 변수를 삭제하지 않습니다. "
        "예측에 도움이 되는지는 이후 모델 평가에서 확인합니다."
    )


# ==========================================
# 메뉴 4. 변수 간 관계 분석
# ==========================================

elif menu == "④ 디지털 매체 이용 특성 간 관계":

    st.header("디지털 매체 이용 특성 간 관계")

    st.write(
        "두 디지털 매체의 이용 여부가 얼마나 비슷하게 "
        "나타나는지 상관계수를 통해 확인합니다."
    )

    st.subheader("상관계수의 의미")

    st.write(
        "상관계수가 1에 가까우면 양의 관계, "
        "-1에 가까우면 음의 관계, "
        "0에 가까우면 선형적인 관계가 약하다는 뜻입니다."
    )

    corr = X_media.corr()

    labels = {
        col: short_name(col)
        for col in media
    }

    renamed = corr.rename(
        index=labels,
        columns=labels
    )

    heat = renamed.stack().reset_index()
    heat.columns = [
        "매체 A",
        "매체 B",
        "상관계수"
    ]

    heat["표시값"] = heat["상관계수"].map(
        lambda x: f"{x:.2f}"
        if pd.notna(x) else "-"
    )

    st.subheader("매체 간 상관관계 히트맵")

    base = alt.Chart(heat).encode(
        x=alt.X(
            "매체 A:N",
            title=None,
            axis=alt.Axis(labelAngle=-45)
        ),
        y=alt.Y(
            "매체 B:N",
            title=None
        )
    )

    rects = base.mark_rect().encode(
        color=alt.Color(
            "상관계수:Q",
            scale=alt.Scale(
                scheme="redblue",
                domain=[-1, 1]
            )
        ),
        tooltip=[
            "매체 A",
            "매체 B",
            alt.Tooltip(
                "상관계수:Q",
                format=".3f"
            )
        ]
    )

    texts = base.mark_text(
        fontSize=11
    ).encode(
        text="표시값:N",
        color=alt.condition(
            "abs(datum['상관계수']) > 0.6",
            alt.value("white"),
            alt.value("black")
        )
    )

    st.altair_chart(
        (rects + texts).properties(height=480),
        width="stretch"
    )

    st.subheader("상관관계가 큰 매체 조합")

    pairs = []

    for i, col_a in enumerate(media):

        for col_b in media[i + 1:]:

            value = corr.loc[col_a, col_b]

            pairs.append({
                "매체 A": short_name(col_a),
                "매체 B": short_name(col_b),
                "상관계수": value
            })

    pairs_df = pd.DataFrame(pairs)

    pairs_df["절댓값"] = (
        pairs_df["상관계수"].abs()
    )

    pairs_df = pairs_df.sort_values(
        "절댓값",
        ascending=False
    )

    st.dataframe(
        pairs_df.drop(
            columns="절댓값"
        ).head(10).round(3),
        hide_index=True,
        width="stretch"
    )

    st.info(
        "상관관계가 높다고 해서 두 변수 중 하나를 "
        "무조건 삭제하지는 않습니다. "
        "실제 분류 모델의 성능을 확인한 뒤 판단해야 합니다."
    )
