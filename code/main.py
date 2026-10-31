
from pathlib import Path
import io

import pandas as pd
import altair as alt
import streamlit as st


# =========================================
# 1. 기본 설정
# =========================================

st.set_page_config(
    page_title="청소년 생성형 AI 이용 분류 탐구",
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

SCHOOL_LABELS = {
    1: "초등학교",
    2: "중학교",
    3: "고등학교"
}


# =========================================
# 2. 공통 함수
# =========================================

@st.cache_data
def load_csv(path):
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False
    )


def short_name(name):
    if " - " in name:
        return name.split(" - ", 1)[1].split(") ", 1)[-1]
    return name


def find_columns(df):
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


def reason(number):
    if number == 1:
        return "응답자를 구분하기 위해 유지하였다. 분류 모델의 입력에는 사용하지 않는다."

    if number == 12:
        return "생성형 AI 이용 여부는 예측해야 할 타깃 변수이므로 유지하였다."

    if number in [36, 40]:
        return "학교급과 학년에 따른 디지털 매체 이용 특성을 함께 분석하기 위해 유지하였다."

    if number in KEEP:
        return "디지털 매체 이용 경험을 나타내므로 생성형 AI 이용 여부 분류에 활용하기 위해 유지하였다."

    if number == 2:
        return "학교를 구분하는 번호로 디지털 매체 이용 특성을 직접 나타내지 않아 제외하였다."

    if number in [8, 10, 15, 16]:
        return "디지털 매체 이용과 관련이 있지만 이번 탐구의 분석 범위를 제한하기 위해 제외하였다."

    if number in [17, 18]:
        return "종이신문과 종이책은 디지털 매체가 아니므로 탐구 주제와 직접적인 관련성이 낮아 제외하였다."

    if number in [19, 20, 21]:
        return "자주 이용하는 매체에 생성형 AI가 직접 포함될 수 있어 정답 정보가 입력 변수에 포함되는 것을 방지하기 위해 제외하였다."

    if number in [22, 23, 24, 25, 26, 27, 28, 29]:
        return "개인정보 보호 및 온라인 안전 경험으로 매체 이용 여부 자체를 나타내지 않아 제외하였다."

    if number in [30, 31, 32, 33]:
        return "디지털 매체 이용보다 교육 경험과 교육 효과를 나타내므로 제외하였다."

    if number in [34, 35]:
        return "개인 또는 학교의 특성으로 디지털 매체 이용 경험 중심의 분석 범위에서 제외하였다."

    if number in [37, 38, 39, 43, 44]:
        return "기본 학교급과 유사한 정보를 포함하므로 중복 정보를 줄이기 위해 제외하였다."

    if number in [41, 42]:
        return "지역 정보로 디지털 매체 이용 경험에 집중하기 위해 제외하였다."

    if number in [45, 46]:
        return "조사 가중치로 매체 이용 경험을 나타내는 변수가 아니므로 모델 입력에서 제외하였다."

    return "탐구 주제와의 관련성을 고려하여 제외하였다."


def bar_chart(data, x, y, height=350):
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X(x),
        y=alt.Y(y, sort="-x"),
        tooltip=list(data.columns)
    ).properties(height=height)

    st.altair_chart(chart, width="stretch")


# =========================================
# 3. 사이드바
# =========================================

st.sidebar.title("🧭 탐구 메뉴")

menu = st.sidebar.radio(
    "분석 단계 선택",
    [
        "① 탐구 소개",
        "② 전처리 및 근거",
        "③ 데이터 분포 시각화",
        "④ 입력 변수와 타깃의 관계"
    ]
)

uploaded = st.sidebar.file_uploader(
    "13열 CSV 직접 업로드 (선택)",
    type=["csv"]
)


# =========================================
# 4. 데이터 불러오기
# =========================================

if uploaded is not None:
    df = pd.read_csv(
        io.BytesIO(uploaded.getvalue()),
        encoding="utf-8-sig"
    )

elif SELECTED_FILE.exists():
    df = load_csv(str(SELECTED_FILE))

else:
    st.error("13열 CSV를 찾지 못했습니다.")
    st.info("GitHub data 폴더의 파일을 확인해 주세요.")
    st.stop()

target, id_col, school, grade, media = find_columns(df)

if target is None:
    st.error("생성형 AI 이용 여부 변수를 찾지 못했습니다.")
    st.stop()

# 원본 데이터를 변경하지 않고 분석용 복사본 생성
# 코드북: 1=이용함, 2=전혀 이용 안함

X_media = df[media].copy()

for col in media:
    X_media[col] = pd.to_numeric(
        X_media[col],
        errors="coerce"
    ).map({1: 1, 2: 0})

Y = pd.to_numeric(
    df[target],
    errors="coerce"
).map({1: 1, 2: 0})

if X_media.isna().any().any() or Y.isna().any():
    st.error("매체 이용 변수 또는 타깃에 결측치나 잘못된 코드가 있습니다.")
    st.stop()

st.title("📊 청소년 디지털 매체 이용과 생성형 AI 분류 탐구")

st.caption(
    "탐구 주제: 청소년의 디지털 매체 이용 특성으로 "
    "생성형 AI 이용 여부를 분류할 수 있을까?"
)


# =========================================
# 메뉴 1. 탐구 소개
# =========================================

if menu == "① 탐구 소개":

    st.header("탐구 소개")

    st.write(
        "청소년의 디지털 매체 이용 경험을 분석하고 "
        "이러한 특성이 생성형 AI 이용 여부를 "
        "분류하는 데 활용될 수 있는지 알아본다."
    )

    a, b, c = st.columns(3)

    a.metric("전체 응답자", f"{len(df):,}명")
    b.metric("선택 변수", len(df.columns))
    c.metric("결측치", int(df.isna().sum().sum()))

    st.subheader("입력 변수(X)")

    st.write(", ".join(short_name(c) for c in media))

    st.write("추가 입력 변수: 학교급, 학년")

    st.subheader("타깃 변수(Y)")

    st.write("생성형 AI 이용 여부")

    st.info(
        "생성형 AI 이용 여부는 예측해야 할 정답입니다. "
        "모델 입력에는 포함하지 않으며, "
        "응답자 ID도 모델 학습에 사용하지 않습니다."
    )

    st.subheader("탐구 진행 과정")

    st.write(
        "① 데이터 전처리 → "
        "② 입력 변수 분포 시각화 → "
        "③ 입력 변수와 타깃 관계 분석 → "
        "④ 분류 모델 비교 및 평가(추후 진행)"
    )


# =========================================
# 메뉴 2. 전처리 및 근거
# =========================================

elif menu == "② 전처리 및 근거":

    st.header("데이터 전처리 및 근거")

    a, b, c = st.columns(3)

    a.metric("원본 변수", 46)
    b.metric("유지한 변수", 13)
    c.metric("제외한 변수", 33)

    st.write(
        "원본 46개 변수 중 탐구에 사용할 13개를 선택하였다. "
        "선택한 변수의 응답 코드, 열 이름, 행 순서는 유지하였다."
    )

    st.subheader("데이터 품질 검사")

    a, b, c = st.columns(3)

    a.metric("응답자 수", f"{len(df):,}")
    b.metric("결측치", int(df.isna().sum().sum()))
    c.metric("중복 ID", int(df[id_col].duplicated().sum()))

    st.subheader("변수별 유지 및 삭제 근거")

    if ORIGINAL_FILE.exists():

        original = load_csv(str(ORIGINAL_FILE))

        records = []

        for i, col in enumerate(original.columns, start=1):
            records.append({
                "번호": i,
                "변수명": col,
                "처리": "유지" if i in KEEP else "제외",
                "근거": reason(i)
            })

        report = pd.DataFrame(records)

        selected = st.selectbox(
            "변수 구분",
            ["전체", "유지", "제외"]
        )

        if selected != "전체":
            report = report[report["처리"] == selected]

        st.dataframe(
            report,
            hide_index=True,
            width="stretch"
        )

        st.subheader("원본과 전처리 결과 비교")

        expected = original.iloc[
            :, [i - 1 for i in KEEP]
        ].reset_index(drop=True)

        actual = df.reset_index(drop=True)

        same_columns = (
            expected.columns.tolist() == actual.columns.tolist()
        )

        same_values = (
            expected.astype("string").fillna("__NA__")
            .equals(
                actual.astype("string").fillna("__NA__")
            )
        )

        if same_columns and same_values:
            st.success(
                "선택한 열의 이름, 순서 및 모든 셀 값이 "
                "원본 데이터와 일치합니다."
            )
        else:
            st.warning("원본과 전처리 데이터 사이에 차이가 있습니다.")

    else:
        st.warning(
            "원본 46열 CSV를 찾지 못했습니다. "
            "data 폴더의 원본 파일 이름을 확인해 주세요."
        )

    st.info(
        "변수를 제외한 이유는 탐구 주제에 맞게 분석 범위를 "
        "선정한 것이며, 실제 예측력이 낮다고 검증한 것은 아닙니다."
    )


# =========================================
# 메뉴 3. 데이터 분포 시각화
# =========================================

elif menu == "③ 데이터 분포 시각화":

    st.header("입력 변수(X)의 데이터 분포")

    st.write(
        "생성형 AI 이용 여부를 예측하기 전에 "
        "디지털 매체 이용 경험의 분포를 확인한다."
    )

    rates = (X_media.mean() * 100).sort_values(
        ascending=False
    )

    rate_df = pd.DataFrame({
        "매체": [short_name(c) for c in rates.index],
        "이용률": rates.values
    })

    st.subheader("매체별 이용 경험 비율")

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

    stacked_df = pd.DataFrame(rows)

    stacked_chart = alt.Chart(stacked_df).mark_bar().encode(
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

    st.altair_chart(stacked_chart, width="stretch")

    if school is not None:

        st.subheader("학교급별 응답자 분포")

        school_data = pd.to_numeric(
            df[school],
            errors="coerce"
        ).map(SCHOOL_LABELS)

        school_counts = school_data.value_counts().reindex(
            list(SCHOOL_LABELS.values()),
            fill_value=0
        )

        st.bar_chart(school_counts)

    if grade is not None:

        st.subheader("학년별 응답자 분포")

        grade_data = pd.to_numeric(
            df[grade],
            errors="coerce"
        ).map(GRADE_LABELS)

        grade_counts = grade_data.value_counts().reindex(
            list(GRADE_LABELS.values()),
            fill_value=0
        )

        st.bar_chart(grade_counts)

        st.caption(
            "원본 학년 코드 1~9는 코드북에 따라 "
            "초4, 초5, 초6, 중1, 중2, 중3, 고1, 고2, 고3으로 표시합니다."
        )

    st.info(
        "이용률이 높거나 낮다는 이유만으로 변수를 제외하지 않습니다. "
        "실제 분류 성능에 미치는 영향은 이후 모델 평가에서 확인합니다."
    )


# =========================================
# 메뉴 4. 입력 변수(X)와 타깃(Y)의 관계
# =========================================

elif menu == "④ 입력 변수와 타깃의 관계":

    st.header("디지털 매체 이용과 생성형 AI 이용 여부의 관계")

    st.write(
        "각 디지털 매체 이용 경험이 생성형 AI 이용 여부와 "
        "어느 정도 관련되어 있는지 분석합니다."
    )

    st.info(
        "입력 변수(X): OTT, 숏폼, SNS 등 매체 이용 경험\n\n"
        "타깃 변수(Y): 생성형 AI 이용 여부\n\n"
        "분석용 복사본에서 이용함=1, 이용 안함=0으로 변환했습니다."
    )

    # -------------------------------------
    # 4-1. 입력 변수와 타깃 상관계수
    # -------------------------------------

    st.subheader("① 입력 변수와 타깃의 상관관계")

    correlations = []

    for col in media:
        value = X_media[col].corr(Y)

        correlations.append({
            "매체": short_name(col),
            "상관계수": value,
            "관련성 크기": abs(value)
        })

    corr_df = pd.DataFrame(correlations)

    corr_df = corr_df.sort_values(
        "관련성 크기",
        ascending=False
    )

    chart = alt.Chart(corr_df).mark_bar().encode(
        x=alt.X(
            "상관계수:Q",
            scale=alt.Scale(domain=[-1, 1]),
            title="상관계수 (-1 ~ 1)"
        ),
        y=alt.Y(
            "매체:N",
            sort=list(corr_df["매체"]),
            title=None
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
    ).properties(height=400)

    st.altair_chart(chart, width="stretch")

    st.dataframe(
        corr_df.drop(columns="관련성 크기").round(3),
        hide_index=True,
        width="stretch"
    )

    st.write(
        "**해석 방법:** 상관계수가 양수이면 해당 매체를 "
        "이용하는 학생이 생성형 AI도 이용하는 경향이 있음을 뜻합니다. "
        "음수이면 반대 경향이며, 0에 가까울수록 관련성이 약합니다."
    )

    if not corr_df.empty:

        strongest = corr_df.iloc[0]

        st.info(
            f"현재 선택한 매체 가운데 생성형 AI 이용 여부와 "
            f"상관계수의 절댓값이 가장 큰 변수는 "
            f"'{strongest['매체']}'입니다. "
            f"상관계수는 {strongest['상관계수']:.3f}입니다."
        )

    # -------------------------------------
    # 4-2. AI 이용 집단별 매체 이용률
    # -------------------------------------

    st.subheader("② 생성형 AI 이용 집단별 매체 이용률 비교")

    st.write(
        "생성형 AI를 이용한 학생과 이용하지 않은 학생이 "
        "다른 매체를 얼마나 이용하는지 비교합니다."
    )

    comparison = []

    for col in media:

        used = X_media.loc[Y == 1, col].mean() * 100
        unused = X_media.loc[Y == 0, col].mean() * 100

        comparison.append({
            "매체": short_name(col),
            "AI 이용자 이용률": used,
            "AI 미이용자 이용률": unused,
            "이용률 차이": used - unused
        })

    compare_df = pd.DataFrame(comparison)

    compare_df = compare_df.sort_values(
        "이용률 차이",
        key=lambda s: s.abs(),
        ascending=False
    )

    display_rows = []

    for _, row in compare_df.iterrows():

        display_rows.append({
            "매체": row["매체"],
            "집단": "AI 이용자",
            "이용률": row["AI 이용자 이용률"]
        })

        display_rows.append({
            "매체": row["매체"],
            "집단": "AI 미이용자",
            "이용률": row["AI 미이용자 이용률"]
        })

    group_df = pd.DataFrame(display_rows)

    group_chart = alt.Chart(group_df).mark_bar().encode(
        x=alt.X(
            "이용률:Q",
            scale=alt.Scale(domain=[0, 100]),
            title="매체 이용률(%)"
        ),
        y=alt.Y(
            "매체:N",
            sort=list(compare_df["매체"]),
            title=None
        ),
        color="집단:N",
        yOffset="집단:N",
        tooltip=[
            "매체",
            "집단",
            alt.Tooltip("이용률:Q", format=".2f")
        ]
    ).properties(height=450)

    st.altair_chart(group_chart, width="stretch")

    st.subheader("집단별 이용률 차이")

    st.dataframe(
        compare_df.round(2),
        hide_index=True,
        width="stretch"
    )

    st.caption(
        "이용률 차이는 AI 이용자의 매체 이용률에서 "
        "AI 미이용자의 매체 이용률을 뺀 값이며, 단위는 %p입니다."
    )

    # -------------------------------------
    # 4-3. 학교급과 타깃의 관계
    # -------------------------------------

    if school is not None:

        st.subheader("③ 학교급별 생성형 AI 이용률")

        school_values = pd.to_numeric(
            df[school],
            errors="coerce"
        ).map(SCHOOL_LABELS)

        school_df = pd.DataFrame({
            "학교급": school_values,
            "AI 이용함": Y
        })

        school_rate = (
            school_df.groupby("학교급")["AI 이용함"]
            .agg(["mean", "count"])
            .reset_index()
        )

        school_rate["AI 이용률"] = (
            school_rate["mean"] * 100
        )

        school_rate["학교급"] = pd.Categorical(
            school_rate["학교급"],
            categories=list(SCHOOL_LABELS.values()),
            ordered=True
        )

        school_rate = school_rate.sort_values("학교급")

        school_chart = alt.Chart(school_rate).mark_bar().encode(
            x=alt.X("학교급:N", sort=list(SCHOOL_LABELS.values())),
            y=alt.Y(
                "AI 이용률:Q",
                scale=alt.Scale(domain=[0, 100]),
                title="생성형 AI 이용률(%)"
            ),
            tooltip=[
                "학교급",
                alt.Tooltip("AI 이용률:Q", format=".2f"),
                alt.Tooltip("count:Q", title="응답자 수")
            ]
        ).properties(height=320)

        st.altair_chart(school_chart, width="stretch")

    # -------------------------------------
    # 4-4. 학년과 타깃의 관계
    # -------------------------------------

    if grade is not None:

        st.subheader("④ 학년별 생성형 AI 이용률")

        grade_values = pd.to_numeric(
            df[grade],
            errors="coerce"
        ).map(GRADE_LABELS)

        grade_df = pd.DataFrame({
            "학년": grade_values,
            "AI 이용함": Y
        })

        grade_rate = (
            grade_df.groupby("학년")["AI 이용함"]
            .mean()
            .reindex(list(GRADE_LABELS.values()))
            .reset_index(name="AI 이용률")
        )

        grade_rate["AI 이용률"] *= 100

        grade_chart = alt.Chart(grade_rate).mark_line(
            point=True
        ).encode(
            x=alt.X(
                "학년:N",
                sort=list(GRADE_LABELS.values())
            ),
            y=alt.Y(
                "AI 이용률:Q",
                scale=alt.Scale(domain=[0, 100]),
                title="생성형 AI 이용률(%)"
            ),
            tooltip=[
                "학년",
                alt.Tooltip("AI 이용률:Q", format=".2f")
            ]
        ).properties(height=350)

        st.altair_chart(grade_chart, width="stretch")

        st.caption(
            "학년 코드는 초4부터 고3까지의 순서이며, "
            "그래프의 연결선은 학년별 비율을 보기 쉽게 나타낸 것입니다."
        )

    # -------------------------------------
    # 4-5. 분석 결과 해석
    # -------------------------------------

    st.subheader("⑤ 분석 결과와 전처리 판단")

    st.write(
        "이번 분석은 각 입력 변수가 생성형 AI 이용 여부와 "
        "어떤 관련이 있는지 탐색하는 과정입니다."
    )

    st.write(
        "상관관계가 크거나 두 집단의 이용률 차이가 큰 변수는 "
        "분류 모델에서 활용 가능성이 있으므로 우선 유지합니다."
    )

    st.write(
        "다만 상관관계가 작다는 이유만으로 변수를 바로 삭제하지는 않습니다. "
        "여러 변수를 함께 사용했을 때 분류 성능이 달라질 수 있기 때문입니다."
    )

    st.info(
        "이 분석은 변수 사이의 관련성을 보여주지만 "
        "원인과 결과를 증명하거나 분류 성능을 보장하지는 않습니다. "
        "최종 변수 선택은 이후 분류 모델 3개의 성능을 비교하여 판단합니다."
    )
