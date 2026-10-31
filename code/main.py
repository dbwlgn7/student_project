
from pathlib import Path
import re
import pandas as pd
import altair as alt
import streamlit as st


# ==========================================
# 1. 기본 설정
# ==========================================

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

MEDIA_NAMES = {
    3: "TV",
    4: "OTT",
    5: "인터넷 개인방송",
    6: "숏폼",
    7: "포털사이트",
    9: "SNS",
    11: "메타버스",
    13: "온라인 게임",
    14: "웹툰"
}

MEDIA_NUMBERS = list(MEDIA_NAMES.keys())


# ==========================================
# 2. 데이터 처리 함수
# ==========================================

@st.cache_data
def load_csv(path):
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False
    )


def get_reason(number):
    if number == 1:
        return (
            "응답자를 구분하기 위한 식별자이다. "
            "중복 검사에 사용하지만 모델 입력에서는 제외한다."
        )

    if number == 12:
        return (
            "생성형 AI 이용 여부는 분류 모델이 예측할 "
            "타깃 변수(Y)이므로 유지한다."
        )

    if number in [36, 40]:
        return (
            "학교급과 학년은 청소년의 발달 단계에 따른 "
            "AI 이용 차이를 확인하기 위해 유지한다."
        )

    if number in KEEP:
        return (
            "디지털 매체 이용 경험을 나타내는 "
            "입력 변수(X)이므로 유지한다."
        )

    if number == 2:
        return "학교 식별 번호로 이용 행동을 직접 나타내지 않아 제외한다."

    if number in [8, 10, 15, 16]:
        return (
            "디지털 매체와 관련된 변수이지만 "
            "이번 탐구의 입력 변수 범위를 제한하기 위해 제외한다."
        )

    if number in [17, 18]:
        return (
            "종이 매체 이용 경험이므로 "
            "디지털 매체 중심의 분석에서 제외한다."
        )

    if number in [19, 20, 21]:
        return (
            "자주 이용하는 매체 응답에 생성형 AI가 포함될 수 있다. "
            "타깃 정보 유출 가능성을 줄이기 위해 제외한다."
        )

    if number in list(range(22, 30)):
        return (
            "온라인 안전 및 개인정보 보호 관련 경험으로 "
            "매체 이용 여부 자체를 나타내지 않아 제외한다."
        )

    if number in [30, 31, 32, 33]:
        return (
            "교육 경험 및 효과 관련 변수이므로 "
            "이용 경험 중심의 분석 범위에서 제외한다."
        )

    if number in [34, 35]:
        return (
            "개인 또는 학교 특성으로 "
            "선정한 분석 범위에서 제외한다."
        )

    if number in [37, 38, 39, 43, 44]:
        return (
            "학교급 및 학년과 관련된 추가 정보로, "
            "이번 탐구에서는 기본 학교급과 학년을 사용한다."
        )

    if number in [41, 42]:
        return "지역 관련 변수로 이번 탐구의 범위에서 제외한다."

    if number in [45, 46]:
        return (
            "설문조사 가중치로 매체 이용 경험을 "
            "직접 나타내지 않아 입력에서 제외한다."
        )

    return "탐구 범위를 고려하여 제외한다."


def show_chart(chart):
    st.altair_chart(chart, width="stretch")


def classify_strength(value):
    if pd.isna(value):
        return "계산할 수 없음"
    v = abs(value)
    if v < 0.1:
        return "매우 약한 관계"
    if v < 0.3:
        return "약한 관계"
    if v < 0.5:
        return "중간 정도의 관계"
    return "비교적 강한 관계"


def find_column(columns, keyword):
    return next(
        (c for c in columns if keyword in str(c)),
        None
    )


# ==========================================
# 3. 사이드바
# ==========================================

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
    st.error(
        "13열 CSV 파일을 찾지 못했습니다. "
        "data 폴더의 파일 이름을 확인하세요."
    )
    st.stop()

# 원본에서 선택한 열의 순서를 기준으로 변수 식별
if df.shape[1] != len(KEEP):
    st.error(
        f"13열 데이터가 필요하지만 현재 {df.shape[1]}열입니다. "
        "올바른 CSV를 사용해 주세요."
    )
    st.stop()

columns = list(df.columns)
position = {number: index for index, number in enumerate(KEEP)}

id_col = columns[position[1]]
target = columns[position[12]]
school = columns[position[36]]
grade = columns[position[40]]

media_cols = {
    number: columns[position[number]]
    for number in MEDIA_NUMBERS
}

# 분석용 데이터 생성
# 원본 CSV의 숫자 값과 셀은 변경하지 않음

X_media = pd.DataFrame(index=df.index)

for number, col in media_cols.items():
    X_media[MEDIA_NAMES[number]] = pd.to_numeric(
        df[col], errors="coerce"
    ).map({1: 1, 2: 0})

Y = pd.to_numeric(
    df[target], errors="coerce"
).map({1: 1, 2: 0})

school_data = pd.to_numeric(
    df[school], errors="coerce"
).map(SCHOOL_LABELS)

grade_data = pd.to_numeric(
    df[grade], errors="coerce"
).map(GRADE_LABELS)

if X_media.isna().any().any() or Y.isna().any():
    st.error(
        "매체 또는 타깃 변수에 1·2 이외의 코드나 "
        "결측치가 있어 분석을 중단했습니다."
    )
    st.stop()


# ==========================================
# 5. 페이지 상단
# ==========================================

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

    st.write(
        "청소년의 디지털 매체 이용 특성을 분석하여 "
        "생성형 AI 이용 여부를 분류할 수 있는지 알아본다."
    )

    a, b, c = st.columns(3)

    a.metric("응답자 수", f"{len(df):,}명")
    b.metric("선택 변수", len(df.columns))
    c.metric("결측치", int(df.isna().sum().sum()))

    st.subheader("탐구 질문")

    st.write(
        "**청소년의 디지털 매체 이용 경험과 학교급, 학년을 "
        "활용하면 생성형 AI 이용 여부를 분류할 수 있을까?**"
    )

    st.subheader("데이터의 속성과 유형")

    types = pd.DataFrame([
        {
            "속성": "응답자 ID",
            "데이터 유형": "식별자",
            "분석 역할": "중복 검사",
            "모델 입력": "제외"
        },
        {
            "속성": "9개 디지털 매체 이용 경험",
            "데이터 유형": "이진 범주형",
            "분석 역할": "입력 변수 X",
            "모델 입력": "포함"
        },
        {
            "속성": "학교급",
            "데이터 유형": "순서형 범주",
            "분석 역할": "입력 변수 X",
            "모델 입력": "포함"
        },
        {
            "속성": "학년",
            "데이터 유형": "순서형 범주",
            "분석 역할": "입력 변수 X",
            "모델 입력": "포함"
        },
        {
            "속성": "생성형 AI 이용 여부",
            "데이터 유형": "이진 범주형",
            "분석 역할": "타깃 변수 Y",
            "모델 입력": "제외"
        }
    ])

    st.dataframe(types, hide_index=True, width="stretch")

    st.info(
        "설문 응답의 숫자 1과 2는 수량이 아닙니다. "
        "이용함과 이용 안함을 나타내는 범주형 코드입니다."
    )

    st.subheader("시각화 방법 선정")

    methods = pd.DataFrame([
        {
            "분석 대상": "매체별 이용 경험",
            "시각화 방법": "가로 막대그래프",
            "선택 이유": "매체별 비율과 순위 비교에 적합"
        },
        {
            "분석 대상": "AI 이용 여부",
            "시각화 방법": "막대그래프",
            "선택 이유": "두 집단의 규모 비교에 적합"
        },
        {
            "분석 대상": "매체 X와 AI 이용 Y",
            "시각화 방법": "상관계수 막대그래프",
            "선택 이유": "관계의 방향과 크기 비교"
        },
        {
            "분석 대상": "AI 이용 집단별 매체 경험",
            "시각화 방법": "그룹 막대그래프",
            "선택 이유": "두 집단의 매체 이용률 차이 비교"
        },
        {
            "분석 대상": "학교급 및 학년",
            "시각화 방법": "막대그래프",
            "선택 이유": "집단별 응답 수와 AI 이용률 비교"
        }
    ])

    st.dataframe(methods, hide_index=True, width="stretch")

    st.subheader("탐구 진행 과정")

    st.write(
        "① 변수 선택과 전처리 → "
        "② 데이터 분포 확인 → "
        "③ 입력 변수와 타깃의 관계 분석 → "
        "④ 분류 모델 3개 비교(추후 진행)"
    )


# ==========================================
# 메뉴 2. 전처리 및 근거
# ==========================================

elif menu == "② 전처리 및 근거":

    st.header("전처리 과정과 변수 선택 근거")

    a, b, c = st.columns(3)

    a.metric("원본 변수", 46)
    b.metric("유지 변수", len(KEEP))
    c.metric("제외 변수", 46 - len(KEEP))

    st.write(
        "원본 46개 변수 중 탐구 주제에 필요한 "
        "13개를 선택하였다."
    )

    st.subheader("데이터 품질 확인")

    a, b, c = st.columns(3)

    a.metric("응답자 수", f"{len(df):,}")
    b.metric("결측치 수", int(df.isna().sum().sum()))
    c.metric("중복 ID 수", int(df[id_col].duplicated().sum()))

    if ORIGINAL_FILE.exists():

        original = load_csv(str(ORIGINAL_FILE))

        rows = []

        for i, col in enumerate(original.columns, start=1):
            rows.append({
                "원본 열 번호": i,
                "변수명": col,
                "처리": "유지" if i in KEEP else "제외",
                "이유": get_reason(i)
            })

        report = pd.DataFrame(rows)

        selection = st.selectbox(
            "변수 확인",
            ["전체", "유지", "제외"]
        )

        if selection != "전체":
            report = report[
                report["처리"] == selection
            ]

        st.dataframe(
            report,
            hide_index=True,
            width="stretch"
        )

        st.subheader("원본과 선택 데이터 비교")

        expected = original.iloc[
            :, [i - 1 for i in KEEP]
        ].reset_index(drop=True)

        actual = df.reset_index(drop=True)

        same_names = (
            expected.columns.tolist()
            == actual.columns.tolist()
        )

        same_values = (
            expected.astype("string").fillna("__NA__")
            .equals(
                actual.astype("string").fillna("__NA__")
            )
        )

        if same_names and same_values:
            st.success(
                "선택한 열의 이름, 순서와 응답 값이 "
                "원본 데이터와 일치합니다."
            )
        else:
            st.warning(
                "원본 데이터와 일부 값이 일치하지 않습니다. "
                "업로드한 CSV와 원본 파일을 확인하세요."
            )

    else:
        st.warning("원본 46열 CSV를 찾지 못했습니다.")

    st.subheader("전처리 해석")

    st.write(
        "응답자 ID는 분석 대상을 구분하는 데만 사용한다. "
        "생성형 AI 이용 여부는 정답(Y)이므로 입력 변수(X)에서 제외한다."
    )

    st.write(
        "디지털 매체 이용 경험은 숫자 1과 2로 표현되어 있지만 "
        "실제로는 범주형 데이터이다. "
        "따라서 분석용 복사본에서만 이용함=1, "
        "이용 안함=0으로 변환한다."
    )

    st.info(
        "변수 선택은 탐구 범위를 설정하기 위한 것이며, "
        "제외한 변수가 예측력이 없다는 뜻은 아닙니다."
    )


# ==========================================
# 메뉴 3. 데이터 분포 시각화
# ==========================================

elif menu == "③ 데이터 분포 시각화":

    st.header("데이터 속성별 분포 분석")

    st.write(
        "입력 변수와 타깃 변수가 어떻게 분포하는지 "
        "확인하고 시각화 결과를 해석한다."
    )

    # --------------------------------------
    # 3-1. 매체별 이용률
    # --------------------------------------

    st.subheader("① 디지털 매체별 이용률")

    st.caption(
        "데이터 유형: 이진 범주형 / "
        "시각화: 가로 막대그래프 / "
        "선택 이유: 매체별 이용률을 쉽게 비교할 수 있음"
    )

    rates = (
        X_media.mean() * 100
    ).sort_values(ascending=False)

    rate_df = pd.DataFrame({
        "매체": rates.index,
        "이용률": rates.values
    })

    chart = alt.Chart(rate_df).mark_bar().encode(
        x=alt.X(
            "이용률:Q",
            scale=alt.Scale(domain=[0, 100]),
            title="이용률(%)"
        ),
        y=alt.Y(
            "매체:N",
            sort="-x",
            title=None
        ),
        tooltip=[
            "매체",
            alt.Tooltip("이용률:Q", format=".2f")
        ]
    ).properties(height=370)

    show_chart(chart)

    highest = rate_df.iloc[0]
    lowest = rate_df.iloc[-1]

    st.write(
        f"**분석 결과:** 가장 이용률이 높은 매체는 "
        f"{highest['매체']}({highest['이용률']:.1f}%)이고, "
        f"가장 낮은 매체는 "
        f"{lowest['매체']}({lowest['이용률']:.1f}%)이다."
    )

    st.write(
        "이 결과를 통해 매체별 이용 경험의 분포 차이를 "
        "확인할 수 있다. 다만 이용률이 높다고 "
        "생성형 AI 이용 여부를 더 잘 분류한다고 볼 수는 없다."
    )

    # --------------------------------------
    # 3-2. AI 이용 여부
    # --------------------------------------

    st.subheader("② 생성형 AI 이용 여부 분포")

    st.caption(
        "데이터 유형: 이진 범주형 / "
        "시각화: 막대그래프 / "
        "선택 이유: 이용자와 미이용자의 규모 비교"
    )

    yes_count = int((Y == 1).sum())
    no_count = int((Y == 0).sum())

    target_df = pd.DataFrame({
        "이용 여부": ["이용함", "이용 안함"],
        "응답자 수": [yes_count, no_count]
    })

    target_chart = alt.Chart(target_df).mark_bar().encode(
        x=alt.X(
            "이용 여부:N",
            sort=["이용함", "이용 안함"]
        ),
        y="응답자 수:Q",
        color="이용 여부:N",
        tooltip=["이용 여부", "응답자 수"]
    ).properties(height=280)

    show_chart(target_chart)

    st.write(
        f"**분석 결과:** 생성형 AI 이용자는 "
        f"{yes_count:,}명({yes_count / len(Y) * 100:.1f}%), "
        f"미이용자는 "
        f"{no_count:,}명({no_count / len(Y) * 100:.1f}%)이다."
    )

    difference = abs(yes_count - no_count) / len(Y) * 100

    st.write(
        f"두 집단의 비율 차이는 {difference:.1f}%p이다. "
        "이 비율은 이후 분류 모델의 평가 기준을 "
        "정할 때 참고할 수 있다."
    )

    # --------------------------------------
    # 3-3. 학교급 분포
    # --------------------------------------

    st.subheader("③ 학교급별 응답자 수")

    st.caption(
        "데이터 유형: 순서형 범주 / "
        "시각화: 막대그래프 / "
        "선택 이유: 학교급별 응답 규모 비교"
    )

    school_counts = school_data.value_counts().reindex(
        list(SCHOOL_LABELS.values()),
        fill_value=0
    )

    st.bar_chart(school_counts)

    st.write(
        "**분석 결과:** 학교급별 응답자 수가 "
        "서로 다를 수 있으므로, "
        "이후 AI 이용 여부를 비교할 때는 "
        "응답자 수보다 집단별 이용률을 살펴보는 것이 적절하다."
    )

    # --------------------------------------
    # 3-4. 학년 분포
    # --------------------------------------

    st.subheader("④ 학년별 응답자 수")

    st.caption(
        "데이터 유형: 순서형 범주 / "
        "시각화: 막대그래프 / "
        "선택 이유: 학년별 응답 규모 비교"
    )

    grade_counts = grade_data.value_counts().reindex(
        list(GRADE_LABELS.values()),
        fill_value=0
    )

    st.bar_chart(grade_counts)

    st.caption(
        "코드 1~9는 초4, 초5, 초6, 중1, 중2, 중3, "
        "고1, 고2, 고3 순서로 표시한다."
    )

    st.info(
        "이 메뉴에서는 데이터 자체의 분포를 확인했다. "
        "다음 메뉴에서는 어떤 입력 변수가 "
        "생성형 AI 이용 여부와 관련이 있는지 분석한다."
    )


# ==========================================
# 메뉴 4. X와 Y의 관계 분석
# ==========================================

elif menu == "④ 입력 변수와 타깃의 관계":

    st.header("입력 변수(X)와 타깃 변수(Y)의 관계")

    st.write(
        "청소년의 디지털 매체 이용 경험과 "
        "생성형 AI 이용 여부 사이의 관계를 분석한다."
    )

    st.info(
        "입력 변수(X): 디지털 매체 이용 경험, 학교급, 학년\n\n"
        "타깃 변수(Y): 생성형 AI 이용 여부\n\n"
        "관계 분석에는 이용함=1, 이용 안함=0으로 "
        "변환한 분석용 데이터를 사용한다."
    )

    # --------------------------------------
    # 4-1. X와 Y의 상관관계
    # --------------------------------------

    st.subheader("① 매체별 생성형 AI 이용 여부와의 상관관계")

    st.caption(
        "데이터 유형: 이진 범주형 X와 이진 범주형 Y / "
        "시각화: 상관계수 막대그래프 / "
        "선택 이유: 관계의 방향과 크기를 비교할 수 있음"
    )

    correlations = []

    for col in X_media.columns:

        value = X_media[col].corr(Y)

        correlations.append({
            "매체": col,
            "상관계수": value,
            "절댓값": abs(value)
        })

    corr_df = pd.DataFrame(correlations)

    corr_df = corr_df.sort_values(
        "절댓값",
        ascending=False
    )

    corr_chart = alt.Chart(corr_df).mark_bar().encode(
        x=alt.X(
            "상관계수:Q",
            scale=alt.Scale(domain=[-1, 1]),
            title="상관계수"
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
    ).properties(height=370)

    show_chart(corr_chart)

    st.dataframe(
        corr_df.drop(columns="절댓값").round(3),
        hide_index=True,
        width="stretch"
    )

    strongest = corr_df.dropna(
        subset=["상관계수"]
    )

    if not strongest.empty:

        strongest = strongest.iloc[0]

        direction = (
            "양의 관계"
            if strongest["상관계수"] > 0
            else "음의 관계"
            if strongest["상관계수"] < 0
            else "상관관계가 거의 없는 상태"
        )

        strength = classify_strength(
            strongest["상관계수"]
        )

        st.write(
            f"**분석 결과:** 분석한 매체 중 "
            f"생성형 AI 이용 여부와 상관계수의 "
            f"절댓값이 가장 큰 매체는 "
            f"**{strongest['매체']}**이다. "
            f"상관계수는 {strongest['상관계수']:.3f}으로 "
            f"{direction}를 나타내며, "
            f"설명용 기준으로 {strength}에 해당한다."
        )

    st.write(
        "상관계수가 양수이면 해당 매체 이용과 "
        "생성형 AI 이용이 함께 나타나는 경향이 있고, "
        "음수이면 반대 경향이 있다."
    )

    st.caption(
        "상관계수의 강도 구분은 이해를 돕기 위한 "
        "참고 기준이며, 인과관계나 예측 성능을 뜻하지 않는다."
    )

    # --------------------------------------
    # 4-2. 두 집단의 매체 이용률 비교
    # --------------------------------------

    st.subheader("② AI 이용자와 미이용자의 매체 이용률 비교")

    st.caption(
        "데이터 유형: 두 범주형 변수 / "
        "시각화: 그룹 막대그래프 / "
        "선택 이유: 집단별 매체 이용 경험의 차이를 직접 비교"
    )

    rows = []
    differences = []

    for col in X_media.columns:

        ai_yes = X_media.loc[Y == 1, col].mean() * 100
        ai_no = X_media.loc[Y == 0, col].mean() * 100

        rows.append({
            "매체": col,
            "집단": "AI 이용자",
            "매체 이용률": ai_yes
        })

        rows.append({
            "매체": col,
            "집단": "AI 미이용자",
            "매체 이용률": ai_no
        })

        differences.append({
            "매체": col,
            "AI 이용자 이용률": ai_yes,
            "AI 미이용자 이용률": ai_no,
            "차이(%p)": ai_yes - ai_no
        })

    grouped_df = pd.DataFrame(rows)
    diff_df = pd.DataFrame(differences)

    diff_df["차이 절댓값"] = diff_df["차이(%p)"].abs()

    diff_df = diff_df.sort_values(
        "차이 절댓값",
        ascending=False
    )

    group_chart = alt.Chart(grouped_df).mark_bar().encode(
        x=alt.X(
            "매체 이용률:Q",
            scale=alt.Scale(domain=[0, 100]),
            title="매체 이용률(%)"
        ),
        y=alt.Y(
            "매체:N",
            sort=list(diff_df["매체"]),
            title=None
        ),
        yOffset="집단:N",
        color="집단:N",
        tooltip=[
            "매체",
            "집단",
            alt.Tooltip("매체 이용률:Q", format=".2f")
        ]
    ).properties(height=430)

    show_chart(group_chart)

    st.dataframe(
        diff_df.drop(
            columns="차이 절댓값"
        ).round(2),
        hide_index=True,
        width="stretch"
    )

    top_diff = diff_df.iloc[0]

    if top_diff["차이(%p)"] > 0:
        comparison_text = "AI 이용자 집단에서 더 높았다"
    elif top_diff["차이(%p)"] < 0:
        comparison_text = "AI 미이용자 집단에서 더 높았다"
    else:
        comparison_text = "두 집단에서 같았다"

    st.write(
        f"**분석 결과:** 두 집단의 매체 이용률 차이가 "
        f"가장 크게 나타난 매체는 "
        f"**{top_diff['매체']}**이다. "
        f"차이는 {abs(top_diff['차이(%p)']):.1f}%p였으며, "
        f"이용률은 {comparison_text}."
    )

    st.write(
        "이러한 집단 간 차이는 해당 매체의 이용 경험이 "
        "생성형 AI 이용 여부와 관련되어 있을 가능성을 "
        "보여준다. 하지만 이 결과만으로 "
        "분류 정확도를 알 수는 없다."
    )

    # --------------------------------------
    # 4-3. 학교급별 AI 이용률
    # --------------------------------------

    st.subheader("③ 학교급과 생성형 AI 이용 여부의 관계")

    st.caption(
        "데이터 유형: 순서형 범주 X와 이진 범주형 Y / "
        "시각화: 막대그래프 / "
        "선택 이유: 학교급별 AI 이용률 비교"
    )

    school_df = pd.DataFrame({
        "학교급": school_data,
        "AI 이용": Y
    })

    school_rate = (
        school_df.groupby("학교급")["AI 이용"]
        .agg(["mean", "count"])
        .reset_index()
    )

    school_rate["AI 이용률"] = school_rate["mean"] * 100

    school_chart = alt.Chart(school_rate).mark_bar().encode(
        x=alt.X(
            "학교급:N",
            sort=list(SCHOOL_LABELS.values())
        ),
        y=alt.Y(
            "AI 이용률:Q",
            scale=alt.Scale(domain=[0, 100]),
            title="AI 이용률(%)"
        ),
        tooltip=[
            "학교급",
            alt.Tooltip("AI 이용률:Q", format=".2f"),
            alt.Tooltip("count:Q", title="응답자 수")
        ]
    ).properties(height=300)

    show_chart(school_chart)

    if not school_rate.empty:

        max_school = school_rate.loc[
            school_rate["AI 이용률"].idxmax()
        ]

        min_school = school_rate.loc[
            school_rate["AI 이용률"].idxmin()
        ]

        st.write(
            f"**분석 결과:** AI 이용률이 가장 높은 학교급은 "
            f"{max_school['학교급']}"
            f"({max_school['AI 이용률']:.1f}%)이고, "
            f"가장 낮은 학교급은 "
            f"{min_school['학교급']}"
            f"({min_school['AI 이용률']:.1f}%)이다."
        )

    # --------------------------------------
    # 4-4. 학년별 AI 이용률
    # --------------------------------------

    st.subheader("④ 학년과 생성형 AI 이용 여부의 관계")

    st.caption(
        "데이터 유형: 순서형 범주 X와 이진 범주형 Y / "
        "시각화: 막대그래프 / "
        "선택 이유: 학년별 AI 이용률을 순서대로 비교"
    )

    grade_df = pd.DataFrame({
        "학년": grade_data,
        "AI 이용": Y
    })

    grade_rate = (
        grade_df.groupby("학년")["AI 이용"]
        .agg(["mean", "count"])
        .reindex(list(GRADE_LABELS.values()))
        .reset_index()
    )

    grade_rate["AI 이용률"] = grade_rate["mean"] * 100

    grade_chart = alt.Chart(grade_rate).mark_bar().encode(
        x=alt.X(
            "학년:N",
            sort=list(GRADE_LABELS.values())
        ),
        y=alt.Y(
            "AI 이용률:Q",
            scale=alt.Scale(domain=[0, 100]),
            title="AI 이용률(%)"
        ),
        tooltip=[
            "학년",
            alt.Tooltip("AI 이용률:Q", format=".2f"),
            alt.Tooltip("count:Q", title="응답자 수")
        ]
    ).properties(height=310)

    show_chart(grade_chart)

    valid_grade = grade_rate.dropna(
        subset=["AI 이용률"]
    )

    if not valid_grade.empty:

        high_grade = valid_grade.loc[
            valid_grade["AI 이용률"].idxmax()
        ]

        low_grade = valid_grade.loc[
            valid_grade["AI 이용률"].idxmin()
        ]

        st.write(
            f"**분석 결과:** 학년별 AI 이용률은 "
            f"{high_grade['학년']}에서 "
            f"{high_grade['AI 이용률']:.1f}%로 가장 높고, "
            f"{low_grade['학년']}에서 "
            f"{low_grade['AI 이용률']:.1f}%로 가장 낮다."
        )

    st.caption(
        "학년은 초4~고3으로 표시하며, "
        "단순히 코드 1~9를 1~9학년으로 해석하지 않는다."
    )

    # --------------------------------------
    # 4-5. 종합 해석
    # --------------------------------------

    st.subheader("⑤ 시각화를 통해 도출한 종합 해석")

    st.write(
        "이번 분석에서는 청소년의 디지털 매체 이용 경험과 "
        "학교급, 학년이 생성형 AI 이용 여부와 "
        "어떤 관련성을 보이는지 확인하였다."
    )

    st.write(
        f"매체 이용 여부와 AI 이용 여부의 상관계수, "
        f"그리고 두 집단의 매체 이용률 차이를 비교한 결과, "
        f"**{top_diff['매체']}**에서 가장 큰 "
        f"집단 간 이용률 차이가 나타났다."
    )

    st.write(
        "이러한 결과는 디지털 매체 이용 특성을 "
        "생성형 AI 이용 여부의 분류에 활용할 가능성을 "
        "탐색하는 근거가 된다."
    )

    st.write(
        "다만 각 변수와 타깃 사이의 관계만으로 "
        "실제 분류 모델의 성능을 결정할 수는 없다. "
        "여러 입력 변수를 함께 사용했을 때의 예측력을 "
        "별도의 학습 및 평가 데이터로 검증해야 한다."
    )

    st.info(
        "현재 단계에서는 변수를 성급하게 삭제하지 않는다. "
        "이후 분류 알고리즘 3개를 비교하고, "
        "정확도와 F1 점수 등의 지표를 확인하여 "
        "최종 모델과 변수 구성을 판단할 예정이다."
    )
