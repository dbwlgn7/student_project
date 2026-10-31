import io
from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title='청소년 디지털 매체와 생성형 AI 이용 분석', page_icon='📊', layout='wide')
st.title('📊 청소년 디지털 매체 이용과 생성형 AI 분류 탐구')
st.caption('원본 응답 코드와 셀 값은 변경하지 않습니다. 그래프의 범례에만 코드북의 의미를 표시합니다.')

@st.cache_data
def load_csv(source):
    if isinstance(source, bytes):
        return pd.read_csv(io.BytesIO(source), encoding='utf-8-sig')
    return pd.read_csv(source, encoding='utf-8-sig')

DEFAULT_PATH = Path(__file__).parent / '청소년_AI이용분류_ID포함13열.csv'
with st.sidebar:
    st.header('데이터 불러오기')
    upload = st.file_uploader('13열 CSV 업로드', type=['csv'])
    st.caption('Streamlit Cloud에서 CSV도 같은 저장소에 넣으면 자동으로 불러옵니다.')
try:
    if upload is not None:
        df = load_csv(upload.getvalue())
    elif DEFAULT_PATH.exists():
        df = load_csv(str(DEFAULT_PATH))
    else:
        st.info('왼쪽에서 청소년_AI이용분류_ID포함13열.csv를 업로드해 주세요. 또는 CSV를 main.py와 같은 폴더에 넣으세요.')
        st.stop()
except Exception as e:
    st.error(f'CSV 읽기에 실패했습니다: {e}')
    st.stop()

media_cols = [c for c in df.columns if c.startswith('문1. 매체 이용 경험')]
target_matches = [c for c in media_cols if '생성형 AI' in c]
if len(target_matches) != 1:
    st.error('생성형 AI 이용 경험 열을 정확히 하나 찾지 못했습니다. 13열 CSV를 확인해 주세요.')
    st.stop()
target = target_matches[0]
inputs = [c for c in media_cols if c != target]
school_col = '학교급(기본)'
grade_col = '학년별'
id_col = '응답자 ID'
missing_required = [c for c in [school_col, grade_col, id_col] if c not in df.columns]
if missing_required:
    st.error('필수 열이 없습니다: ' + ', '.join(missing_required))
    st.stop()

short = lambda c: c.split(') ', 1)[-1] if ') ' in c else c
valid_codes = {1, 2}
invalid = {short(c): sorted(set(df[c].dropna().unique()) - valid_codes) for c in media_cols if set(df[c].dropna().unique()) - valid_codes}

st.subheader('1. 원본 데이터 점검')
a,b,c,d = st.columns(4)
a.metric('응답자', f'{len(df):,}명')
b.metric('변수', f'{df.shape[1]}개')
c.metric('결측치', f'{df.isna().sum().sum():,}개')
d.metric('중복 응답자 ID', f'{df[id_col].duplicated().sum():,}개')
if invalid:
    st.warning(f'코드북의 1·2 외 값이 있습니다: {invalid}. 그래프가 누락된 값을 제외할 수 있습니다.')
else:
    st.success('모든 매체 이용 문항에서 코드 1·2만 확인됐습니다.')
with st.expander('원본 데이터 미리보기 및 변수 이름'):
    st.dataframe(df.head(20), use_container_width=True)
    st.write(df.columns.tolist())

st.subheader('2. 생성형 AI 이용 여부 (타깃 Y)')
counts = df[target].value_counts().reindex([1,2], fill_value=0)
target_chart = pd.DataFrame({'응답': ['이용함','전혀 이용 안함'], '명수': counts.to_list()})
st.bar_chart(target_chart.set_index('응답'))
st.write(f"이용함 **{counts.loc[1]:,}명 ({counts.loc[1]/len(df)*100:.1f}%)**, 이용 안함 **{counts.loc[2]:,}명 ({counts.loc[2]/len(df)*100:.1f}%)**")

st.subheader('3. 매체별 이용 경험 비율')
all_rate = pd.DataFrame([{'매체':short(c), '이용률(%)':(df[c]==1).mean()*100} for c in inputs]).sort_values('이용률(%)',ascending=False)
st.bar_chart(all_rate.set_index('매체'))
st.dataframe(all_rate.round(2), hide_index=True, use_container_width=True)
st.caption('이용률은 전체 응답자 중 코드 1(이용함)의 비율입니다. 여러 매체 이용이 중복될 수 있습니다.')

st.subheader('4. AI 이용 여부에 따른 다른 매체 이용률 비교')
rows=[]
for col in inputs:
    for code,name in [(1,'AI 이용함'),(2,'AI 이용 안함')]:
        group=df[df[target]==code]
        rows.append({'매체':short(col),'AI 이용 여부':name,'매체 이용률(%)':(group[col]==1).mean()*100 if len(group) else None})
compare=pd.DataFrame(rows)
chart=alt.Chart(compare).mark_bar().encode(
    x=alt.X('매체:N',sort=all_rate['매체'].to_list(),axis=alt.Axis(labelAngle=-35)),
    xOffset='AI 이용 여부:N',
    y=alt.Y('매체 이용률(%):Q',scale=alt.Scale(domain=[0,100])),
    color='AI 이용 여부:N',
    tooltip=['매체','AI 이용 여부',alt.Tooltip('매체 이용률(%):Q',format='.1f')]
).properties(height=400)
st.altair_chart(chart, use_container_width=True)
wide=compare.pivot(index='매체',columns='AI 이용 여부',values='매체 이용률(%)')
wide['격차(%p)']=wide['AI 이용함']-wide['AI 이용 안함']
