"""청소년 디지털 매체 이용 특성 기반 생성형 AI 이용 여부 분류 대시보드."""
from pathlib import Path
import io
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

st.set_page_config(page_title="청소년 AI 이용 분류 탐구", page_icon="🧭", layout="wide")
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = "청소년_AI이용분류_ID포함13열.csv"
ORIG_FILE = "청소년_AI이용분류_정제데이터.csv"
KEEP = [1,3,4,5,6,7,9,11,12,13,14,36,40]

@st.cache_data(ttl=3600)
def read_file(path):
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

def load_file(upload, filename):
    if upload is not None:
        return pd.read_csv(io.BytesIO(upload.getvalue()), encoding="utf-8-sig", low_memory=False)
    path = ROOT / "data" / filename
    if path.exists():
        return read_file(str(path))
    return None

def show_chart(chart):
    st.altair_chart(chart, width="stretch")

def selected_columns(df):
    cols=list(df.columns)
    target = next((c for c in cols if "생성형 AI" in c and "이용 경험" in c), None)
    ident = next((c for c in cols if "응답자 ID" in c), cols[0])
    grade = next((c for c in cols if "학교급(기본)" in c), None)
    year = next((c for c in cols if "학년별" in c), None)
    media = [c for c in cols if c not in (target,ident,grade,year)]
    return target,ident,grade,year,media

def short(c):
    if " - " in c: return c.split(" - ",1)[1].split(") ",1)[-1]
    return c

def encode_binary(df, media):
    # 원본 데이터는 수정하지 않고 분석 복사본만 변환
    out=df[media].copy()
    for c in media:
        out[c]=pd.to_numeric(out[c],errors="coerce").map({1:1,2:0})
    return out

def source_reason(i):
    if i==1: return "응답자를 구분하기 위해 유지했다. 머신러닝 입력에는 사용하지 않는다."
    if i==12: return "생성형 AI 이용 여부는 분류 모델이 맞혀야 할 정답(Y)이므로 유지했다."
    if i in (36,40): return "학교급·학년별 차이도 살펴보기 위한 보조 입력 변수로 유지했다."
    if i in KEEP: return "디지털 매체 이용 경험을 직접 나타내므로 모델 입력(X)에 활용하기 위해 유지했다."
    if i==2: return "학교를 구분하는 번호로, 디지털 매체 이용 특성이 아니어서 분석용 열에서 제외했다."
    if i in (8,10,15,16): return "디지털 매체 이용과 관련이 있지만 이번 모델의 매체 종류를 일부로 한정하여 제외했다. 예측력이 없다고 검증된 것은 아니다."
    if i in (17,18): return "종이 매체 이용 경험으로, 디지털 매체 이용 특성에 집중하려는 탐구 범위에서 제외했다."
    if i in (19,20,21): return "자주 이용하는 매체 목록에 생성형 AI가 직접 포함될 수 있어 정답 누수 위험을 줄이기 위해 제외했다."
    if i==22: return "보호 프로그램 설치 여부는 이용하는 매체의 종류 자체가 아니므로 제외했다."
    if i in (23,24): return "타인의 정보 사용 경험으로, 디지털 매체의 종류별 이용 여부를 직접 나타내지 않아 제외했다."
    if i in (25,26,27,28,29): return "개인정보 노출·제공 경험으로, 매체별 이용 여부에 초점을 맞춘 분석 범위에서 제외했다."
    if i in (30,31): return "교육 경험을 나타내며 이용하는 디지털 매체의 종류를 나타내지는 않으므로 제외했다."
    if i in (32,33): return "교육 도움 정도는 매체 이용 특성이 아니고 교육 미경험자의 비해당 결측치가 있어 제외했다."
    if i==34: return "성별은 매체 이용 여부가 아닌 인구통계적 정보로, 이번 분석에서는 제외했다."
    if i==35: return "학교 환경 특성으로 디지털 매체 이용 자체가 아니어서 제외했다."
    if i==37: return "기본 학교급을 유지해 유사한 학교급 세부 변수는 제외했다."
    if i in (38,39,43,44): return "기본 학교급과 정보가 일부 겹치며 특정 학교급에 비해당하는 결측치가 있으므로 제외했다."
    if i in (41,42): return "지역 정보는 매체 이용 자체를 나타내지 않아 이번 기본 모델에서 제외했다."
    if i in (45,46): return "조사 가중치는 이용 경험이 아닌 설문 추정 정보이므로 분류 입력 변수에서 제외했다."
    return "디지털 매체 이용 특성 중심으로 탐구 범위를 정하여 제외했다."

st.sidebar.title("🧭 탐구 메뉴")
menu=st.sidebar.radio("이동할 메뉴",[
    "① 탐구 소개","② 전처리 및 근거","③ 데이터 분포 시각화",
    "④ 변수 간 관계","⑤ 분류 모델 평가","⑥ 탐구 결론"
])
upload=st.sidebar.file_uploader("13열 CSV 직접 올리기 (선택)",type="csv")
df=load_file(upload,DATA_FILE)
if df is None:
    st.error(f"data/{DATA_FILE} 파일을 찾지 못했어. GitHub의 data 폴더를 확인하거나 왼쪽에서 업로드해 줘.")
    st.stop()
target,ident,grade,year,media=selected_columns(df)
if target is None or len(media)<1:
    st.error("생성형 AI 타깃 또는 매체 이용 변수를 찾지 못했습니다. 13열 CSV의 열 이름을 확인해 주세요.")
    st.dataframe(pd.DataFrame({"열 이름":df.columns}))
    st.stop()
bad={c: sorted(set(pd.to_numeric(df[c],errors="coerce").dropna())-{1,2}) for c in media+[target]}
bad={k:v for k,v in bad.items() if v}
xmedia=encode_binary(df,media)
ytarget=pd.to_numeric(df[target],errors="coerce").map({1:1,2:0})
st.title("🧭 청소년 디지털 매체 이용과 생성형 AI 분류 탐구")
st.caption("청소년의 디지털 매체 이용 특성으로 생성형 AI 이용 여부를 분류할 수 있을까? · CSV 원본 값은 변경하지 않습니다.")

if menu.startswith("①"):
    st.header("탐구 소개")
    st.write("**탐구 질문:** 다른 디지털 매체 이용 경험과 학교급·학년을 바탕으로 청소년의 생성형 AI 이용 여부를 예측할 수 있을까?")
    a,b,c=st.columns(3)
    a.metric("응답자",f"{len(df):,}명")
    b.metric("선택한 변수",f"{len(df.columns)}개")
    c.metric("예측 대상","생성형 AI 이용 여부")
     st.info("응답자 ID는 데이터 확인에만 사용하고, 생성형 AI 이용 여부는 정답(Y)으로만 사용합니다. 모델에는 정답을 입력하지 않습니다.")
    st.write("**입력 변수(X):**", ", ".join([short(c) for c in media])+(", 학교급, 학년" if grade and year else ""))
    st.write("**분석 흐름:** 원본 비교 → 전처리 근거 → 입력 변수 분포 → 관계 분석 → 분류 모델 평가")

elif menu.startswith("②"):
    st.header("데이터 전처리 및 근거")
    original=load_file(None,ORIG_FILE)
    a,b,c,d=st.columns(4)
    a.metric("원본 열",len(original.columns) if original is not None else 46)
    b.metric("선택 열",len(df.columns))
    c.metric("제외 열",46-len(df.columns))
    d.metric("남은 결측치",int(df.isna().sum().sum()))
    st.write("**전처리 원칙:** 필요하지 않은 열만 제외하고, 남긴 열의 응답 코드·열 이름·행 순서는 유지합니다.")
    st.write(f"중복 응답자 ID: **{int(df[ident].duplicated().sum()):,}개** / 코드 1·2 이외의 값이 나타난 매체 열: **{len(bad)}개**")
    st.subheader("전체 46개 변수의 유지·제외 이유")
    if original is not None:
        names=list(original.columns)
        original_ok=len(names)==46
        if not original_ok: st.warning("원본 CSV 열 개수가 46개가 아닙니다. 열 이름과 순서를 확인해 주세요.")
    else:
        st.warning("원본 46열 CSV를 찾지 못했습니다. 정확한 삭제 변수명 및 전후 셀 검증에는 원본 CSV가 필요합니다.")
        names=[f"원본 {i}번 열 (원본 CSV 필요)" for i in range(1,47)]
    reasons=pd.DataFrame([{"원본 열 번호":i,"변수 이름":names[i-1],
       "처리":"유지" if i in KEEP else "제외","이유":source_reason(i)}
       for i in range(1,min(46,len(names))+1)])
    filter_status=st.selectbox("표시할 변수",["전체","유지","제외"])
    view=reasons if filter_status=="전체" else reasons[reasons["처리"]==filter_status]
    st.dataframe(view,hide_index=True,width="stretch")
    st.download_button("전처리 근거표 CSV 다운로드",reasons.to_csv(index=False).encode("utf-8-sig"),"전처리_근거표.csv","text/csv")
    if original is not None and len(original.columns)>=40:
        expected=original.iloc[:,[i-1 for i in KEEP]]
        same_headers=expected.columns.tolist()==df.columns.tolist()
        same_values=expected.reset_index(drop=True).astype("string").fillna("<NA>").equals(df.reset_index(drop=True).astype("string").fillna("<NA>"))
        if same_headers and same_values:
            st.success("전체 선택 셀 검증 통과: 13개 열의 이름, 순서 및 모든 값이 원본과 일치합니다.")
        else:
            st.warning(f"원본 일치 확인 필요: 열 이름 일치={same_headers}, 셀 값 일치={same_values}")
    st.caption("변수를 제외한 이유는 주제 범위에 관한 판단입니다. 예측력이 낮다고 실험으로 검증한 의미는 아닙니다.")

elif menu.startswith("③"):
    st.header("데이터 분포 시각화 — 입력 변수(X)")
    rates=(xmedia.mean()*100).sort_values(ascending=False)
    rate_df=pd.DataFrame({"매체":[short(c) for c in rates.index],"이용 경험 비율(%)":rates.values})
    chart=alt.Chart(rate_df).mark_bar().encode(
        x=alt.X("이용 경험 비율(%):Q",scale=alt.Scale(domain=[0,100])),
        y=alt.Y("매체:N",sort="-x"),tooltip=["매체","이용 경험 비율(%)"]).properties(height=380)
    show_chart(chart)
    st.dataframe(rate_df.round(2),hide_index=True,width="stretch")
    st.subheader("매체별 이용 여부 비율")
    stacked=pd.concat([pd.DataFrame({"매체":short(c),"응답":["이용함","이용 안함"],
      "비율(%)":[xmedia[c].mean()*100,(1-xmedia[c].mean())*100]}) for c in media],ignore_index=True)
    show_chart(alt.Chart(stacked).mark_bar().encode(
        y=alt.Y("매체:N",sort=[short(c) for c in rates.index]),
        x=alt.X("비율(%):Q",stack="normalize",axis=alt.Axis(format="%")),
        color="응답:N",tooltip=["매체","응답","비율(%)"]).properties(height=370))
    st.subheader("매체 이용 조합 (상위 10개)")
    combos=xmedia.fillna(-1).astype(int).astype(str).agg("".join,axis=1).value_counts().head(10)
    combo_frame=pd.DataFrame({"이용 패턴":[" · ".join(short(media[j]) for j,s in enumerate(code) if s=="1") or "해당 없음" for code in combos.index],"응답자 수":combos.values})
    st.dataframe(combo_frame,hide_index=True,width="stretch")
    if grade:
        st.subheader("학교급 분포")
        school=df[grade].astype("string").map({"1":"초등학교","2":"중학교","3":"고등학교"}).fillna("기타").value_counts()
        st.bar_chart(school)
    if year:
        st.subheader("학년별 응답자 분포")
        st.bar_chart(df[year].astype("string").value_counts().sort_index())

elif menu.startswith("④"):
    st.header("변수 간 관계 분석")
    st.caption("각 매체 이용함=1, 이용 안함=0으로 임시 변환한 복사본으로 상관계수를 계산합니다.")
    corr=xmedia.corr()
    melted=corr.stack().reset_index()
    melted.columns=["변수 A","변수 B","상관계수"]
    melted["변수 A"]=melted["변수 A"].map(short)
    melted["변수 B"]=melted["변수 B"].map(short)
    show_chart(alt.Chart(melted).mark_rect().encode(
        x=alt.X("변수 A:N",axis=alt.Axis(labelAngle=-45)),
        y="변수 B:N",
        color=alt.Color("상관계수:Q",scale=alt.Scale(scheme="redblue",domain=[-1,1])),
        tooltip=["변수 A","변수 B",alt.Tooltip("상관계수:Q",format=".3f")]).properties(height=500))
    st.info("상관관계가 높아도 바로 삭제하지 않습니다. 실제 분류 성능을 평가한 뒤 결정해야 합니다.")

elif menu.startswith("⑤") or menu.startswith("⑥"):
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (accuracy_score,precision_score,recall_score,
        f1_score,confusion_matrix,roc_auc_score,roc_curve)
    st.header("분류 모델 학습 및 평가" if menu.startswith("⑤") else "탐구 결론")
    if bad or ytarget.isna().any():
        st.error("이용 경험 코드 또는 타깃에 잘못된/누락된 값이 있습니다. 모델 학습 전에 확인이 필요합니다.")
        st.stop()
    # ID와 타깃은 X에서 제외. 모든 범주형 인코딩은 학습 파이프라인 안에서 수행.
    inputcols=media+([grade] if grade else [])+([year] if year else [])
    rawX=df[inputcols].copy().astype("string")
    y=ytarget.astype(int)
    X_train,X_test,y_train,y_test=train_test_split(rawX,y,test_size=.2,stratify=y,random_state=42)
prep=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),inputcols)])
    models={
      "로지스틱 회귀": LogisticRegression(max_iter=1500,random_state=42),
      "랜덤 포레스트": RandomForestClassifier(n_estimators=100,min_samples_leaf=3,random_state=42,n_jobs=-1)
    }
    records={}
    for name,base in models.items():
        pipe=Pipeline([("preprocess",prep),("model",base)])
        pipe.fit(X_train,y_train)
        pred=pipe.predict(X_test)
        prob=pipe.predict_proba(X_test)[:,list(pipe.classes_).index(1)]
        records[name]={"정확도":accuracy_score(y_test,pred),"정밀도":precision_score(y_test,pred,zero_division=0),
        "재현율":recall_score(y_test,pred,zero_division=0),"F1":f1_score(y_test,pred,zero_division=0),
        "ROC-AUC":roc_auc_score(y_test,prob),"pred":pred,"prob":prob}
    result=pd.DataFrame([{ "모델":name, **{k:v for k,v in item.items() if isinstance(v,(int,float,np.floating))}}
        for name,item in records.items()])
    st.write("학습 80% / 평가 20%, 정답 비율 유지, 평가 데이터를 학습에 사용하지 않음")
    st.dataframe(result.round(3),hide_index=True,width="stretch")
    if menu.startswith("⑤"):
        model=st.selectbox("평가 결과 확인할 모델",list(records))
        r=records[model]
        cm=confusion_matrix(y_test,r["pred"],labels=[0,1])
        heat=pd.DataFrame([{"실제":a,"예측":b,"건수":int(cm[i,j])} for i,a in enumerate(["이용 안함","이용함"]) for j,b in enumerate(["이용 안함","이용함"])])
        st.subheader("혼동행렬")
        show_chart(alt.Chart(heat).mark_rect().encode(x="예측:N",y="실제:N",color="건수:Q",tooltip=["실제","예측","건수"]).properties(height=270))
        st.subheader("ROC 곡선")
        roc=[]
        for name,z in records.items():
            fpr,tpr,_=roc_curve(y_test,z["prob"])
            roc.extend({"모델":name,"거짓양성률":float(x),"참양성률":float(y)} for x,y in zip(fpr,tpr))
        show_chart(alt.Chart(pd.DataFrame(roc)).mark_line().encode(x="거짓양성률:Q",y="참양성률:Q",color="모델:N").properties(height=300))
    else:
        best=result.sort_values("F1",ascending=False).iloc[0]
        st.success(f"현재 평가 데이터에서 F1 점수가 더 높은 모델: {best['모델']} (F1={best['F1']:.3f})")
        st.write("이는 이 데이터에 대한 실험 결과이며, 다른 집단이나 미래 데이터에 같은 성능을 보장하지 않습니다.")
        st.write("전처리에서는 입력할 매체 이용 변수만 선택하고, 결측치가 없는 원본 응답은 유지했습니다. 정답 누수 가능성이 있는 변수도 제외했습니다.")
        st.write("**추가 탐구:** 디지털 매체 이용 변수만 사용한 모델과 학교급·학년을 더한 모델을 비교하고, 다른 학교 자료에서도 성능을 확인할 수 있습니다.")
