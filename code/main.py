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
    st.dataframe(rate_df.round(2),hide_index=True,use_container_width=True)
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
    st.dataframe(combo_frame,hide_index=True,use_container_width=True)
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
    st.dataframe(result.round(3),hide_index=True,use_container_width=True)
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
