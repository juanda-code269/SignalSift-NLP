import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,recall_score,roc_auc_score
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Article Pattern Classifier",page_icon="📰",layout="wide")
st.title("Article Pattern Classifier")
st.warning("This is a dataset-label classifier—not a truth detector, fact checker, or credibility score.")


@st.cache_data
def demo():
    rng=np.random.default_rng(5)
    class0=["city council published the audited budget report","researchers released methods and confidence intervals","agency statement names sources and effective date","court filing documents the procedural timeline","local officials confirmed the vote in public records"]
    class1=["shocking secret they refuse to reveal act now","anonymous insiders expose unbelievable miracle coverup","you will not believe this viral claim share immediately","hidden plot proves everything experts hate this","breaking rumor destroys the official story overnight"]
    context=[" economy election health technology climate education", " policy community international science update analysis"]
    rows=[]
    for i in range(700):
        label=i%2;base=rng.choice(class1 if label else class0);text=base+rng.choice(context)+" "+rng.choice(base.split())
        rows.append((text,label))
    return pd.DataFrame(rows,columns=["text","label"])


def load(upload):
    if upload is None:return demo(),"Synthetic style-pattern corpus; labels mean Demo Class 0/1"
    d=pd.read_csv(upload);text=next((c for c in d if c.lower() in {"text","article","content","title"}),None);label=next((c for c in d if c.lower() in {"label","class","target","fake"}),None)
    if not text or not label:raise ValueError("CSV needs text/article and label/class columns.")
    d=d[[text,label]].dropna().rename(columns={text:"text",label:"label"});values=d.label.unique()
    if len(values)!=2:raise ValueError("This demo supports exactly two dataset labels.")
    if not set(values).issubset({0,1}):d.label=(d.label==values[-1]).astype(int)
    return d,f"Uploaded labeled dataset: {upload.name}"


upload=st.sidebar.file_uploader("Labeled text CSV",type="csv")
try:
    df,source=load(upload)
    if len(df)<100:raise ValueError("At least 100 labeled texts are required.")
except Exception as exc:st.error(str(exc));st.stop()
train,test=train_test_split(np.arange(len(df)),test_size=.25,random_state=42,stratify=df.label)
models={"Logistic Regression":LogisticRegression(max_iter=1000,class_weight="balanced"),"Naive Bayes":MultinomialNB(alpha=.7)}
fitted={};rows=[]
for name,model in models.items():
    pipe=Pipeline([("tfidf",TfidfVectorizer(stop_words="english",ngram_range=(1,2),min_df=2,max_features=18000)),("model",model)]).fit(df.text.iloc[train],df.label.iloc[train])
    p=pipe.predict_proba(df.text.iloc[test])[:,1];pred=p>=.5;fitted[name]=(pipe,p,pred)
    rows.append({"Model":name,"Accuracy":accuracy_score(df.label.iloc[test],pred),"Precision":precision_score(df.label.iloc[test],pred),"Recall":recall_score(df.label.iloc[test],pred),"F1":f1_score(df.label.iloc[test],pred),"ROC-AUC":roc_auc_score(df.label.iloc[test],p)})
choice=st.sidebar.selectbox("Model",list(models));pipe,prob,pred=fitted[choice]
tabs=st.tabs(["Evaluate","Classify text","Influential features","Error analysis","Limitations"])
with tabs[0]:
    st.info(source);st.dataframe(pd.DataFrame(rows).style.format({c:"{:.3f}" for c in rows[0] if c!="Model"}),width="stretch")
    matrix=confusion_matrix(df.label.iloc[test],pred);st.plotly_chart(px.imshow(matrix,text_auto=True,labels={"x":"Predicted","y":"Actual"},title=f"Confusion matrix — {choice}"),width="stretch")
with tabs[1]:
    text=st.text_area("Article text",height=200,placeholder="Paste text to analyze against the selected dataset labels…")
    if text:
        p=float(pipe.predict_proba([text])[0,1]);st.metric("Predicted dataset class",f"Class {int(p>=.5)}");st.metric("Model probability for Class 1",f"{p:.1%}")
        st.caption("This output reflects linguistic patterns learned from the supplied labels. It does not determine factual truth.")
with tabs[2]:
    vectorizer=pipe.named_steps["tfidf"];model=pipe.named_steps["model"];terms=np.array(vectorizer.get_feature_names_out())
    score=model.coef_[0] if hasattr(model,"coef_") else model.feature_log_prob_[1]-model.feature_log_prob_[0]
    show=pd.concat([pd.DataFrame({"Term":terms[np.argsort(score)[:18]],"Association":"Class 0","Weight":np.sort(score)[:18]}),pd.DataFrame({"Term":terms[np.argsort(score)[-18:]],"Association":"Class 1","Weight":np.sort(score)[-18:]})])
    st.plotly_chart(px.bar(show,x="Weight",y="Term",color="Association",orientation="h",title="Most class-associated TF-IDF features"),width="stretch")
with tabs[3]:
    errors=df.iloc[test].copy();errors["Predicted"]=pred.astype(int);errors["Probability class 1"]=prob;errors=errors[errors.label!=errors.Predicted]
    st.dataframe(errors[["text","label","Predicted","Probability class 1"]].head(30),width="stretch",hide_index=True)
    st.write("Inspect errors for source shortcuts, quoted sensational language, ambiguous labels, sarcasm, and topic leakage.")
with tabs[4]:
    st.markdown("""A text classifier can learn publisher, topic, era, style, or annotation artifacts instead of reliability. Labels may encode source bias and political context. Language changes; satire, quotations, and sarcasm are difficult; probability can be poorly calibrated out of domain. Use a legitimately licensed labeled dataset and preserve its documentation. Factual verification requires evidence and source checking beyond statistical text patterns.""")

