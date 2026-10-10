import streamlit as st
import joblib
import requests
model=joblib.load("model.pkl")
vectors=joblib.load("vectors.pkl")
df=joblib.load("dataframe.pkl")

st.set_page_config(layout='wide')
st.markdown("""
<style>
.baner{
background:linear-gradient(90deg,#4aaaaa,#000000);
padding:25px;
border-radius:12px;
text-align:center;
color:white;
font-size:35px;
font-weight:bold;
margin-bottom:25px
}
</style>
<div class="baner">
        Movie Recommendation Project

</div>
""",unsafe_allow_html=True)

st.sidebar.image("926015_passport_photo.PNG")
st.sidebar.title("About us")
st.sidebar.text("we are developing ml projects based on NLP ")
st.sidebar.title("About Projects")
st.sidebar.text("This project recommend movie bases on story write direcor of watched movie")
st.sidebar.title("Contact us")
st.sidebar.text("+916283008506")
watched_movie=st.selectbox("Select Watched Movie",options=df['name'])
if st.button("Predict",key="b1"):
    index=df[df.name==watched_movie].index[0]
    distance,indexes=model.kneighbors(vectors[index].toarray())
    for i in indexes[0][1:]:
        st.text(df.loc[i].values[1])
        mid=df.loc[i].values[0]
        resp=requests.get(" http://www.omdbapi.com/?i={mid}&apikey=443f1039")
        data=resp.json()
        st.image(data['Poster'],width=300)
