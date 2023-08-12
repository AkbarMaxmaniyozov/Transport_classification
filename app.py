import streamlit as st
from fastai.vision.all import *
import pathlib
plt=platform.system()
if plt == 'Linux':pathlib.WindowsPath = pathlib.PosixPath
import plotly.express as px

# title
st.title("Transportni klassifikatsiya qiluvchi model (car, boat, airplane)")

#rasmni joylash
file = st.file_uploader("Rasm yuklash", type=['png', 'jpeg', 'gif', 'svg'])
if file:
    st.image(file)
    #PIL convert
    img = PILImage.create(file)

    #Modelni yuklash
    model = load_learner("transport_model.pkl")

    #prediction
    pred, pred_id, probs = model.predict(img)
    st.success(pred)
    st.info(f"Ehtimollik: {probs[pred_id]:.1%}")

    #plotting
    fig = px.bar(x=probs*100, 
                 y=model.dls.vocab,
                title="Klasslar bo'yicha ehtimollik")
    st.plotly_chart(fig)
