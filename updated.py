import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

#title
st.title("Pneumonia diagnosis model")

#Upload picture
file = st.file_uploader("Upload picture", type=['png', 'jpeg', 'gif', 'svg', 'jpg'])
if file:
    st.image(file)

    #PIL convert image
    img =PILImage.create(file)

    #Model
    model = load_learner('filter.pkl')

    #Prediction
    pred, pred_id, probs = model.predict(img)
    filter = probs[pred_id]*100
    #Print result
    #st.success(f"Prediction: {pred}")
    #st.info(f"Probability: {probs[pred_id]*100:.2f}%")

    #Plotting
    #fig = px.bar(x=probs*100, y=model.dls.vocab)
    #st.plotly_chart(fig)
    if filter > 65:
        
        image = Image.open(file)
        new_image = image.resize((600, 500))
        st.image(new_image)

        #PIL convert image
        img =PILImage.create(file)

        #Model
        model1 = load_learner('pneumonia_diagnosis.pkl')

        #Prediction
        pred, pred_id, probs = model1.predict(img)

        #Print result
        st.success(f"Prediction: {pred}")
        st.info(f"Probability: {probs[pred_id]*100:.2f}%")

        #Plotting
        fig = px.bar(x=probs*100, y=model1.dls.vocab)
        st.plotly_chart(fig)
    else:
        print("Please upload chest X-ray test")

