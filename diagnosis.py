import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

#title
st.title("Pneumonia diagnosis model")
st.header('The model identifies if the patient has pneumonia or not')
st.write("Please kindly upload only chest X-ray image to get diagnosis result")

#Upload picture
file = st.file_uploader("", type=['png', 'jpeg', 'gif', 'svg', 'jpg'])

if file:
    #PIL convert image
    img =PILImage.create(file)
    image = Image.open(file)
    new_image = image.resize((600, 500))
    
    #Model
    model = load_learner('filter.pkl')

    #Prediction
    row, pred_id, probs = model.predict(img)
    filter = pred_id*100

    
    if filter > 65:
       
        st.image(new_image)
        
        #Model
        main_model = load_learner('pneumonia_diagnosis.pkl')

        #Prediction
        pred, pred_id, probs = main_model.predict(img)

        #Print result
        st.success(f"Prediction: {pred}")
        st.info(f"Probability: {probs[pred_id]*100:.2f}%")

        #Plotting
        fig = px.bar(x=probs*100, y=main_model.dls.vocab)
        st.plotly_chart(fig)
    else:
        st.warning('Please upload only chest X-ray image', icon="⚠️")
        st.write("The image you have uploaded is:")
        st.image(new_image)


