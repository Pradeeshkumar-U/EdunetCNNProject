import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import math
import time

st.set_page_config(page_title="Waste Classification", page_icon="♻️", layout="wide")

model = load_model('waste_classifier.keras')

# file = st.file_uploader("Upload a file", type=["jpg", "png", "csv"])

st.title("♻️ Waste Classification App")
st.write("Upload an image to classify whether it's **Organic** or **Recyclable**.")
# st.image("banner.jpg", use_column_width=True)  # Add a banner image (optional)

st.title("Upload an Image")

st.sidebar.write("Developed by :\n**Pradeeshkumar** **U**")
    
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=False)

        img = image.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.reshape(img,[-1,224,224,3])

        with st.spinner("Classifying... Please wait"):
            time.sleep(2)  
            predictions = model.predict(img)
            predicted_class = np.argmax(predictions, axis=1)[0]
            

        st.success(f"### 🏷️ Predicted Waste Type: **{'Organic Waste' if predicted_class==0 else 'Recyclable Waste' if predicted_class==1 else 'Unknown'}**")

        st.write(f'Confidence Level : **{math.ceil(float(predictions[0][predicted_class]))*100}%**')

        # with col2:
        #     st.progress(float(predictions[0][predicted_class]))

        # with col3:
        #     st.empty()
            
        if predicted_class == 0:
            st.info("Organic waste can be composted! ♻️")
        else:
            st.warning("Recyclable waste should be processed properly! 🔄")
 
st.sidebar.markdown('---')           
st.sidebar.markdown("🔗 https://github.com/Pradeeshkumar-U")
st.sidebar.markdown("📧 23am044@kpriet.ac.in")

# def predict_image(image):
#     img = image.resize((224, 224)) 
#     img = img_to_array(img)
#     img = img / 255.0  
#     img = np.reshape(img,[-1,224,224,3]) 
    
#     predictions = model.predict(img)
#     predicted_class = np.argmax(predictions, axis=1)[0]
#     return 'Organic Waste' if predicted_class==0 else 'Recyclable Waste' if predicted_class==1 else 'Unknown'

# if file is not None:
#     img = Image.open(file)
    
#     prediction = predict_image(img)
    
#     st.image(img, caption="Uploaded Image", use_container_width=True)
    
#     st.success(f"Predicted Waste Type: **{prediction}**")
 