#import os
#os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from keras.models import load_model, Model
from keras.applications import DenseNet201
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.sequence import pad_sequences


@st.cache_resource
def loadCNN():
    """
    Loads DenseNet201 model for feature extraction.
    Args:
        None
    Returns:
        fe: keras.Model class object with modified output layer which outputs features and works as a feature extractor.
    """
    
    #model = DenseNet201(input_shape=(224, 224, 3))
    #model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    model = DenseNet201()
    # Modifying output layer to get features by removing final classification layer
    fe = Model(inputs=model.input, outputs=model.layers[-2].output)
    return fe

denseNet = loadCNN()

def loadDecoder():
    model = load_model("SuperAwesomeCaptioner.keras")
    return model

model = loadDecoder()

with open('tokenizer.pkl', 'rb') as tokenizerFile:
    tokenizer = pickle.load(tokenizerFile)
    
st.title("Image Caption Generator(Non-Attention)")
st.markdown("Simply upload an image and get an erroneous caption for it! Courtesy of LSTMs and CNNs")

inputImage = st.file_uploader("Pick any picture", type=['jpg', 'jpeg', 'png'])

if inputImage is not None:
    st.subheader("Input Picture")
    st.image(inputImage, caption="Uploaded Picture", use_column_width=True)
    
    st.subheader("Generated Caption")
    
    with st.spinner("Generating.........."):
        max_length = 34
        # Image Preprocessing
        image = load_img(inputImage, target_size=(224, 224))
        image = img_to_array(image)
        image = image / 255.
        image = np.expand_dims(image, axis=0)
        
        # Extracting image features using DenseNet201()
        features = denseNet.predict(image, verbose=0)
        
        def idx_to_word(integer, tokenizer):
            for word, index in tokenizer.word_index.items():
                if index == integer:
                    return word
            return None
        
        def predictCaption(model, feature, tokenizer, max_length):
            caption = "startseq"
            for i in range(max_length):
                sequence = tokenizer.texts_to_sequences([caption])[0]
                sequence = pad_sequences([sequence], max_length)
                
                y_pred = np.argmax(model.predict([feature, sequence]))
                
                word = idx_to_word(y_pred, tokenizer)
                
                if word is None or word == "endseq":
                    break
                    
                caption += " " + word
                
            return caption
        
        # Generate caption!
        generatedCaption = predictCaption(model, features, tokenizer, max_length).replace("startseq", "").replace("endseq", "")
        
        st.markdown(f'<p style="font-style: italic;">"{generatedCaption}"<p>', unsafe_allow_html=True)
