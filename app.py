import streamlit as st
import requests  
import json
from fastapi import FastAPI, File, UploadFile
import uvicorn
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import io

from util import upload_image

col1, col2 = st.columns(2, gap="large")

with col1:

    # Título de la aplicación
    st.title('Text to Image')

    # Crear un campo de entrada de texto
    input_text = st.text_input('Ingrese algún texto')

    # Crear un botón
    if st.button('Mostrar Imagen'):
        url =  "https://stablediffusionapi.com/api/v4/dreambooth"  

        payload = json.dumps({  
        "key":  "1MO8AHiPgViQs3Pk1SfKfEWL2RtP7C9Uxs3ZPvuUjF75635XTAuTL47RfeOU",  
        "model_id":  "juggernaut-xl-v5",  
        "prompt":  input_text,  
        "negative_prompt":  "painting, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, deformed, ugly, blurry, bad anatomy, bad proportions, extra limbs, cloned face, skinny, glitchy, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted face, extra legs, anime",  
        "width":  "512",  
        "height":  "512",  
        "samples":  "1",  
        "num_inference_steps":  "30",  
        "safety_checker":  "no",  
        "enhance_prompt":  "yes",  
        "seed":  None,  
        "guidance_scale":  7.5,  
        "multi_lingual":  "no",  
        "panorama":  "no",  
        "self_attention":  "no",  
        "upscale":  "no",  
        "embeddings":  "embeddings_model_id",  
        "lora":  "lora_model_id",  
        "webhook":  None,  
        "track_id":  None  
        })  

        headers =  {  
        'Content-Type':  'application/json'  
        }  
        
        response = requests.request("POST", url, headers=headers, data=payload) 

        image_link = response.json()['output'][0]

        # Si se hace clic en el botón, mostrar la imagen
        st.image(image_link,
                caption='Imagen Mostrada')

with col2:

    st.title('Clasificación de Imágenes')

    file=st.file_uploader('Cargar imagen para clasificación:')
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    
    if file is not None:
        image = Image.open(file)
        inputs = feature_extractor(images=image, return_tensors="pt")

        # Realizar la predicción
        outputs = model(**inputs)
        logits = outputs.logits

        # Obtener la clase con la mayor probabilidad
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = model.config.id2label[predicted_class_idx]
        st.image(image)
        st.write(predicted_class)
