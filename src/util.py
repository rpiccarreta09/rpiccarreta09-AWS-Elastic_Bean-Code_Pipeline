import base64
from fastapi import FastAPI, File, UploadFile
import uvicorn
from transformers import ViTFeatureExtractor, ViTForImageClassification
import streamlit as st
from PIL import ImageOps, Image
import numpy as np
import io

app = FastAPI()

# Cargar el extractor de características y el modelo de Hugging Face
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

@app.get("/")
def home():
    # Retorna un simple mensaje de texto
    return ''

# Definir un endpoint para manejar la subida de imágenes y su clasificación
@app.post("/computer-vision")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Leer el archivo de imagen y convertirlo en una imagen PIL
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Preprocesar la imagen y prepararla para el modelo
        inputs = feature_extractor(images=image, return_tensors="pt")

        # Realizar la predicción
        outputs = model(**inputs)
        logits = outputs.logits

        # Obtener la clase con la mayor probabilidad
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = model.config.id2label[predicted_class_idx]

        # Retornar la clase predicha
        return {"message": "La imagen ha sido analizada!", "predicted_class": predicted_class}
    
    except Exception as e:
        # Retornar un mensaje de error si ocurre alguna excepción
        return {"error": f"Ocurrió un error: {str(e)}"}
