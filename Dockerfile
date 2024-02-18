FROM gcr.io/google.com/cloudsdktool/google-cloud-cli:455.0.0-slim

# Usar una imagen oficial de Python como imagen base
FROM python:3.10

# Establecer el directorio de trabajo en el contenedor
WORKDIR /usr/src/app

# Copiar el contenido del directorio actual en el contenedor en /usr/src/app
COPY . .

# Instalar PyTorch, torchvision y torchaudio
# Se especifica la URL del índice para descargar desde un sitio específico
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Instalar cualquier otro paquete necesario especificado en requirements.txt
# Se usa --no-cache-dir para no almacenar los archivos de caché de pip, reduciendo el tamaño de la imagen
RUN pip install --no-cache-dir -r requirements.txt

# Hacer disponible el puerto 8000 al mundo exterior a este contenedor
# Esto no publica el puerto, solo indica que el puerto está destinado a ser publicado
EXPOSE 8000

# Service must listen to $PORT environment variable.
# This default value facilitates local development.
ENV PORT 8080

# Ejecutar app.py cuando se inicie el contenedor
# uvicorn se usa como servidor ASGI para ejecutar la aplicación FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]


# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
#CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 app:app
