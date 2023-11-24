# Para crear el requirements.txt ejecutamos 
# pipreqs --encoding=utf8 --force

# Primera Carga a Github
# git init
# git add .
# git remote add origin https://github.com/nicoig/VetGPT.git
# git commit -m "Initial commit"
# git push -u origin master

# Actualizar Repo de Github
# git add .
# git commit -m "Se actualizan las variables de entorno"
# git push origin master

# En Render
# agregar en variables de entorno
# PYTHON_VERSION = 3.9.12

# git remote set-url origin https://github.com/nicoig/VetGPT.git
# git remote -v
# git push -u origin main


################################################
##


import streamlit as st
import base64
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from io import BytesIO

# Cargar las variables de entorno para las claves API
load_dotenv(find_dotenv())

# Función para codificar imágenes en base64
def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

# Función para descargar información
def download_button(object_to_download, download_filename, button_text):
    buffer = BytesIO()
    buffer.write(object_to_download.encode())
    buffer.seek(0)
    return st.download_button(
        button_text,
        buffer,
        download_filename,
        "text/plain"
    )

# Configura el título y subtítulo de la aplicación en Streamlit
st.title("VetGPT")

st.markdown("""
    <style>
    .small-font {
        font-size:18px !important;
    }
    </style>
    <p class="small-font">Hola, soy VetGPT y te ayudaré con dudas que tengas sobre tu mascota, carga tu imágen y te proporcionaré consejos e información sobre tu mascota</p>
    """, unsafe_allow_html=True)

# Imagen de cabecera
st.image('img/robot.png', width=250)

# Carga de imagen y texto por el usuario
uploaded_file = st.file_uploader("Carga una imagen de tu mascota", type=["jpg", "png", "jpeg"])
input_text = st.text_input("Describe tu consulta veterinaria aquí")

# Variable para almacenar la información generada
informacion_generada = ""

# Botón de enviar y proceso principal
if st.button("Enviar Consulta") and uploaded_file is not None and input_text:
    with st.spinner('Analizando tu consulta...'):
        image = encode_image(uploaded_file)

        # Analizar la imagen y el texto con la IA
        chain = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024)
        msg = chain.invoke(
            [AIMessage(content="Basándose en la imagen y la descripción proporcionada, identifique el animal y la raza, y proporcione información relevante."),
             HumanMessage(content=[{"type": "text", "text": input_text},
                                   {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}])
            ]
        )

        informacion_mascota = msg.content
        informacion_generada += "**Información sobre tu mascota basada en tu consulta:**\n" + informacion_mascota + "\n\n"

        # Generar recomendaciones de cuidado para la mascota
        chain = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=1024)
        prompt_cuidado = PromptTemplate.from_template(
            """
            Dada la siguiente información sobre una mascota:
            {pet_info}
            
            Basándose en la edad, raza y condición del animal, responda las siguientes preguntas:
            1. ¿Cuál es el origen y la historia breve de esta raza?
            2. ¿Cuál es la esperanza de vida promedio de esta raza?
            3. ¿Cuáles son las características físicas y de comportamiento típicas de esta raza?
            4. ¿Cuáles son las necesidades de ejercicio específicas para esta raza?
            5. ¿Qué problemas de salud comunes pueden afectar a esta raza y cómo se pueden prevenir o manejar?
            6. ¿Cuáles son las recomendaciones de alimentación y nutrición para esta raza, teniendo en cuenta su edad y tamaño?
            7. ¿Qué cuidado del pelaje y aseo se recomienda para esta raza?
            8. ¿Hay consejos específicos de socialización y entrenamiento que se deban considerar para esta raza?

            Output:
            """
        )
        runnable = prompt_cuidado | chain | StrOutputParser()
        cuidado = runnable.invoke({"pet_info": informacion_mascota})
        informacion_generada += "**Recomendaciones e información extra sobre tu mascota:**\n" + cuidado

        st.markdown(informacion_generada)

# Botón para descargar la información generada
if informacion_generada:
    download_button(informacion_generada, "informacion_mascota.txt", "Descargar Información")