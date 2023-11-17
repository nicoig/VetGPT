# Para crear el requirements.txt ejecutamos 
# pipreqs --encoding=utf8 --force

# Primera Carga a Github
# git init
# git add .
# git remote add origin https://github.com/nicoig/docGPT.git
# git commit -m "Initial commit"
# git push -u origin master

# Actualizar Repo de Github
# git add .
# git commit -m "Se actualizan las variables de entorno"
# git push origin master

# En Render
# agregar en variables de entorno
# PYTHON_VERSION = 3.9.12

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

# Configura el título y subtítulo de la aplicación en Streamlit
st.title("DocGPT")
st.subheader("Consulta médica virtual con análisis de imágenes")
# Imagen
st.image('img/robot.jpg', width=350)


# Carga de imagen y texto por el usuario
uploaded_file = st.file_uploader("Carga una imagen relacionada con tu consulta", type=["jpg", "png", "jpeg"])
input_text = st.text_input("Describe tu consulta o síntomas aquí")

# Botón de enviar y proceso principal
if st.button("Enviar Consulta") and uploaded_file is not None and input_text:
    with st.spinner('Analizando tu consulta...'):
        image = encode_image(uploaded_file)

        # Analizar la imagen y el texto con la IA
        chain = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024)
        msg = chain.invoke(
            [AIMessage(content="Basándose en la imagen y la descripción proporcionada, identifique posibles afecciones relacionadas y proporcione información general."),
             HumanMessage(content=[{"type": "text", "text": input_text},
                                   {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}])
            ]
        )

        diagnostico = msg.content
        st.markdown("**Información general basada en tu consulta:**")
        st.write(diagnostico)

        # Generar recomendaciones de tratamiento
        chain = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=1024)
        prompt_tratamiento = PromptTemplate.from_template(
            """
            Dada la siguiente información general basada en una consulta médica:
            {diagnosis}
            ¿Qué recomendaciones generales se podrían dar? Estas son solo sugerencias generales y no deben reemplazar el asesoramiento médico profesional.

            Output:
            """
        )
        runnable = prompt_tratamiento | chain | StrOutputParser()
        tratamiento = runnable.invoke({"diagnosis": diagnostico})
        st.markdown("**Sugerencias generales de tratamiento:**")
        st.write(tratamiento)

        # Sugerir qué tipo de profesional médico visitar
        prompt_profesional = PromptTemplate.from_template(
            """
            Basándose en la siguiente información general:
            {diagnosis}
            ¿Qué tipo de profesional médico sería adecuado para una consulta? Esta es solo una sugerencia y no reemplaza la consulta médica profesional.

            Output:
            """
        )
        runnable = prompt_profesional | chain | StrOutputParser()
        profesional = runnable.invoke({"diagnosis": diagnostico})
        st.markdown("**Profesional Médico que debes visitar:**")
        st.write(profesional)

# Nota: Asegúrate de tener configuradas tus claves API y cualquier otro ajuste específico necesario para tu entorno y APIs.
