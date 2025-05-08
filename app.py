import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

st.set_page_config(page_title="Generador de historias divertidas", layout="centered")
st.title("🎨 Generador de historias divertidas con tu imagen")

# Cargar modelo
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

# Subir imagen
uploaded_file = st.file_uploader("Sube tu imagen divertida 🐶🏀", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Procesar imagen y generar caption
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    st.subheader("🧠 Descripción generada:")
    st.write(caption)

    # Generar historia divertida a partir del caption
    prompt = f"Escribe una historia graciosa basada en esta descripción: '{caption}'."
    
    # Usamos otro modelo como text-davinci o FLAN-T5
    from transformers import pipeline
    story_generator = pipeline("text2text-generation", model="google/flan-t5-small")

    story = story_generator(prompt, max_length=100, do_sample=True)[0]['generated_text']
    
    st.subheader("🤣 Historia divertida:")
    st.write(story)
