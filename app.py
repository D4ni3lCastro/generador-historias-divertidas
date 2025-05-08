import streamlit as st
from PIL import Image
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    pipeline,
    MarianMTModel,
    MarianTokenizer
)

st.set_page_config(page_title="Generador de historias divertidas", layout="centered")
st.title("🎨 Generador de historias divertidas con tu imagen")

# Cargar modelo de caption de imagen
@st.cache_resource
def load_image_captioning_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_image_captioning_model()

# Cargar traductor inglés -> español
@st.cache_resource
def load_translator():
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    return tokenizer, model

translator_tokenizer, translator_model = load_translator()

def translate_to_spanish(text):
    inputs = translator_tokenizer(text, return_tensors="pt", padding=True)
    translated = translator_model.generate(**inputs)
    return translator_tokenizer.decode(translated[0], skip_special_tokens=True)

# Cargar modelo generador de historias
@st.cache_resource
def load_story_generator():
    return pipeline("text2text-generation", model="google/flan-t5-small")

story_generator = load_story_generator()

# Subir imagen
uploaded_file = st.file_uploader("Sube tu imagen divertida 🐶🏀", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen cargada", use_container_width=True)

    # Generar descripción en inglés
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    # Traducir descripción al español
    caption_es = translate_to_spanish(caption)

    st.subheader("🧠 Descripción generada (en español):")
    st.write(caption_es)

    # Generar historia divertida en español
    prompt = f"Escribe una historia graciosa basada en esta descripción: '{caption_es}'."
    try:
        story = story_generator(prompt, max_length=100, do_sample=True)[0]['generated_text']
        st.subheader("🤣 Historia divertida:")
        st.write(story)
    except Exception as e:
        st.error(f"Ocurrió un error al generar la historia: {e}")
