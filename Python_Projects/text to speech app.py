import streamlit as st
from gtts import gTTS
from IPython.display import Audio
import os

# Page config
st.set_page_config(page_title="Text to Speech App", page_icon="🔊", layout="centered")

# App title
st.title("🔊 Text to Speech Converter")
st.write("Convert your text into speech using **gTTS (Google Text-to-Speech)**")
# Language selection
lang = st.selectbox("Choose language:", ["en", "hi", "fr", "es", "de"])

# Text input
text_input = st.text_area("Enter text here:", placeholder="Type something...")

# Output file
output_file = "speech.mp3"

# Button
if st.button("Convert to Speech"):
    if text_input.strip() == "":
        st.warning("⚠️ Please enter some text before converting.")
    else:
        # Check if file exists → remove old file
        if os.path.exists(output_file):
            os.remove(output_file)

        # Generate speech
        tts = gTTS(text=text_input, lang=lang)
        tts.save(output_file)

        # Play audio
        with open(output_file, "rb") as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/mp3")

        # Download option
        st.download_button(
            label="Download Speech",
            data=audio_bytes,
            file_name="speech.mp3",
            mime="audio/mp3"
        )