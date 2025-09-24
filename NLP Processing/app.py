import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download resources if not already installed
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('omw-1.4')

# Streamlit App
st.set_page_config(page_title="NLP Preprocessing App", layout="centered")

st.title("📝 NLP Preprocessing: Tokenization, Stemming & Lemmatization")

st.write("Enter your text below and see how Tokenization, Stemming, and Lemmatization work in Natural Language Processing.")

# User input
text = st.text_area("✍️ Enter your text here:", "Natural Language Processing makes machines understand human language")

if st.button("Process Text"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        # Tokenization
        tokens = word_tokenize(text)

        # Stemming
        ps = PorterStemmer()
        stemmed_words = [ps.stem(word) for word in tokens]

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]

        # Display Results
        st.subheader("🔹 Tokenization")
        st.write(tokens)

        st.subheader("🔹 Stemming (Porter Stemmer)")
        st.write(stemmed_words)

        st.subheader("🔹 Lemmatization (WordNet Lemmatizer)")
        st.write(lemmatized_words)

        # Comparison Table
        st.subheader("📊 Comparison Table")
        st.table({"Token": tokens, "Stemmed": stemmed_words, "Lemmatized": lemmatized_words})
