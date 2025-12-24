import streamlit as st
import speech_recognition as sr
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""

with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    return " ".join(tokens)

st.title("Voice-Based Sentiment Analysis")
st.markdown(
    "Speak using your microphone OR type text. If voice is provided, it will be used automatically."
)

if st.button("Speak"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = recognizer.listen(source)

        try:
            st.session_state.voice_text = recognizer.recognize_google(audio)
            st.success("Voice captured successfully!")
        except:
            st.error("Could not understand audio")
            st.session_state.voice_text = ""

st.text_area(
    "Transcribed Voice Text",
    st.session_state.voice_text,
    height=100
)

user_input_text = st.text_area("Or type your review here (optional):")

if st.button("Predict Sentiment"):
    final_text = (
        st.session_state.voice_text
        if st.session_state.voice_text.strip() != ""
        else user_input_text
    )

    if final_text.strip() == "":
        st.warning("Please provide some text or voice input!")
    else:
        cleaned = clean_text(final_text)
        vect = vectorizer.transform([cleaned])
        prediction = model.predict(vect)[0]

        if prediction == 1:
            st.success("Positive Sentiment")
        else:
            st.error("Negative Sentiment")
