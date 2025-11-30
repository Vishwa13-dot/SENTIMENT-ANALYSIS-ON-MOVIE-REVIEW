import streamlit as st
import pickle
import re
from nltk.stem import PorterStemmer
import nltk

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load model and vectorizer
model = pickle.load(open("Sentiment Analysis on Movie Review.pkl", "rb"))
vectorizer = pickle.load(open("count-Vectorizer.pkl", "rb"))

# Preprocessing
ps = PorterStemmer()
def preprocess(text):
    review = re.sub("[^a-zA-Z]", " ", text)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stopwords.words("english")]
    return " ".join(review)

# Page config
st.set_page_config(page_title="Movie Review Sentiment Analyzer", layout="centered")

# Background & style
def set_background():
    background_url = "https://wallpaperbat.com/img/426398-red-seat-cinema-and-theatre-hd-4k-wallpaper-and-background.jpg"
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("{background_url}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        .title {{
            color: #fff;
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            padding: 20px;
            text-shadow: 2px 2px 6px #000;
        }}

        .review-label-box {{
            background-color: rgba(240, 240, 240, 0.85);
            padding: 12px 18px;
            border-radius: 12px;
            box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.25);
            font-size: 18px;
            font-weight: 500;
            font-family: 'Poppins', sans-serif;
            color: #222;
            margin-bottom: 15px;
        }}

        .stTextArea textarea {{
            font-size: 16px !important;
            background-color: rgba(255, 255, 255, 0.88);
            color: #000;
            font-weight: 500;
            font-family: 'Poppins', sans-serif;
        }}

        .stButton button {{
            font-size: 18px;
            background-color: #e50914;
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 8px;
            padding: 10px 25px;
            box-shadow: 2px 2px 6px #000;
            font-family: 'Poppins', sans-serif;
        }}

        .stButton button:hover {{
            background-color: #b0060f;
        }}
        </style>
    """, unsafe_allow_html=True)

set_background()

# Title
st.markdown('<div class="title">🎬 Movie Review Sentiment Analyzer</div>', unsafe_allow_html=True)

# Banner
st.image("https://fossbytes.com/wp-content/uploads/2019/06/hindi-movie-sites-.jpg", use_container_width=True)

# White box label above text area
st.markdown('<div class="review-label-box">📝 Enter your movie review below:</div>', unsafe_allow_html=True)
review = st.text_area("", height=200, placeholder="Type your review here...")

# Analyze Button
if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review before analyzing.")
    else:
        processed = preprocess(review)
        vect = vectorizer.transform([processed])
        prediction = model.predict(vect)

        if prediction[0] == 1:
            st.success("🎉 IT'S A POSITIVE REVIEW!")
            st.image("https://media.giphy.com/media/111ebonMs90YLu/giphy.gif", width=400)
        else:
            st.error("😞 IT'S A NEGATIVE REVIEW.")
            st.image("https://media4.giphy.com/media/5hc2bkC60heU/giphy.gif", width=350)
