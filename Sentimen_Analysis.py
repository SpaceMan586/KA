import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# Fungsi untuk menganalisis sentimen
def analyze_sentiment(text):
    analysis = TextBlob(str(text))  # Pastikan teks berupa string
    return analysis.sentiment.polarity

# Fungsi untuk mengategorikan sentimen
def categorize_sentiment(score):
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Judul aplikasi
st.title("Sentiment Analysis App")

# Upload file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    # Membaca file yang diunggah
    data = pd.read_csv(uploaded_file)

    # Pastikan kolom 'article_text' ada
    if 'article_text' in data.columns:
        st.write("Preview of the uploaded data:")
        st.dataframe(data.head())

        # Proses analisis sentimen
        st.write("Analyzing sentiment...")
        data['sentiment_score'] = data['article_text'].apply(analyze_sentiment)
        data['sentiment_category'] = data['sentiment_score'].apply(categorize_sentiment)

        # Menampilkan hasil analisis
        st.write("Sentiment analysis completed. Here are the results:")
        st.dataframe(data[['article_text', 'sentiment_score', 'sentiment_category']].head())

        # Visualisasi distribusi kategori sentimen
        st.write("Sentiment Category Distribution:")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(data=data, x='sentiment_category', palette='viridis', ax=ax)
        ax.set_title("Sentiment Category Distribution", fontsize=16)
        ax.set_xlabel("Sentiment Category", fontsize=14)
        ax.set_ylabel("Count", fontsize=14)
        st.pyplot(fig)

        # Visualisasi distribusi skor sentimen
        st.write("Sentiment Score Distribution:")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data['sentiment_score'], bins=20, kde=True, color='blue', ax=ax)
        ax.set_title("Sentiment Score Distribution", fontsize=16)
        ax.set_xlabel("Sentiment Score", fontsize=14)
        ax.set_ylabel("Frequency", fontsize=14)
        st.pyplot(fig)

        # Download hasil analisis
        csv = data.to_csv(index=False)
        st.download_button(
            label="Download Analysis Results",
            data=csv,
            file_name="sentiment_analysis_results.csv",
            mime="text/csv",
        )
    else:
        st.error("The uploaded file does not contain a column named 'article_text'.")
else:
    st.info("Please upload a CSV file to start the analysis.")
