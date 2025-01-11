import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline

# Initialize default sentiment analysis pipeline
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis")  # Using the default pipeline

sentiment_analyzer = load_sentiment_pipeline()

# Function to analyze sentiment
def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return result['label'], result['score']

# Title
st.title("Sentiment Analysis App (Default Model)")

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    # Read uploaded file
    data = pd.read_csv(uploaded_file)
    st.write("Preview of the uploaded data:")
    st.dataframe(data.head())

    # Dropdown to select column for sentiment analysis
    text_column = st.selectbox(
        "Select the column to analyze sentiment:",
        options=data.columns,
        help="Choose the column containing text data for sentiment analysis."
    )

    if text_column:
        # Button to start analysis
        if st.button("Start Sentiment Analysis"):
            # Perform sentiment analysis
            st.write(f"Analyzing sentiment for column: {text_column}")
            sentiments = data[text_column].apply(lambda x: analyze_sentiment(str(x)))
            data['sentiment_category'] = sentiments.apply(lambda x: x[0])
            data['sentiment_score'] = sentiments.apply(lambda x: x[1])

            # Display results
            st.write("Sentiment analysis completed. Here are the results:")
            st.dataframe(data[[text_column, 'sentiment_category', 'sentiment_score']].head())

            # Plot sentiment category distribution
            st.write("Sentiment Category Distribution:")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.countplot(data=data, x='sentiment_category', palette='viridis', ax=ax)
            ax.set_title("Sentiment Category Distribution", fontsize=16)
            ax.set_xlabel("Sentiment Category", fontsize=14)
            ax.set_ylabel("Count", fontsize=14)
            st.pyplot(fig)

            # Plot sentiment score distribution
            st.write("Sentiment Score Distribution:")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(data['sentiment_score'], bins=20, kde=True, color='blue', ax=ax)
            ax.set_title("Sentiment Score Distribution", fontsize=16)
            ax.set_xlabel("Sentiment Score", fontsize=14)
            ax.set_ylabel("Frequency", fontsize=14)
            st.pyplot(fig)

            # Download analysis results
            csv = data.to_csv(index=False)
            st.download_button(
                label="Download Analysis Results",
                data=csv,
                file_name="sentiment_analysis_results.csv",
                mime="text/csv",
            )
else:
    st.info("Please upload a CSV file to start the analysis.")
