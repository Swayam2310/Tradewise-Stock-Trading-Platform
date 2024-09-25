import pandas as pd
import streamlit as st
import requests

import plotly.express as px
from datetime import datetime
from textblob import TextBlob
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from collections import defaultdict

news_api_key = 'a3dc05159a6446d19d52933d7965f8cc'

# Load the FinBERT model and tokenizer
model_name = "yiyanghkust/finbert-tone"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


# Fetch news articles using News API
def fetch_news(api_key, query):
    url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&apiKey={api_key}"
    response = requests.get(url)
    news_data = response.json()

    # Debugging: Print the status code and response content
    print(f"News API Status Code: {response.status_code}")
    print(f"News API Response: {news_data}")

    if response.status_code == 200:
        articles = news_data.get('articles', [])
        refined_articles = []

        for article in articles:
            # Format the publishedAt date
            published_at = article.get('publishedAt')
            if published_at:
                formatted_date = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ").date()
            else:
                formatted_date = None

            refined_articles.append({
                'title': article.get('title'),
                'date': str(formatted_date),  # Convert date to string format YYYY-MM-DD
                'link': article.get('url'),
                'content': article.get('description')  # You can also use 'content' if available
            })

        return refined_articles  # Return a list of refined articles
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return []


# Perform sentiment analysis on fetched articles
@st.cache_data
def analyze_sentiment(articles):
    sentiments = []
    for article in articles:
        if 'title' in article:
            analysis = TextBlob(article['title'])
            sentiment_score = analysis.sentiment.polarity
            sentiments.append(sentiment_score)

            # Debugging: Print the sentiment score for each article
            print(f"Article: {article['title']}\nSentiment Score: {sentiment_score}\n")
        else:
            print("No title found in article.")

    return sentiments

@st.cache_data
def analyze_sentiment_finbert(articles):
    """
    Analyzes the sentiment of the provided articles using FinBERT.

    Args:
    - articles (list): A list of articles.

    Returns:
    - List of dictionaries containing date, headline, description, sentiment type, and score.
    """
    refined_articles = []

    for article in articles:
        # Get headline and description
        date = article.get('date')
        headline = article.get('title')
        description = article.get('content', '')  # Default to empty string if not available

        # Skip articles without a description
        if not description:
            continue

        # Perform sentiment analysis using FinBERT
        sentiment_result = sentiment_pipeline(description)[0]  # Analyze description for sentiment
        sentiment_type = sentiment_result['label']
        sentiment_score = sentiment_result['score']

        # Append the refined article with sentiment
        refined_articles.append({
            'date': date,
            'headline': headline,
            'description': description,
            'sentiment_type': sentiment_type,
            'sentiment_score': sentiment_score
        })

    return refined_articles  # Return a list of refined articles


def calculate_daily_average_sentiment(sentiment_results):
    """
    Calculates the average sentiment score for each day.

    Args:
    - sentiment_results (list): List of dictionaries containing date, headline, description, sentiment type, and score.

    Returns:
    - Dictionary with dates as keys and average sentiment scores as values.
    """
    daily_sentiment = defaultdict(list)

    # Aggregate sentiment scores by date
    for result in sentiment_results:
        date = result['date']
        sentiment_score = result['sentiment_score']
        daily_sentiment[date].append(sentiment_score)

    # Calculate the average sentiment score for each day
    daily_average_sentiment = {date: sum(scores) / len(scores) for date, scores in daily_sentiment.items()}

    return daily_average_sentiment


articles = fetch_news(news_api_key, st.session_state['ticker'])
sentiments = analyze_sentiment(articles)

# Display sentiment analysis results
if sentiments:
    st.write(f"Sentiment analysis for {st.session_state['ticker']}:")
    st.bar_chart(pd.DataFrame(sentiments, columns=["Sentiment"]))
else:
    st.write("No sentiment data available.")


if 'ticker' in st.session_state:
    if st.session_state['ticker'] == 'AAPL':
        # Load the CSV data
        apple_sentiment_file = 'apple_news_sentiment.csv'
        apple_df = pd.read_csv(apple_sentiment_file)

        # Convert 'date' column to datetime format
        apple_df['date'] = pd.to_datetime(apple_df['date'])

        # Sort by date if not already sorted
        apple_df = apple_df.sort_values(by='date')

        # Aggregate sentiment Scores
        daily_sentiment = apple_df.groupby('date').agg({
                            'score': 'mean'
                            }).reset_index()

        apple_price = st.session_state['retrieved_data']
        apple_price['Date'] = pd.to_datetime(apple_price['Date'])
        apple_price = apple_price.rename(columns={'Date': 'date'})
        apple_price = apple_price.sort_values(by='date')

        # Merge sentiment data with stock price data
        apple_price_df = pd.merge(apple_price, apple_df, on='date')

        correlation = apple_price_df['score'].corr(apple_price['Close'])

        # Determine the description based on the correlation value
        if correlation >= 0.8:
            description = "Very strong positive association"
        elif correlation >= 0.6:
            description = "Strong positive association"
        elif correlation >= 0.4:
            description = "Moderate positive association"
        elif correlation >= 0.2:
            description = "Weak positive association"
        elif correlation > 0:
            description = "Very weak positive association"
        elif correlation > -0.2:
            description = "Very weak negative association"
        elif correlation > -0.4:
            description = "Weak negative association"
        elif correlation > -0.6:
            description = "Moderate negative association"
        elif correlation > -0.8:
            description = "Strong negative association"
        elif correlation >= -1.0:
            description = "Very strong negative association"
        else:
            description = "Perfect negative association"

        # Display the correlation result and description
        st.markdown(f"### Correlation between Apple and News")
        st.write(f"The value of Correlation is: **{correlation:.4f}**")
        st.write(f"Which mean they have: {description}")

        fig = px.scatter(apple_price_df,
                         x='score',
                         y='Close',
                         labels={'Close': 'Stock Close Price', 'score': 'The level of Sentiment'},
                         trendline="ols")

        # Show plot
        st.plotly_chart(fig)
    else:
        st.warning(f"Since rate limit, we only retrieved some Apple related news. \
            There is no sentiment information for this stock {st.session_state['ticker']}.")
else:
    st.warning('No ticker selected, pls select ticker in Company Info.')