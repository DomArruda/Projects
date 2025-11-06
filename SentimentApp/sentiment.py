import streamlit as st
from bs4 import BeautifulSoup
import numpy as np
import re
import pandas as pd
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.express as px
import pyodbc
from collections import Counter
import os
import io
import urllib.parse
import time
from requests.exceptions import RequestException

st.set_page_config(page_title="Jim Bot - Sentiment Analysis", page_icon="🤖", layout="wide")

# Canonical column names used throughout the app
review_column = "review"
score_column = "score"

# Constants
MAX_TOKEN_LENGTH = 512
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
SENTIMENT_DICT = {
    '1': 'Very Negative',
    '2': 'Negative',
    '3': 'Neutral',
    '4': 'Positive',
    '5': 'Very Positive'
}
SAMPLE_DATA_PATH = 'AmazonProductReviews.csv'  # Fixed path (unused, kept for compatibility)

@st.cache_resource
def load_model(boolean):
    """Load and cache the sentiment analysis model"""
    if boolean:
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                'nlptown/bert-base-multilingual-uncased-sentiment'
            )
            model.eval()
            return model
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None

@st.cache_resource
def load_tokenizer(boolean):
    """Load and cache the tokenizer"""
    if boolean:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                'nlptown/bert-base-multilingual-uncased-sentiment'
            )
            return tokenizer
        except Exception as e:
            st.error(f"Error loading tokenizer: {str(e)}")
            return None

# Load model and tokenizer
model = load_model(True)
tokenizer = load_tokenizer(True)

def sentiment_score(review):
    """Calculate sentiment score for a given text"""
    if not review or not isinstance(review, str):
        return 3  # Neutral for empty or non-string inputs
    if model is None or tokenizer is None:
        # If loading failed, keep app usable with neutral score
        return 3

    # Truncate to max token length (by characters, tokenizer will also truncate by tokens)
    review = review[-MAX_TOKEN_LENGTH:]

    try:
        inputs = tokenizer(review, return_tensors='pt', truncation=True, max_length=MAX_TOKEN_LENGTH)
        with torch.no_grad():
            result = model(**inputs)
            score = int(torch.argmax(result.logits)) + 1
        return score
    except Exception as e:
        st.warning(f"Error processing sentiment: {str(e)}")
        return 3  # Neutral on errors

def reviews_scrape(url, num_pages, progress_bar=None):
    """
    Scrape reviews from a website with improved error handling and flexibility

    Args:
        url: Base URL to scrape
        num_pages: Number of pages to scrape
        progress_bar: Optional Streamlit progress bar

    Returns:
        DataFrame with reviews and sentiment scores
    """
    # Parse the URL to handle pagination correctly
    parsed_url = urllib.parse.urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"

    # Prepare headers to avoid being blocked
    headers = {
        'User-Agent': USER_AGENT,
        'Accept': 'text/html,application/xhtml+xml,application/xml',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    # Prepare page URLs based on common pagination patterns
    link_list = [url]  # Start with the original URL

    # Check if this is a TripAdvisor URL
    is_tripadvisor = "tripadvisor.com" in url.lower() and "restaurant_review" in url.lower()

    if is_tripadvisor:
        # Extract base URL parts
        if "Reviews-" in url:
            parts = url.split("Reviews")
            if len(parts) == 2:
                base_before = parts[0] + "Reviews"
                # Check if there's already a page indicator like -or15-
                if "-or" in parts[1]:
                    after_parts = parts[1].split("-", 2)
                    if len(after_parts) >= 3:
                        base_after = "-" + after_parts[2]
                    else:
                        base_after = parts[1]
                else:
                    base_after = parts[1]

                # Create URLs for additional pages
                for i in range(2, num_pages + 1):
                    offset = (i - 1) * 10  # TripAdvisor uses offsets of 10 per page
                    next_page = f"{base_before}-or{offset}{base_after}"
                    link_list.append(next_page)
        else:
            # Fallback to basic URL manipulation
            for i in range(2, num_pages + 1):
                offset = (i - 1) * 10
                if ".html" in url:
                    next_page = url.replace(".html", f"-or{offset}.html")
                elif ".htm" in url:
                    next_page = url.replace(".htm", f"-or{offset}.htm")
                else:
                    next_page = f"{url}-or{offset}"
                link_list.append(next_page)
    else:
        # For other websites, try common pagination patterns
        query_params = urllib.parse.parse_qs(parsed_url.query)

        # Helper to rebuild base query without certain keys
        def base_query_without(*keys):
            return "&".join([f"{k}={v[0]}" for k, v in query_params.items() if k not in keys])

        # Different pagination formats
        pagination_formats = [
            lambda i: f"{base_url}?{base_query_without('page', 'start')}&page={i}" if query_params else f"{base_url}?page={i}",
            lambda i: f"{base_url}?{base_query_without('page', 'start')}&start={i*10}" if query_params else f"{base_url}?start={i*10}",
            lambda i: f"{base_url}/page/{i}/" + (f"?{urllib.parse.urlencode(query_params, doseq=True)}" if query_params else ""),
        ]

        # Use the original URL format if it has pagination indicators
        if 'page=' in url or 'start=' in url or '/page/' in url:
            if 'page=' in url:
                pattern_index = 0
            elif 'start=' in url:
                pattern_index = 1
            else:
                pattern_index = 2
            for i in range(2, num_pages + 1):  # Start from page 2
                link_list.append(pagination_formatsi)
        else:
            # If no pattern is detected, try the first format
            for i in range(2, num_pages + 1):
                link_list.append(pagination_formatsi)

    # Initialize results
    df_list = []
    total_reviews = 0

    # Common review container classes/IDs
    review_patterns = [
        re.compile('.*comment.*'),
        re.compile('.*review.*'),
        re.compile('.*feedback.*'),
        re.compile('.*testimonial.*'),
    ]

    # Update progress bar if provided
    if progress_bar:
        progress_bar.progress(0)

    # Scrape each page
    for idx, page_url in enumerate(link_list):
        # Add delay to avoid being blocked
        if idx > 0:
            time.sleep(1.5)  # Slightly longer delay, helpful for TripAdvisor too

        try:
            st.sidebar.info(f"Scraping page {idx+1}: {page_url}")
            r = requests.get(page_url, headers=headers, timeout=15)
            r.raise_for_status()  # Raise exception for 4XX/5XX responses

            soup = BeautifulSoup(r.text, 'html.parser')

            # Try different patterns to find review elements
            reviews = []

            # TripAdvisor-specific approach
            if is_tripadvisor:
                # Look for review-body elements
                review_bodies = soup.find_all('div', attrs={'data-test-target': 'review-body'})
                if review_bodies:
                    for body in review_bodies:
                        text_divs = body.select('.biGQs._P.pZUbB.KxBGd span.JguWG')
                        if text_divs:
                            for div in text_divs:
                                review_text = div.get_text(strip=True)
                                if review_text:
                                    reviews.append(review_text)

                # Alternative selectors
                if not reviews:
                    review_cards = soup.find_all('div', attrs={'data-automation': 'reviewCard'})
                    for card in review_cards:
                        body = card.find('div', attrs={'data-test-target': 'review-body'})
                        if body:
                            review_text = body.get_text(strip=True)
                            if review_text:
                                reviews.append(review_text)

                if not reviews:
                    selectors = [
                        'div[data-test-target="review-body"] span',
                        '.biGQs._P.pZUbB.KxBGd span',
                        '.prw_reviews_text_summary_hsx div',
                        '.partial_entry',
                        '.review-container .reviewText',
                    ]
                    for selector in selectors:
                        elements = soup.select(selector)
                        if elements:
                            for el in elements:
                                review_text = el.get_text(strip=True)
                                if review_text and len(review_text) > 20:
                                    reviews.append(review_text)

            # Generic approach if no reviews found
            if not reviews:
                for pattern in review_patterns:
                    results = soup.find_all('p', {'class': pattern})
                    if results:
                        reviews.extend([result.text.strip() for result in results])
                        break

                    results = soup.find_all('div', {'class': pattern})
                    if results:
                        reviews.extend([result.text.strip() for result in results])
                        break

                    results = soup.find_all(id=pattern)
                    if results:
                        reviews.extend([result.text.strip() for result in results])
                        break

            # Fallback common class names
            if not reviews:
                for tag in ['div', 'p', 'span']:
                    for class_name in ['review-content', 'review-text', 'comment-content', 'comment-text']:
                        results = soup.find_all(tag, class_=class_name)
                        if results:
                            reviews.extend([result.text.strip() for result in results])

            # Filter out empty reviews and duplicates
            reviews = [r for r in reviews if r.strip()]
            reviews = list(dict.fromkeys(reviews))  # Deduplicate preserving order

            if reviews:
                df = pd.DataFrame(reviews, columns=[review_column])
                df[score_column] = df[review_column].apply(sentiment_score)
                df['page'] = idx + 1
                df['url'] = page_url
                df_list.append(df)
                total_reviews += len(reviews)

                st.sidebar.success(f"Page {idx+1}: Found {len(reviews)} reviews")
            else:
                st.sidebar.warning(f"Page {idx+1}: No reviews found")

            # Update progress
            if progress_bar:
                progress_bar.progress((idx + 1) / len(link_list))

        except RequestException as e:
            st.warning(f"Error scraping page {page_url}: {str(e)}")
            continue
        except Exception as e:
            st.warning(f"Unexpected error processing page {page_url}: {str(e)}")
            continue

    if len(df_list) == 0:
        st.error("No reviews found. The website may have a different structure or could be blocking scrapers.")
        return pd.DataFrame(columns=[review_column, score_column, 'page', 'url'])

    merged = pd.concat(df_list, ignore_index=True)
    st.success(f"Successfully scraped {total_reviews} reviews from {len(df_list)} pages")
    return merged

def _normalize_selected_column(df: pd.DataFrame, selected_column: str) -> pd.DataFrame:
    """
    Return a copy of df where the selected_column is renamed to the canonical review_column
    and coerced to string. Leaves original df unchanged.
    """
    out = df.copy()
    if selected_column != review_column:
        if selected_column not in out.columns:
            raise KeyError(f"Selected column '{selected_column}' not found in data.")
        out.rename(columns={selected_column: review_column}, inplace=True)
    # Ensure review text is str
    out[review_column] = out[review_column].astype(str)
    return out

def reviews_csv(file_obj):
    """Process reviews from a CSV file"""
    try:
        data = pd.read_csv(file_obj, header="infer")
        data_columns = list(data.columns)

        # Insert 'None Selected' at the beginning
        data_columns_with_none = ['None Selected'] + data_columns

        selected_column = st.selectbox('Which column contains the reviews?', data_columns_with_none)
        totalReviews = len(data)
        st.text(f'Total number of reviews detected: {totalReviews}')

        max_reviews = min(totalReviews, 1000)  # Limit to 1000 reviews for performance
        numReviews = int(st.number_input('How many reviews should I analyze?',
                                         min_value=1, max_value=max_reviews, value=min(100, max_reviews)))

        if selected_column == 'None Selected':
            st.warning("Please select a column containing reviews.")
            return None

        # Normalize and subset
        data_subset = _normalize_selected_column(data.iloc[:numReviews], selected_column)

        with st.spinner('Analyzing sentiment. This may take a moment...'):
            progress_bar = st.progress(0)
            total_rows = len(data_subset)

            # Process in chunks for better UX
            sentiment_scores = []
            chunk_size = max(1, min(100, max(1, total_rows // 10)))

            for i in range(0, total_rows, chunk_size):
                end_idx = min(i + chunk_size, total_rows)
                chunk = data_subset.iloc[i:end_idx]

                chunk_scores = chunk[review_column].apply(sentiment_score)
                sentiment_scores.extend(chunk_scores)

                progress_bar.progress(end_idx / total_rows)

            data_subset[score_column] = sentiment_scores
            data_subset['sentiment'] = data_subset[score_column].map(lambda x: SENTIMENT_DICT[str(x)])

        return data_subset

    except Exception as e:
        st.error(f"Error processing CSV: {str(e)}")
        return None

def SQL_scrape(query, conn):
    """Process reviews from SQL database"""
    cursor = None
    try:
        data = pd.read_sql(query, conn)
        data_columns = list(data.columns)

        data_columns_with_none = ['None Selected'] + data_columns
        selected_column = st.selectbox('Which column contains the reviews?', data_columns_with_none)
        totalReviews = len(data)
        st.text(f'Total number of reviews detected: {totalReviews}')

        max_reviews = min(totalReviews, 1000)
        numReviews = int(st.number_input('How many reviews should I analyze?',
                                         min_value=1, max_value=max_reviews, value=min(100, max_reviews)))

        if selected_column == 'None Selected':
            st.warning("Please select a column containing reviews.")
            return None

        data_subset = _normalize_selected_column(data.iloc[:numReviews], selected_column)

        with st.spinner('Analyzing sentiment. This may take a moment...'):
            progress_bar = st.progress(0)
            total_rows = len(data_subset)

            sentiment_scores = []
            chunk_size = max(1, min(100, max(1, total_rows // 10)))

            for i in range(0, total_rows, chunk_size):
                end_idx = min(i + chunk_size, total_rows)
                chunk = data_subset.iloc[i:end_idx]

                chunk_scores = chunk[review_column].apply(sentiment_score)
                sentiment_scores.extend(chunk_scores)

                progress_bar.progress(end_idx / total_rows)

            data_subset[score_column] = sentiment_scores
            data_subset['sentiment'] = data_subset[score_column].map(lambda x: SENTIMENT_DICT[str(x)])

        return data_subset

    except Exception as e:
        st.error(f"Error processing SQL query: {str(e)}")
        return None
    finally:
        # Ensure connection resources are properly closed
        if cursor:
            cursor.close()

def get_most_frequent_words(reviews, min_word_length=3, max_words=20, stopwords=None):
    """Get most frequent words in reviews with additional filtering"""
    if stopwords is None:
        stopwords = {
            'the', 'and', 'is', 'in', 'it', 'to', 'that', 'was', 'for', 'net', 'on',
            'with', 'as', 'this', 'at', 'from', 'an', 'by', 'are', 'be', 'or',
            'has', 'had', 'have', 'not', 'but', 'what', 'all', 'were', 'when',
            'we', 'they', 'you', 'she', 'his', 'her', 'their', 'our', 'who',
            'which', 'will', 'more', 'no', 'if', 'out', 'so', 'up', 'very'
        }

    words = []
    for review in reviews:
        if not isinstance(review, str):
            continue
        review_words = [word.lower() for word in re.findall(r'\b[a-zA-Z]+\b', review)]
        review_words = [word for word in review_words if len(word) >= min_word_length and word not in stopwords]
        words.extend(review_words)

    word_counts = Counter(words)
    most_common_words = word_counts.most_common(max_words)
    return most_common_words

def get_word_frequency_by_sentiment(dataframe, review_col=review_column, score_col=score_column):
    """Create a dataframe with word frequencies partitioned by sentiment score"""
    if dataframe.empty:
        return pd.DataFrame(columns=['sentiment_score', 'word', 'frequency'])

    # Get unique sentiment scores
    sentiment_scores = sorted(pd.Series(dataframe[score_col].unique()).dropna().astype(int).tolist())

    # Build frequency rows
    word_freq_data = []
    for score in sentiment_scores:
        sentiment_reviews = dataframe[dataframe[score_col] == score][review_col].tolist()
        word_freqs = get_most_frequent_words(sentiment_reviews, max_words=30)
        for word, freq in word_freqs:
            word_freq_data.append({
                'sentiment_score': score,
                'word': word,
                'frequency': freq
            })

    return pd.DataFrame(word_freq_data)

def save_to_excel(dataframe):
    """Save analysis results to Excel with error handling"""
    try:
        output = io.BytesIO()

        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Write the sentiment dataframe to the Excel file
            dataframe.to_excel(writer, sheet_name='Sentiment Scores', index=False)

            workbook = writer.book
            worksheet = writer.sheets['Sentiment Scores']

            # Summary worksheet
            worksheet_summary = workbook.add_worksheet('Summary')
            worksheet_summary.write(0, 0, 'Sentiment')
            worksheet_summary.write(0, 1, 'Count')
            worksheet_summary.write(0, 2, 'Percentage')

            sentiment_counts = dataframe[score_column].value_counts().sort_index()
            total_reviews = len(dataframe)

            for i, (sentiment, count) in enumerate(sentiment_counts.items()):
                worksheet_summary.write(i+1, 0, f"{sentiment} ({SENTIMENT_DICT[str(sentiment)]})")
                worksheet_summary.write(i+1, 1, int(count))
                worksheet_summary.write(i+1, 2, f"{count/total_reviews:.2%}")

            chart = workbook.add_chart({'type': 'column'})
            chart.add_series({
                'name': 'Sentiment Distribution',
                'categories': ['Summary', 1, 0, len(sentiment_counts), 0],
                'values': ['Summary', 1, 1, len(sentiment_counts), 1],
            })
            chart.set_title({'name': 'Sentiment Distribution'})
            chart.set_x_axis({'name': 'Sentiment'})
            chart.set_y_axis({'name': 'Count'})
            worksheet_summary.insert_chart('E2', chart, {'x_scale': 1.5, 'y_scale': 1.5})

            # Word frequencies by sentiment
            word_freq_df = get_word_frequency_by_sentiment(dataframe, review_col=review_column, score_col=score_column)
            word_freq_df.to_excel(writer, sheet_name='Word Frequencies', index=False)

            # Per-score sheets with charts
            for score in sorted(dataframe[score_column].unique()):
                score_words = word_freq_df[word_freq_df['sentiment_score'] == score]
                if score_words.empty:
                    continue

                sheet_name = f"Score {score} Words"
                score_words.to_excel(writer, sheet_name=sheet_name, index=False)
                score_worksheet = writer.sheets[sheet_name]

                word_chart = workbook.add_chart({'type': 'bar'})
                top_words = score_words.nlargest(10, 'frequency')
                # Excel coordinates: rows are 0-index in xlsxwriter; +1 to skip header.
                word_chart.add_series({
                    'name': f'Frequency for Score {score}',
                    'categories': [sheet_name, 1, 1, min(10, len(top_words)) , 1],
                    'values':     [sheet_name, 1, 2, min(10, len(top_words)) , 2],
                    'data_labels': {'value': True},
                })
                word_chart.set_title({'name': f'Most Common Words for Score {score}'})
                word_chart.set_x_axis({'name': 'Word'})
                word_chart.set_y_axis({'name': 'Frequency'})
                score_worksheet.insert_chart('E2', word_chart, {'x_scale': 1.5, 'y_scale': 1.5})

        output.seek(0)
        return output.getvalue()

    except Exception as e:
        st.error(f"Error creating Excel file: {str(e)}")
        return None

def main():
    """Main application function"""
    st.title('Jim Bot 🤖')
    st.markdown("_Sentiment analysis for your reviews_")

    tab1, tab2, tab3, tab4 = st.tabs(["Test Text", "CSV Upload", "Web Scraping", "SQL Connection"])

    # ---- Tab 1: Single Text ----
    with tab1:
        st.subheader("Test Single Review")
        test_text = st.text_area(
            "Write a mock-review here and I'll return a score from 1 (negative emotion) to 5 (positive emotion):",
            height=150
        )
        if test_text:
            try:
                with st.spinner('Analyzing...'):
                    sentScore = sentiment_score(test_text)
                    sentCategory = SENTIMENT_DICT[str(sentScore)]
                    color_map = {1: "🔴", 2: "🟠", 3: "🟡", 4: "🟢", 5: "🔵"}
                    st.markdown(f"### Sentiment Score: {sentScore} {color_map[sentScore]}")
                    st.markdown(f"**Category: {sentCategory}**")
            except Exception as e:
                st.error(f'Error processing the sentiment score: {str(e)}')

    # ---- Tab 2: CSV Upload ----
    with tab2:
        st.subheader("Analyze CSV File")
        uploaded_file = st.file_uploader("Upload a CSV file containing reviews", type=["csv"])

        if uploaded_file is not None:
            st.success('Successfully uploaded CSV!')
            review_df = reviews_csv(uploaded_file)

            if review_df is not None and not review_df.empty:
                st.dataframe(review_df, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    fig = px.histogram(
                        review_df, x=score_column, color=score_column,
                        labels={score_column: 'Sentiment Score'},
                        title='Distribution of Sentiment Scores',
                        color_discrete_map={1: "red", 2: "orange", 3: "yellow", 4: "lightgreen", 5: "green"}
                    )
                    fig.update_traces(marker_line_color='white', marker_line_width=1.0)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig2 = px.pie(
                        review_df, names='sentiment', title='Sentiment Distribution',
                        color='sentiment',
                        color_discrete_map={
                            "Very Negative": "red",
                            "Negative": "orange",
                            "Neutral": "yellow",
                            "Positive": "lightgreen",
                            "Very Positive": "green"
                        }
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                # Word frequency analysis
                st.subheader("Common Words by Sentiment")
                word_freq_df = get_word_frequency_by_sentiment(review_df, review_col=review_column, score_col=score_column)

                if not word_freq_df.empty:
                    st.dataframe(word_freq_df, use_container_width=True)

                sentiment_tabs = st.tabs([SENTIMENT_DICT[str(i)] for i in range(1, 6)])
                for i, tab in enumerate(sentiment_tabs, 1):
                    with tab:
                        sentiment_words = word_freq_df[word_freq_df['sentiment_score'] == i]
                        if not sentiment_words.empty:
                            c1, c2 = st.columns([2, 3])
                            with c1:
                                st.dataframe(sentiment_words, use_container_width=True)
                            with c2:
                                top_words = sentiment_words.nlargest(15, 'frequency')
                                fig = px.bar(top_words, x='word', y='frequency',
                                             title=f'Most Common Words in {SENTIMENT_DICT[str(i)]} Reviews')
                                st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(f"No {SENTIMENT_DICT[str(i)]} reviews found.")

                # Export
                st.subheader("Export Results")
                c1, c2 = st.columns(2)
                with c1:
                    review_csv = review_df.to_csv(index=False).encode('utf-8')
                    st.download_button('Download CSV Report', review_csv, 'sentiment_analysis.csv')
                with c2:
                    excel_data = save_to_excel(review_df)
                    if excel_data:
                        st.download_button('Download Excel Report (with charts)', excel_data, 'sentiment_analysis.xlsx')

    # ---- Tab 3: Web Scraping ----
    with tab3:
        st.subheader("Web Scraping")
        uploaded_link = st.text_input('Enter website URL to scrape reviews:', placeholder='https://example.com/reviews')
        num_pages = st.number_input('Number of pages to scrape:', min_value=1, max_value=10, value=1)
        scrape_button = st.button('Start Scraping')

        if scrape_button and uploaded_link:
            if not uploaded_link.startswith(('http://', 'https://')):
                uploaded_link = 'https://' + uploaded_link

            st.info(f'Scraping reviews from {uploaded_link}. This may take a moment...')
            progress_bar = st.progress(0)

            with st.spinner('Scraping in progress...'):
                try:
                    review_data = reviews_scrape(uploaded_link, num_pages, progress_bar)

                    if review_data is not None and not review_data.empty:
                        st.success(f'Successfully scraped {len(review_data)} reviews!')

                        review_data['sentiment'] = review_data[score_column].map(lambda x: SENTIMENT_DICT[str(x)])
                        st.dataframe(review_data, use_container_width=True)

                        col1, col2 = st.columns(2)
                        with col1:
                            fig = px.histogram(
                                review_data, x=score_column, color=score_column,
                                labels={score_column: 'Sentiment Score'},
                                title='Distribution of Sentiment Scores',
                                color_discrete_map={1: "red", 2: "orange", 3: "yellow", 4: "lightgreen", 5: "green"}
                            )
                            fig.update_traces(marker_line_color='white', marker_line_width=1.0)
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            fig2 = px.pie(
                                review_data, names='sentiment', title='Sentiment Distribution',
                                color='sentiment',
                                color_discrete_map={
                                    "Very Negative": "red",
                                    "Negative": "orange",
                                    "Neutral": "yellow",
                                    "Positive": "lightgreen",
                                    "Very Positive": "green"
                                }
                            )
                            st.plotly_chart(fig2, use_container_width=True)

                        # Export
                        st.subheader("Export Results")
                        c1, c2 = st.columns(2)
                        with c1:
                            review_csv = review_data.to_csv(index=False).encode('utf-8')
                            st.download_button('Download CSV Report', review_csv, 'scraped_sentiment.csv')
                        with c2:
                            excel_data = save_to_excel(review_data)
                            if excel_data:
                                st.download_button('Download Excel Report (with charts)', excel_data, 'scraped_sentiment.xlsx')
                except Exception as e:
                    st.error(f"Error during scraping: {str(e)}")

    # ---- Tab 4: SQL Connection ----
    with tab4:
        st.subheader("SQL Connection")
        col1, col2 = st.columns(2)
        with col1:
            server = st.text_input('Server name:', placeholder='SERVERNAME')
        with col2:
            database = st.text_input('Database name:', placeholder='DBNAME')

        auth_type = st.radio("Authentication type:", ["Windows Authentication", "SQL Server Authentication"])

        username = password = None
        if auth_type == "SQL Server Authentication":
            c1, c2 = st.columns(2)
            with c1:
                username = st.text_input('Username:', placeholder='username')
            with c2:
                password = st.text_input('Password:', type='password', placeholder='password')

        query = st.text_area('SQL Query (must return a column with review text):',
                             placeholder='SELECT review_text FROM reviews')

        connect_button = st.button('Connect and Run Query')

        if connect_button and server and database and query:
            try:
                if auth_type == "Windows Authentication":
                    connection_string = f'Driver={{SQL Server}};Server={server};Database={database};Trusted_Connection=yes;'
                else:
                    if not username or not password:
                        st.error("Username and password are required for SQL Server Authentication")
                        st.stop()
                    connection_string = f'Driver={{SQL Server}};Server={server};Database={database};UID={username};PWD={password};'

                with st.spinner('Connecting to database...'):
                    conn = pyodbc.connect(connection_string)

                    review_data = SQL_scrape(query, conn)
                    conn.close()

                    if review_data is not None and not review_data.empty:
                        review_data['sentiment'] = review_data[score_column].map(lambda x: SENTIMENT_DICT[str(x)])

                        st.success('Query executed successfully!')
                        st.dataframe(review_data, use_container_width=True)

                        col1, col2 = st.columns(2)
                        with col1:
                            fig = px.histogram(
                                review_data, x=score_column, color=score_column,
                                labels={score_column: 'Sentiment Score'},
                                title='Distribution of Sentiment Scores',
                                color_discrete_map={1: "red", 2: "orange", 3: "yellow", 4: "lightgreen", 5: "green"}
                            )
                            fig.update_traces(marker_line_color='white', marker_line_width=1.0)
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            fig2 = px.pie(
                                review_data, names='sentiment', title='Sentiment Distribution',
                                color='sentiment',
                                color_discrete_map={
                                    "Very Negative": "red",
                                    "Negative": "orange",
                                    "Neutral": "yellow",
                                    "Positive": "lightgreen",
                                    "Very Positive": "green"
                                }
                            )
                            st.plotly_chart(fig2, use_container_width=True)

                        st.subheader("Export Results")
                        c1, c2 = st.columns(2)
                        with c1:
                            review_csv = review_data.to_csv(index=False).encode('utf-8')
                            st.download_button('Download CSV Report', review_csv, 'sql_sentiment.csv')
                        with c2:
                            excel_data = save_to_excel(review_data)
                            if excel_data:
                                st.download_button('Download Excel Report (with charts)', excel_data, 'sql_sentiment.xlsx')
            except Exception as e:
                st.error(f"Database error: {str(e)}")

# Run the app
if __name__ == "__main__":
    main()
