import streamlit as st
from bs4 import BeautifulSoup
import numpy as np
import re
import pandas as pd
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
import plotly.express as px
import pyodbc
from collections import Counter
import os
import io
import urllib.parse
import time
from requests.exceptions import RequestException

# Optional dependency: hdbscan
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except Exception:
    HDBSCAN_AVAILABLE = False

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
# Small, fast sentence embedding model (loaded via transformers only)
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
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
    """Load and cache the tokenizer for the sentiment model"""
    if boolean:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                'nlptown/bert-base-multilingual-uncased-sentiment'
            )
            return tokenizer
        except Exception as e:
            st.error(f"Error loading tokenizer: {str(e)}")
            return None

# ---- Embedding model (transformers-based) ----
@st.cache_resource
def load_embedding_components(boolean):
    """
    Load and cache embedding tokenizer and model using transformers only
    (no sentence-transformers lib required).
    """
    if boolean:
        try:
            tok = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
            mdl = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
            mdl.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            mdl.to(device)
            return tok, mdl, device
        except Exception as e:
            st.error(f"Error loading embedding model: {str(e)}")
            return None, None, torch.device('cpu')
    return None, None, torch.device('cpu')

# Load models/tokenizers
model = load_model(True)
tokenizer = load_tokenizer(True)
embed_tokenizer, embed_model, embed_device = load_embedding_components(True)

def sentiment_score(review):
    """Calculate sentiment score for a given text"""
    if not review or not isinstance(review, str):
        return 3  # Neutral for empty or non-string inputs
    if model is None or tokenizer is None:
        return 3

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
    parsed_url = urllib.parse.urlparse(url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"

    headers = {
        'User-Agent': USER_AGENT,
        'Accept': 'text/html,application/xhtml+xml,application/xml',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    link_list = [url]  # Start with the original URL

    is_tripadvisor = "tripadvisor.com" in url.lower() and "restaurant_review" in url.lower()

    if is_tripadvisor:
        if "Reviews-" in url:
            parts = url.split("Reviews")
            if len(parts) == 2:
                base_before = parts[0] + "Reviews"
                if "-or" in parts[1]:
                    after_parts = parts[1].split("-", 2)
                    if len(after_parts) >= 3:
                        base_after = "-" + after_parts[2]
                    else:
                        base_after = parts[1]
                else:
                    base_after = parts[1]
                for i in range(2, num_pages + 1):
                    offset = (i - 1) * 10
                    next_page = f"{base_before}-or{offset}{base_after}"
                    link_list.append(next_page)
        else:
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
        query_params = urllib.parse.parse_qs(parsed_url.query)

        def base_query_without(*keys):
            return "&".join([f"{k}={v[0]}" for k, v in query_params.items() if k not in keys])

        pagination_formats = [
            lambda i: f"{base_url}?{base_query_without('page', 'start')}&page={i}" if query_params else f"{base_url}?page={i}",
            lambda i: f"{base_url}?{base_query_without('page', 'start')}&start={i*10}" if query_params else f"{base_url}?start={i*10}",
            lambda i: f"{base_url}/page/{i}/" + (f"?{urllib.parse.urlencode(query_params, doseq=True)}" if query_params else ""),
        ]

        if 'page=' in url or 'start=' in url or '/page/' in url:
            if 'page=' in url:
                pattern_index = 0
            elif 'start=' in url:
                pattern_index = 1
            else:
                pattern_index = 2
            for i in range(2, num_pages + 1):
                link_list.append(pagination_formats[pattern_index](i))
         for i in range(2, num_pages + 1):
                link_list.append(pagination_formatsi)

    df_list = []
    total_reviews = 0

    review_patterns = [
        re.compile('.*comment.*'),
        re.compile('.*review.*'),
        re.compile('.*feedback.*'),
        re.compile('.*testimonial.*'),
    ]

    if progress_bar:
        progress_bar.progress(0)

    for idx, page_url in enumerate(link_list):
        if idx > 0:
            time.sleep(1.5)

        try:
            st.sidebar.info(f"Scraping page {idx+1}: {page_url}")
            r = requests.get(page_url, headers=headers, timeout=15)
            r.raise_for_status()

            soup = BeautifulSoup(r.text, 'html.parser')

            reviews = []

            if is_tripadvisor:
                review_bodies = soup.find_all('div', attrs={'data-test-target': 'review-body'})
                if review_bodies:
                    for body in review_bodies:
                        text_divs = body.select('.biGQs._P.pZUbB.KxBGd span.JguWG')
                        if text_divs:
                            for div in text_divs:
                                review_text = div.get_text(strip=True)
                                if review_text:
                                    reviews.append(review_text)

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

            if not reviews:
                for tag in ['div', 'p', 'span']:
                    for class_name in ['review-content', 'review-text', 'comment-content', 'comment-text']:
                        results = soup.find_all(tag, class_=class_name)
                        if results:
                            reviews.extend([result.text.strip() for result in results])

            reviews = [r for r in reviews if r.strip()]
            reviews = list(dict.fromkeys(reviews))  # dedupe

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
    out[review_column] = out[review_column].astype(str)
    return out

def reviews_csv(file_obj):
    """Process reviews from a CSV file"""
    try:
        data = pd.read_csv(file_obj, header="infer")
        data_columns = list(data.columns)

        data_columns_with_none = ['None Selected'] + data_columns

        selected_column = st.selectbox('Which column contains the reviews?', data_columns_with_none)
        totalReviews = len(data)
        st.text(f'Total number of reviews detected: {totalReviews}')

        max_reviews = min(totalReviews, 1000)  # performance cap
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

    sentiment_scores = sorted(pd.Series(dataframe[score_col].unique()).dropna().astype(int).tolist())
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
            dataframe.to_excel(writer, sheet_name='Sentiment Scores', index=False)

            workbook = writer.book
            worksheet = writer.sheets['Sentiment Scores']

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

            word_freq_df = get_word_frequency_by_sentiment(dataframe, review_col=review_column, score_col=score_column)
            word_freq_df.to_excel(writer, sheet_name='Word Frequencies', index=False)

            for score in sorted(dataframe[score_column].unique()):
                score_words = word_freq_df[word_freq_df['sentiment_score'] == score]
                if score_words.empty:
                    continue

                sheet_name = f"Score {score} Words"
                score_words.to_excel(writer, sheet_name=sheet_name, index=False)
                score_worksheet = writer.sheets[sheet_name]

                word_chart = workbook.add_chart({'type': 'bar'})
                top_words = score_words.nlargest(10, 'frequency')
                word_chart.add_series({
                    'name': f'Frequency for Score {score}',
                    'categories': [sheet_name, 1, 1, min(10, len(top_words)), 1],
                    'values':     [sheet_name, 1, 2, min(10, len(top_words)), 2],
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

# ----------------------------
# Embedding + HDBSCAN Clustering
# ----------------------------

def _mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling on token embeddings, masking padding tokens."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts

@st.cache_data(show_spinner=False)
def compute_embeddings_cached(texts: tuple) -> np.ndarray:
    """Cacheable wrapper to compute embeddings for a tuple of texts."""
    texts_list = list(texts)
    return compute_embeddings(texts_list)

def compute_embeddings(texts, batch_size: int = 64) -> np.ndarray:
    """Compute sentence embeddings using a small transformer model (no extra deps)."""
    if embed_model is None or embed_tokenizer is None:
        st.error("Embedding model failed to load.")
        return np.zeros((len(texts), 384), dtype=np.float32)

    all_vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch = [t if isinstance(t, str) and t.strip() else "" for t in batch]
            enc = embed_tokenizer(
                batch, padding=True, truncation=True, max_length=256,
                return_tensors='pt'
            )
            enc = {k: v.to(embed_device) for k, v in enc.items()}
            outputs = embed_model(**enc)
            pooled = _mean_pooling(outputs.last_hidden_state, enc['attention_mask'])
            # Normalize embeddings (optional but often helps clustering)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_vecs.append(pooled.cpu().numpy().astype(np.float32))
    return np.vstack(all_vecs) if all_vecs else np.zeros((0, 384), dtype=np.float32)

def pca_reduce_2d(X: np.ndarray) -> np.ndarray:
    """Reduce to 2D via PCA using NumPy SVD (no sklearn dependency)."""
    if X.shape[1] < 2 or X.shape[0] < 2:
        return np.hstack([X, np.zeros((X.shape[0], max(0, 2 - X.shape[1])))])

    Xc = X - X.mean(axis=0, keepdims=True)
    # economy SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:2]  # top 2 PCs
    return Xc @ components.T

def auto_hdbscan_fit(embeddings: np.ndarray):
    """
    Auto-tune HDBSCAN by scanning min_cluster_size and selecting the model
    with best weighted persistence minus a small penalty for noise.
    """
    if not HDBSCAN_AVAILABLE:
        return None

    n = embeddings.shape[0]
    if n < 5:
        return None

    # Candidate min_cluster_size values (scale with n; keep small for speed)
    base_candidates = [5, 8, 10, 12, 15, 20, 25, 30]
    scaled = [max(5, int(n * r)) for r in [0.01, 0.015, 0.02, 0.03, 0.05]]
    candidates = sorted(set([c for c in base_candidates + scaled if c <= max(50, n // 2)]))

    best = None
    best_score = -1e9

    for mcs in candidates:
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=mcs,
                min_samples=None,
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True,
                core_dist_n_jobs=1  # stay single-threaded in shared envs
            )
            labels = clusterer.fit_predict(embeddings)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters == 0:
                # All noise: skip
                continue
            noise_frac = float(np.mean(labels == -1))

            # Persistence heuristic (higher is better)
            # cluster_persistence_ length equals number of clusters
            if hasattr(clusterer, "cluster_persistence_") and len(clusterer.cluster_persistence_) > 0:
                persistence = float(np.mean(clusterer.cluster_persistence_))
            else:
                persistence = 0.0

            # Composite objective: prefer stable clusters with less noise,
            # and mildly reward more clusters (diversity) without over-fragmentation.
            score = persistence - 0.30 * noise_frac + 0.02 * n_clusters

            if score > best_score:
                best = (clusterer, labels, score)
                best_score = score
        except Exception:
            continue

    return best[0] if best else None

def build_cluster_summary(df: pd.DataFrame, labels: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    """Return a per-cluster summary: size, exemplar (most confident review), and sample keywords."""
    df_tmp = df.copy()
    df_tmp['cluster'] = labels
    df_tmp['confidence'] = probabilities

    summaries = []
    clusters = [c for c in sorted(set(labels)) if c != -1]
    for c in clusters:
        sub = df_tmp[df_tmp['cluster'] == c]
        size = len(sub)
        # exemplar = row with max confidence
        exemplar_row = sub.iloc[sub['confidence'].values.argmax()]
        exemplar_text = exemplar_row[review_column]

        # top words (quick heuristic)
        top_words = get_most_frequent_words(sub[review_column].tolist(), max_words=8)
        top_words_str = ", ".join([w for w, _ in top_words])

        summaries.append({
            'cluster': int(c),
            'size': int(size),
            'exemplar': exemplar_text[:2000],  # keep UI safe
            'keywords': top_words_str
        })

    return pd.DataFrame(summaries).sort_values('size', ascending=False)

def render_clustering_ui(dataframe: pd.DataFrame, title_suffix: str):
    """Shared clustering UI used in CSV/Web/SQL tabs."""
    st.subheader(f"🔎 Clustering (HDBSCAN) — {title_suffix}")

    if dataframe is None or dataframe.empty:
        st.info("No data to cluster yet.")
        return

    if not HDBSCAN_AVAILABLE:
        st.warning(
            "HDBSCAN is not installed. To enable clustering, please install it in your environment:\n\n"
            "`pip install hdbscan`"
        )
        return

    # Allow user to choose text column to embed; default to canonical review column
    candidate_cols = [review_column] + [c for c in dataframe.columns if c != review_column]
    text_col = st.selectbox("Text column to cluster:", candidate_cols, index=0)

    max_rows = len(dataframe)
    n_rows = int(st.number_input("How many rows to cluster?", min_value=10, max_value=max_rows, value=min(500, max_rows)))
    run_cluster = st.button("Run clustering")

    if not run_cluster:
        return

    texts = dataframe[text_col].astype(str).iloc[:n_rows].tolist()

    with st.spinner("Embedding reviews..."):
        # Cache by text content
        emb = compute_embeddings_cached(tuple(texts))

    if emb.shape[0] < 5:
        st.error("Not enough data to cluster.")
        return

    with st.spinner("Auto-tuning and fitting HDBSCAN..."):
        clusterer = auto_hdbscan_fit(emb)

    if clusterer is None:
        st.error("HDBSCAN could not form stable clusters. Try more data or different content.")
        return

    labels = clusterer.labels_
    probs = getattr(clusterer, "probabilities_", np.ones_like(labels, dtype=float))
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_rate = float(np.mean(labels == -1))

    st.success(f"Found **{n_clusters} cluster(s)** | Noise: **{noise_rate:.0%}** | min_cluster_size: **{clusterer.min_cluster_size}**")

    # Attach to a copy for display/download
    out = dataframe.iloc[:n_rows].copy()
    out['cluster'] = labels
    out['cluster_confidence'] = probs

    # Summary
    summary_df = build_cluster_summary(out, labels, probs)
    if not summary_df.empty:
        st.write("**Cluster summary:**")
        st.dataframe(summary_df, use_container_width=True)

    # 2D plot via PCA
    coords2d = pca_reduce_2d(emb)
    plot_df = pd.DataFrame({
        'x': coords2d[:, 0],
        'y': coords2d[:, 1],
        'cluster': labels.astype(int),
        'confidence': probs,
        'text': [t[:200] for t in texts]
    })
    # Map noise to string label for color legend clarity
    plot_df['cluster_str'] = plot_df['cluster'].map(lambda c: 'noise' if c == -1 else f'c{c}')

    st.write("**2D projection (PCA):**")
    fig = px.scatter(
        plot_df, x='x', y='y',
        color='cluster_str',
        size='confidence',
        hover_data={'text': True, 'cluster_str': True, 'x': False, 'y': False, 'confidence': ':.2f'},
        title='HDBSCAN Clusters (PCA projection)',
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-cluster word bars
    if n_clusters > 0:
        st.write("**Top words per cluster:**")
        tabs = st.tabs([f"Cluster {c}" for c in sorted(set(labels)) if c != -1])
        clusters_sorted = [c for c in sorted(set(labels)) if c != -1]
        for i, tab in enumerate(tabs):
            c = clusters_sorted[i]
            with tab:
                sub = out[out['cluster'] == c]
                top_words = get_most_frequent_words(sub[review_column].tolist(), max_words=20)
                if top_words:
                    wf_df = pd.DataFrame(top_words, columns=['word', 'frequency'])
                    fig_bar = px.bar(wf_df, x='word', y='frequency', title=f"Most Common Words — Cluster {c}")
                    st.plotly_chart(fig_bar, use_container_width=True)
                    st.dataframe(sub[[review_column, 'cluster_confidence']].rename(
                        columns={review_column: 'review'}).head(25), use_container_width=True)
                else:
                    st.info("No significant words found for this cluster.")

    # Download with cluster labels
    st.download_button(
        "Download clustered data (CSV)",
        out.to_csv(index=False).encode('utf-8'),
        file_name="reviews_with_clusters.csv"
    )

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

        review_df = None
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

                st.subheader("Export Results")
                c1, c2 = st.columns(2)
                with c1:
                    review_csv = review_df.to_csv(index=False).encode('utf-8')
                    st.download_button('Download CSV Report', review_csv, 'sentiment_analysis.csv')
                with c2:
                    excel_data = save_to_excel(review_df)
                    if excel_data:
                        st.download_button('Download Excel Report (with charts)', excel_data, 'sentiment_analysis.xlsx')

        # Clustering UI for CSV data
        if review_df is not None and not review_df.empty:
            with st.expander("🔬 Cluster the CSV reviews (HDBSCAN)"):
                render_clustering_ui(review_df, title_suffix="CSV")

    # ---- Tab 3: Web Scraping ----
    with tab3:
        st.subheader("Web Scraping")
        uploaded_link = st.text_input('Enter website URL to scrape reviews:', placeholder='https://example.com/reviews')
        num_pages = st.number_input('Number of pages to scrape:', min_value=1, max_value=10, value=1)
        scrape_button = st.button('Start Scraping')

        scraped_df = None
        if scrape_button and uploaded_link:
            if not uploaded_link.startswith(('http://', 'https://')):
                uploaded_link = 'https://' + uploaded_link

            st.info(f'Scraping reviews from {uploaded_link}. This may take a moment...')
            progress_bar = st.progress(0)

            with st.spinner('Scraping in progress...'):
                try:
                    scraped_df = reviews_scrape(uploaded_link, num_pages, progress_bar)

                    if scraped_df is not None and not scraped_df.empty:
                        st.success(f'Successfully scraped {len(scraped_df)} reviews!')
                        scraped_df['sentiment'] = scraped_df[score_column].map(lambda x: SENTIMENT_DICT[str(x)])
                        st.dataframe(scraped_df, use_container_width=True)

                        col1, col2 = st.columns(2)
                        with col1:
                            fig = px.histogram(
                                scraped_df, x=score_column, color=score_column,
                                labels={score_column: 'Sentiment Score'},
                                title='Distribution of Sentiment Scores',
                                color_discrete_map={1: "red", 2: "orange", 3: "yellow", 4: "lightgreen", 5: "green"}
                            )
                            fig.update_traces(marker_line_color='white', marker_line_width=1.0)
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            fig2 = px.pie(
                                scraped_df, names='sentiment', title='Sentiment Distribution',
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
                            review_csv = scraped_df.to_csv(index=False).encode('utf-8')
                            st.download_button('Download CSV Report', review_csv, 'scraped_sentiment.csv')
                        with c2:
                            excel_data = save_to_excel(scraped_df)
                            if excel_data:
                                st.download_button('Download Excel Report (with charts)', excel_data, 'scraped_sentiment.xlsx')
                except Exception as e:
                    st.error(f"Error during scraping: {str(e)}")

        if scraped_df is not None and not scraped_df.empty:
            with st.expander("🔬 Cluster the scraped reviews (HDBSCAN)"):
                render_clustering_ui(scraped_df, title_suffix="Scraped")

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

        sql_df = None
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
                    sql_df = SQL_scrape(query, conn)
                    conn.close()

                    if sql_df is not None and not sql_df.empty:
                        sql_df['sentiment'] = sql_df[score_column].map(lambda x: SENTIMENT_DICT[str(x)])

                        st.success('Query executed successfully!')
                        st.dataframe(sql_df, use_container_width=True)

                        col1, col2 = st.columns(2)
                        with col1:
                            fig = px.histogram(
                                sql_df, x=score_column, color=score_column,
                                labels={score_column: 'Sentiment Score'},
                                title='Distribution of Sentiment Scores',
                                color_discrete_map={1: "red", 2: "orange", 3: "yellow", 4: "lightgreen", 5: "green"}
                            )
                            fig.update_traces(marker_line_color='white', marker_line_width=1.0)
                            st.plotly_chart(fig, use_container_width=True)

                        with col2:
                            fig2 = px.pie(
                                sql_df, names='sentiment', title='Sentiment Distribution',
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
                            review_csv = sql_df.to_csv(index=False).encode('utf-8')
                            st.download_button('Download CSV Report', review_csv, 'sql_sentiment.csv')
                        with c2:
                            excel_data = save_to_excel(sql_df)
                            if excel_data:
                                st.download_button('Download Excel Report (with charts)', excel_data, 'sql_sentiment.xlsx')
            except Exception as e:
                st.error(f"Database error: {str(e)}")

        if sql_df is not None and not sql_df.empty:
            with st.expander("🔬 Cluster the SQL reviews (HDBSCAN)"):
                render_clustering_ui(sql_df, title_suffix="SQL")

# Run the app
if __name__ == "__main__":
