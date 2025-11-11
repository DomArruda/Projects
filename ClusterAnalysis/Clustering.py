# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN

from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes

from sklearn.feature_extraction.text import TfidfVectorizer
import umap
import hdbscan
from sentence_transformers import SentenceTransformer

from scipy.sparse import hstack, csr_matrix

# ----------------------------------
# Streamlit setup
# ----------------------------------
st.set_page_config(page_title="Cluster Analysis App", layout="wide")
st.title("Cluster Analysis App")

# ----------------------------------
# Helpers
# ----------------------------------
@st.cache_resource(show_spinner=False)
def load_sentence_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def build_text_features(df: pd.DataFrame,
                        text_cols: list[str],
                        method: str = "TF-IDF",
                        max_features: int = 20000,
                        tfidf_ngrams=(1,2),
                        st_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Return matrix for text features (sparse for TF-IDF, dense for embeddings) + fitted model/vectorizer."""
    if not text_cols:
        return None, None

    text_series = df[text_cols].astype(str).fillna("").apply(lambda row: " ".join(row.values), axis=1)

    if method == "TF-IDF":
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=tfidf_ngrams)
        X_text_sparse = vectorizer.fit_transform(text_series)
        return X_text_sparse, vectorizer  # sparse
    else:
        model = load_sentence_model(st_model_name)
        embeddings = model.encode(text_series.to_list(), show_progress_bar=False, normalize_embeddings=True)
        return embeddings.astype(np.float32), model  # dense

def reduce_dimensions(X,
                      reducer_type: str = "UMAP",
                      n_components: int = 50,
                      metric: str = "cosine",
                      random_state: int = 42):
    """Reduce dimensions; returns dense np.ndarray."""
    if X is None:
        return None

    if reducer_type == "SVD":
        svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        return svd.fit_transform(X)
    else:
        # UMAP generally handles dense or sparse
        X_in = X
        try:
            # If sparse and small enough, toarray can help speed
            if hasattr(X, "toarray"):
                X_in = X.toarray()
        except Exception:
            pass

        umap_model = umap.UMAP(
            n_neighbors=15,
            min_dist=0.0,
            n_components=n_components,
            metric=metric,
            random_state=random_state
        )
        return umap_model.fit_transform(X_in)

def one_hot_categorical(df: pd.DataFrame, cat_cols: list[str]):
    if not cat_cols:
        return None, None
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    X_cat = ohe.fit_transform(df[cat_cols].astype(str))
    return X_cat, ohe

def scale_numeric(df: pd.DataFrame, num_cols: list[str]):
    if not num_cols:
        return None, None
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    X_num = pipe.fit_transform(df[num_cols])
    return X_num, pipe

def hstack_safe(mats):
    """Horizontally stack a list of matrices/arrays that may be None, sparse, or dense -> dense ndarray."""
    mats = [m for m in mats if m is not None]
    if not mats:
        return None

    # If any is sparse, stack sparsely first then to dense
    is_any_sparse = any(hasattr(m, "tocsr") or hasattr(m, "toarray") for m in mats)
    if is_any_sparse:
        sparse_parts = []
        dense_parts = []
        for m in mats:
            if hasattr(m, "tocsr") or hasattr(m, "toarray") or isinstance(m, csr_matrix):
                sparse_parts.append(m if hasattr(m, "tocsr") else csr_matrix(m))
            else:
                dense_parts.append(m)

        X = hstack(sparse_parts) if sparse_parts else None
        if dense_parts:
            dense_concat = np.hstack(dense_parts)
            if X is not None:
                X = hstack([X, csr_matrix(dense_concat)])
            else:
                X = csr_matrix(dense_concat)

        try:
            X = X.toarray()
        except Exception:
            pass
        return X
    else:
        return np.hstack(mats)

def default_min_cluster_size(n_rows: int) -> int:
    return max(5, int(0.01 * n_rows))  # ~1% of dataset, minimum 5

# ----------------------------------
# File upload
# ----------------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV to begin.")
    st.stop()

# ----------------------------------
# Main logic in try/except
# ----------------------------------
try:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(data.head(), use_container_width=True)

    # Column discovery
    all_num = list(data.select_dtypes(include=["float64", "int64", "float32", "int32"]).columns)
    all_obj = list(data.select_dtypes(include=["object"]).columns)

    st.subheader("Select Features")
    c1, c2, c3 = st.columns(3)
    with c1:
        numerical_columns = st.multiselect("Numerical columns", all_num)
    with c2:
        categorical_columns = st.multiselect("Categorical columns", all_obj)
    with c3:
        text_columns = st.multiselect(
            "Text columns (e.g., reviews, comments)",
            [c for c in all_obj if c not in categorical_columns]
        )

    if len(numerical_columns) == 0 and len(categorical_columns) == 0 and len(text_columns) == 0:
        st.error("Please select at least one numerical, categorical, or text column.")
        st.stop()

    # ----------------------------------
    # Text embedding & reduction settings
    # ----------------------------------
    st.subheader("Text Embedding & Dimensionality Reduction")
    with st.expander("Text & Dimensionality Settings", expanded=True):
        text_method = st.radio("Text embedding method", ["TF-IDF", "Sentence Embeddings"], index=1)
        max_features = st.slider("TF-IDF max_features", 1000, 100_000, 20_000, step=1000, help="Only used for TF-IDF")
        tfidf_ngrams = st.select_slider("TF-IDF n-grams", options=[(1,1), (1,2), (1,3)], value=(1,2), help="Only used for TF-IDF")

        reduce_text = st.checkbox("Reduce text dimensions", value=True)
        text_reducer_type = st.radio("Text reducer", ["UMAP", "SVD"], index=0,
                                     help="SVD recommended for very large sparse TF-IDF; UMAP (cosine) for embeddings/TF-IDF.")
        text_components = st.slider("Reduced text dimensions", 5, 256, 50)

    # ----------------------------------
    # Optional reductions for other modalities
    # ----------------------------------
    st.subheader("Categorical & Numeric Reduction (optional)")
    with st.expander("Categorical/Numeric Dimensionality", expanded=False):
        reduce_cat = st.checkbox("Reduce one-hot categoricals (SVD)", value=True)
        cat_components = st.slider("Categorical reduced dims", 2, 128, 10)
        reduce_num = st.checkbox("Reduce numeric (UMAP)", value=False)
        num_components = st.slider("Numeric reduced dims", 2, 32, 8)

    # ----------------------------------
    # Algorithm selection (conditional)
    # ----------------------------------
    st.subheader("Clustering Algorithm")
    supported_algos = ["HDBSCAN", "DBSCAN", "KMeans"]
    only_cat = (len(numerical_columns) == 0 and len(text_columns) == 0 and len(categorical_columns) > 0)
    mixed_no_text = (len(numerical_columns) > 0 and len(categorical_columns) > 0 and len(text_columns) == 0)

    if only_cat:
        supported_algos += ["KModes (categorical only)"]
    if mixed_no_text:
        supported_algos += ["KPrototypes (mixed)"]

    algo = st.selectbox("Choose algorithm", supported_algos, index=0)

    # Algo params
    if algo == "KMeans":
        num_clusters = st.slider("Number of clusters (KMeans)", 2, 20, 5)
    elif algo == "KModes (categorical only)":
        num_clusters = st.slider("Number of clusters (KModes)", 2, 20, 5)
    elif algo == "KPrototypes (mixed)":
        num_clusters = st.slider("Number of clusters (KPrototypes)", 2, 20, 5)
    elif algo == "DBSCAN":
        eps = st.number_input("DBSCAN eps", min_value=0.01, value=0.5, step=0.05)
        min_samples = st.number_input("DBSCAN min_samples", min_value=1, value=5, step=1)
    elif algo == "HDBSCAN":
        n_rows = len(data)
        default_min = default_min_cluster_size(n_rows)
        min_cluster_size = st.number_input("HDBSCAN min_cluster_size", min_value=2, value=int(default_min), step=1)
        min_samples_hdb = st.number_input("HDBSCAN min_samples (0 for default)", min_value=0, value=0, step=1)
        cluster_sel_epsilon = st.number_input("HDBSCAN cluster_selection_epsilon", min_value=0.0, value=0.0, step=0.1, format="%.4f")
        metric_hdb = st.selectbox("HDBSCAN metric", ["euclidean", "manhattan", "cosine"], index=2)

    # ----------------------------------
    # Build feature matrix
    # ----------------------------------
    X_num, num_pipe = scale_numeric(data, numerical_columns)

    X_cat, ohe = one_hot_categorical(data, categorical_columns)
    if X_cat is not None and reduce_cat:
        X_cat = TruncatedSVD(n_components=min(cat_components, max(2, min(X_cat.shape) - 1)), random_state=42).fit_transform(X_cat)

    X_text, text_model = build_text_features(
        data,
        text_columns,
        method=text_method,
        max_features=max_features,
        tfidf_ngrams=tfidf_ngrams
    )

    if X_text is not None and reduce_text:
        reducer = "SVD" if (text_method == "TF-IDF" and text_reducer_type == "SVD") else "UMAP"
        metric = "cosine" if reducer == "UMAP" else "euclidean"
        # Cap components to valid range when SVD on sparse with few features
        n_comp = text_components
        if reducer == "SVD" and hasattr(X_text, "shape"):
            n_comp = min(text_components, max(2, min(X_text.shape) - 1))
        X_text = reduce_dimensions(X_text, reducer_type=reducer, n_components=n_comp, metric=metric)

    if X_num is not None and reduce_num:
        X_num = reduce_dimensions(X_num, reducer_type="UMAP", n_components=num_components, metric="euclidean")

    X_all = hstack_safe([X_num, X_cat, X_text])
    if X_all is None:
        st.error("No features could be constructed from your selections.")
        st.stop()

    # ----------------------------------
    # Fit clustering
    # ----------------------------------
    clusters = None
    cluster_prob = None

    if algo == "KMeans":
        st.info("Using KMeans on constructed feature space")
        km = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
        clusters = km.fit_predict(X_all)

    elif algo == "KModes (categorical only)":
        if len(categorical_columns) == 0 or len(text_columns) > 0 or len(numerical_columns) > 0:
            st.error("KModes requires only categorical columns.")
            st.stop()
        st.info("Using KModes on selected categorical columns")
        kmodes = KModes(n_clusters=num_clusters, random_state=42, init="Huang")
        clusters = kmodes.fit_predict(data[categorical_columns].astype(str))

    elif algo == "KPrototypes (mixed)":
        if len(categorical_columns) == 0 or len(numerical_columns) == 0 or len(text_columns) > 0:
            st.error("KPrototypes requires numeric + categorical (no text).")
            st.stop()
        st.info("Using KPrototypes on original selected columns (not embedded text)")
        selected_columns = numerical_columns + categorical_columns
        kproto = KPrototypes(n_clusters=num_clusters, random_state=42, init="Huang")
        cat_idx = [selected_columns.index(c) for c in categorical_columns]
        clusters = kproto.fit_predict(data[selected_columns].astype(object).values, categorical=cat_idx)

    elif algo == "DBSCAN":
        st.info("Using DBSCAN on constructed feature space")
        db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
        clusters = db.fit_predict(X_all)

    elif algo == "HDBSCAN":
        st.info("Using HDBSCAN on constructed feature space")
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=int(min_cluster_size),
            min_samples=(None if min_samples_hdb == 0 else int(min_samples_hdb)),
            cluster_selection_epsilon=float(cluster_sel_epsilon),
            metric=metric_hdb,
            prediction_data=True
        )
        clusters = hdb.fit_predict(X_all)
        try:
            cluster_prob = hdb.probabilities_
        except Exception:
            cluster_prob = None

    # ----------------------------------
    # Output
    # ----------------------------------
    data_out = data.copy()
    data_out["Cluster"] = clusters
    if cluster_prob is not None:
        data_out["Cluster_Prob"] = cluster_prob

    st.subheader("Clustered Data")
    st.write("Note: HDBSCAN/DBSCAN label '-1' means 'noise' (unassigned).")
    st.dataframe(data_out, use_container_width=True)

    # 2D/3D projection for visualization on combined feature space
    st.subheader("2D/3D Cluster Visualization (UMAP on feature space)")
    proj_components = st.slider("UMAP plot components", 2, 3, 2)
    proj_metric = st.selectbox("UMAP plot metric", ["euclidean", "cosine"], index=1)
    proj = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=proj_components,
        metric=proj_metric,
        random_state=42
    ).fit_transform(X_all)

    plt.figure(figsize=(10, 6))
    if proj_components == 2:
        sns.scatterplot(x=proj[:, 0], y=proj[:, 1],
                        hue=data_out["Cluster"].astype(str),
                        palette="viridis", s=20)
        plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
    else:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        ax = plt.axes(projection="3d")
        p = ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2],
                       c=data_out["Cluster"], cmap="viridis", s=15)
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2"); ax.set_zlabel("UMAP-3")
        plt.colorbar(p, ax=ax, fraction=0.03, pad=0.1)

    plt.title(f"Clusters via {algo}")
    st.pyplot(plt)

    # Download results
    st.download_button(
        "Download clustered CSV",
        data_out.to_csv(index=False).encode("utf-8"),
        file_name="clustered_output.csv",
        mime="text/csv"
    )

    # Optional dev notes
    if st.checkbox("Show technical notes / changelog", value=False):
        with st.expander("What changed vs. the original app?"):
            st.markdown("""
- **Text columns** supported via **TF‑IDF** or **Sentence Embeddings** (MiniLM default).
- **HDBSCAN**/**DBSCAN** added for shape-aware clustering and noise handling.
- **Dimensionality reduction** (UMAP/SVD) for stability and performance.
- **One‑Hot Encoding** for categoricals (optional SVD to reduce cardinality).
- **UMAP projection** for visualization even without 2 numeric columns.
- **Conditional algorithms**: show **KModes** only for categorical-only; **KPrototypes** only for mixed numeric+categorical without text.
            """)

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.exception(e)

# ----------------------------------
# Instructions
# ----------------------------------
st.write("""
## Instructions
1. Upload a CSV file containing your data.
2. Select numerical, categorical, and/or text columns.
3. Choose the embedding and reduction options for text (TF‑IDF or Sentence Embeddings).
4. Pick a clustering algorithm (**HDBSCAN** is recommended for reviews/text).
5. Inspect the clustered data and the UMAP plot. Download the results if needed.
""")
