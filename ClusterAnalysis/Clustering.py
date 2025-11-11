import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes

import umap
import hdbscan
from sentence_transformers import SentenceTransformer

from scipy.sparse import hstack, csr_matrix

st.set_page_config(page_title="Cluster Analysis App", layout="wide")
st.title("Cluster Analysis App")

# ---------------------------
# Helper functions
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_sentence_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)

def build_text_features(df: pd.DataFrame,
                        text_cols: list[str],
                        method: str = "TF-IDF",
                        max_features: int = 20000,
                        tfidf_ngrams=(1,2),
                        st_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Return a dense np.ndarray for text features plus the fitted vectorizer/model for reference."""
    if not text_cols:
        return None, None

    # Combine selected text columns into one field
    text_series = df[text_cols].astype(str).fillna("").apply(lambda row: " ".join(row.values), axis=1)

    if method == "TF-IDF":
        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=tfidf_ngrams)
        X_text_sparse = vectorizer.fit_transform(text_series)
        return X_text_sparse, vectorizer  # sparse output
    else:
        # Sentence embeddings (dense)
        model = load_sentence_model(st_model_name)
        embeddings = model.encode(text_series.to_list(), show_progress_bar=False, normalize_embeddings=True)
        return embeddings.astype(np.float32), model  # dense output

def reduce_dimensions(X, reducer_type: str = "UMAP", n_components: int = 50, metric: str = "cosine", random_state: int = 42):
    """Reduce dimensions for stability/perf; returns dense np.ndarray."""
    if X is None:
        return None
    if reducer_type == "SVD":
        # SVD works for sparse TF-IDF
        svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        X_red = svd.fit_transform(X)
        return X_red
    else:
        # UMAP works for both dense/sparse; cast to dense if sparse and n is small enough
        # UMAP with cosine is good for text/embeddings
        if hasattr(X, "toarray"):  # sparse
            # Only toarray if safe; otherwise, let UMAP handle sparse input
            try:
                X_dense = X.toarray()
            except Exception:
                X_dense = X
        else:
            X_dense = X

        umap_model = umap.UMAP(
            n_neighbors=15,
            min_dist=0.0,
            n_components=n_components,
            metric=metric,
            random_state=random_state
        )
        return umap_model.fit_transform(X_dense)

def one_hot_categorical(df: pd.DataFrame, cat_cols: list[str]):
    """One-hot encode categoricals -> sparse matrix."""
    if not cat_cols:
        return None, None
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    X_cat = ohe.fit_transform(df[cat_cols].astype(str))
    return X_cat, ohe

def scale_numeric(df: pd.DataFrame, num_cols: list[str]):
    """Impute + scale numeric -> dense array."""
    if not num_cols:
        return None, None
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    X_num = pipe.fit_transform(df[num_cols])
    return X_num, pipe

def to_dense_if_sparse(X):
    if X is None:
        return None
    if hasattr(X, "toarray"):
        return X.toarray()
    return X

def hstack_safe(mats):
    """Horizontally stack list of matrices/arrays that may be None, sparse, or dense.
       Returns dense np.ndarray for downstream algorithms like HDBSCAN.
    """
    mats = [m for m in mats if m is not None]
    if not mats:
        return None
    # If any is sparse, stack sparsely first, then convert to dense
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
        # Convert to dense for HDBSCAN/UMAP downstream
        try:
            X = X.toarray()
        except Exception:
            pass
        return X
    else:
        return np.hstack(mats)

# ---------------------------
# UI Controls
# ---------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(data.head(), use_container_width=True)

        # Column pickers
        all_num = list(data.select_dtypes(include=["float64", "int64", "float32", "int32"]).columns)
        all_obj = list(data.select_dtypes(include=["object"]).columns)

        st.subheader("Select Features")
        c1, c2, c3 = st.columns(3)
        with c1:
            numerical_columns = st.multiselect("Numerical columns", all_num)
        with c2:
            # Categorical are intended for one-hot encoding
            categorical_columns = st.multiselect("Categorical columns", all_obj)
        with c3:
            # Text columns will be embedded (not label-encoded)
            text_columns = st.multiselect("Text columns (e.g., reviews, comments)", [c for c in all_obj if c not in categorical_columns])

        if len(numerical_columns) == 0 and len(categorical_columns) == 0 and len(text_columns) == 0:
            st.error("Please select at least one numerical, categorical, or text column.")
            st.stop()

        st.subheader("Text Embedding & Dimensionality Reduction")
        with st.expander("Text & Dimensionality Settings", expanded=True):
            text_method = st.radio("Text embedding method", ["TF-IDF", "Sentence Embeddings"], index=1)
            max_features = st.slider("TF-IDF max_features", 1000, 100_000, 20_000, step=1000, help="Only used for TF-IDF")
            tfidf_ngrams = st.select_slider("TF-IDF n-grams", options=[(1,1),(1,2),(1,3)], value=(1,2), help="Only used for TF-IDF")

            reduce_text = st.checkbox("Reduce text dimensions", value=True)
            text_reducer_type = st.radio("Text reducer", ["UMAP", "SVD"], index=0, help="SVD recommended for very large sparse TF-IDF matrices")
            text_components = st.slider("Reduced text dimensions", 5, 256, 50)

        st.subheader("Categorical & Numeric Reduction (optional)")
        with st.expander("Categorical/Numeric Dimensionality", expanded=False):
            reduce_cat = st.checkbox("Reduce one-hot categoricals (SVD)", value=True)
            cat_components = st.slider("Categorical reduced dims", 2, 128, 10)
            reduce_num = st.checkbox("Reduce numeric (UMAP)", value=False)
            num_components = st.slider("Numeric reduced dims", 2, 32, 8)

        st.subheader("Clustering Algorithm")
        algo = st.selectbox("Choose algorithm", ["HDBSCAN", "DBSCAN", "KMeans", "KModes (categorical only)", "KPrototypes (mixed)"], index=0)

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
            min_cluster_size = st.number_input("HDBSCAN min_cluster_size", min_value=2, value=15, step=1)
            min_samples_hdb = st.number_input("HDBSCAN min_samples (0 for default)", min_value=0, value=0, step=1)
            cluster_sel_epsilon = st.number_input("HDBSCAN cluster_selection_epsilon", min_value=0.0, value=0.0, step=0.1, format="%.4f")
            metric_hdb = st.selectbox("HDBSCAN metric", ["euclidean", "manhattan", "cosine"], index=2)

        # ---------------------------
        # Build Feature Matrix
        # ---------------------------
        # Numeric
        X_num, num_pipe = scale_numeric(data, numerical_columns)

        # Categorical
        X_cat, ohe = one_hot_categorical(data, categorical_columns)
        if X_cat is not None and reduce_cat:
            X_cat = TruncatedSVD(n_components=cat_components, random_state=42).fit_transform(X_cat)

        # Text
        X_text, text_model = build_text_features(
            data,
            text_columns,
            method=text_method,
            max_features=max_features,
            tfidf_ngrams=tfidf_ngrams
        )

        if X_text is not None and reduce_text:
            # If TF-IDF (sparse), SVD is fast; UMAP is great in general (cosine).
            reducer = "SVD" if (text_method == "TF-IDF" and text_reducer_type == "SVD") else "UMAP"
            metric = "cosine" if reducer == "UMAP" else "euclidean"
            X_text = reduce_dimensions(X_text, reducer_type=reducer, n_components=text_components, metric=metric)

        # Optionally reduce numeric dims
        if X_num is not None and reduce_num:
            X_num = reduce_dimensions(X_num, reducer_type="UMAP", n_components=num_components, metric="euclidean")

        # Combine all features into a dense matrix for general-purpose clustering
        X_all = hstack_safe([X_num, X_cat, X_text])
        if X_all is None:
            st.error("No features could be constructed from your selections.")
            st.stop()

        # ---------------------------
        # Fit Clustering
        # ---------------------------
        clusters = None
        cluster_prob = None

        if algo == "KMeans":
            st.info("Using KMeans on constructed feature space")
            km = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
            clusters = km.fit_predict(X_all)

        elif algo == "KModes (categorical only)":
            if len(categorical_columns) == 0:
                st.error("KModes requires categorical columns only.")
                st.stop()
            st.info("Using KModes on selected categorical columns")
            kmodes = KModes(n_clusters=num_clusters, random_state=42, init="Huang")
            clusters = kmodes.fit_predict(data[categorical_columns].astype(str))

        elif algo == "KPrototypes (mixed)":
            if len(categorical_columns) == 0 or len(numerical_columns) == 0:
                st.error("KPrototypes requires both numeric and categorical columns.")
                st.stop()
            st.info("Using KPrototypes on original selected columns (not embedded text)")
            # Use original mixed columns (excluding text embeddings)
            selected_columns = numerical_columns + categorical_columns
            kproto = KPrototypes(n_clusters=num_clusters, random_state=42, init="Huang")
            # Categorical indices relative to the selected_columns list
            cat_idx = [selected_columns.index(c) for c in categorical_columns]
            clusters = kproto.fit_predict(data[selected_columns].astype(object).values, categorical=cat_idx)

        elif algo == "DBSCAN":
            st.info("Using DBSCAN on constructed feature space")
            db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
            clusters = db.fit_predict(X_all)

        elif algo == "HDBSCAN":
            st.info("Using HDBSCAN on constructed feature space")
            hdb = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=(None if min_samples_hdb == 0 else int(min_samples_hdb)),
                cluster_selection_epsilon=cluster_sel_epsilon,
                metric=metric_hdb,
                prediction_data=True
            )
            clusters = hdb.fit_predict(X_all)
            # Soft cluster probabilities (when available)
            try:
                cluster_prob = hdb.probabilities_
            except Exception:
                cluster_prob = None

        # ---------------------------
        # Output
        # ---------------------------
        data_out = data.copy()
        data_out["Cluster"] = clusters

        if cluster_prob is not None:
            data_out["Cluster_Prob"] = cluster_prob

        st.subheader("Clustered Data")
        st.write("Note: HDBSCAN/DBSCAN label '-1' means 'noise' (unassigned).")
        st.dataframe(data_out, use_container_width=True)

        # 2D Visualization using UMAP projection of the combined feature space
        st.subheader("2D Cluster Visualization (UMAP on feature space)")
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
            sns.scatterplot(x=proj[:, 0], y=proj[:, 1], hue=data_out["Cluster"].astype(str), palette="viridis", s=20)
            plt.xlabel("UMAP-1"); plt.ylabel("UMAP-2")
        else:
            # simple 3D plot fallback
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            ax = plt.axes(projection="3d")
            p = ax.scatter(proj[:,0], proj[:,1], proj[:,2], c=data_out["Cluster"], cmap="viridis", s=15)
            ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2"); ax.set_zlabel("UMAP-3")
            plt.colorbar(p, ax=ax, fraction=0.03, pad=0.1)

        plt.title(f"Clusters via {algo}")
        st.pyplot(plt)

        # Download
        st.download_button(
            "Download clustered CSV",
            data_out.to_csv(index=False).encode("utf-8"),
            file_name="clustered_output.csv",
            mime="text/csv"
        )

        with st.expander("What changed vs. your original app?"):
            st.markdown("""
- **Text columns** are now supported via **TF‑IDF** or **Sentence Embeddings**.
- **HDBSCAN** and **DBSCAN** are available and recommended for irregular cluster shapes and variable cluster sizes.
- **Dimensionality reduction** (UMAP/SVD) improves cluster quality and speed, especially for text.
- **2D UMAP projection** provides a consistent visualization even when you don’t have two numeric columns to plot.
- Categorical columns use **One‑Hot Encoding** (optionally reduced via SVD) to avoid ordinal artifacts.
            """)

    except Exception as e:
        st.error(f"An error occurred: {e}")

st.write("""
## Instructions
1. Upload a CSV file containing your data.
2. Select numerical, categorical, and/or text columns.
3. Choose the embedding and reduction options for text (TF‑IDF or Sentence Embeddings).
4. Pick a clustering algorithm (HDBSCAN recommended for reviews/text).
5. Inspect the clustered data and the 2D UMAP plot. Download the results if needed.
""")
