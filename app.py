# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from datetime import datetime

st.set_page_config(page_title="Reddit SEO Visualization", layout="wide")

# -----------------------
# Sidebar - Dataset options
# -----------------------
st.sidebar.title("Dataset Options")
dataset_choice = st.sidebar.selectbox(
    "Choose dataset",
    ["Recent r/SEO Reddit", "TOP 2024 r/TechSEO Reddit", "TOP 2024 r/SEO Reddit", "Upload your own"],
    index=1
)

uploaded_csv = None
uploaded_npy = None
if dataset_choice == "Upload your own":
    uploaded_csv = st.sidebar.file_uploader("Upload CSV", type="csv")
    uploaded_npy = st.sidebar.file_uploader("Upload NPY", type="npy")

# -----------------------
# Load data
# -----------------------
@st.cache_data
def load_data(dataset_choice, uploaded_csv=None, uploaded_npy=None):
    if dataset_choice == "TOP 2024 r/TechSEO Reddit":
        df = pd.read_csv("techseo-reddit.csv")
        embeddings = np.load("tech-all-embeddings.npy")
    elif dataset_choice == "TOP 2024 r/SEO Reddit":
        df = pd.read_csv("seo-reddit.csv")
        embeddings = np.load("seo-embeddings.npy")
    elif dataset_choice == "Recent r/SEO Reddit":
        df = pd.read_csv("recent-seo-reddit.csv")
        embeddings = np.load("recent-seo-all-embeddings.npy")
    elif dataset_choice == "Upload your own" and uploaded_csv and uploaded_npy:
        df = pd.read_csv(uploaded_csv)
        embeddings = np.load(uploaded_npy)
    else:
        st.error("Please upload both CSV and NPY for your dataset!")
        return None, None
    return df, embeddings

df, embeddings = load_data(dataset_choice, uploaded_csv, uploaded_npy)
if df is None:
    st.stop()

# Ensure necessary columns
df['numberOfComments'] = df.get('numberOfComments', 0).fillna(0)
df['upVoteRatio'] = df.get('upVoteRatio', 1.0)
df['upVotes'] = df.get('upVotes', 0)
df['createdAt'] = pd.to_datetime(df.get('createdAt', pd.Timestamp.now()), utc=True)

# -----------------------
# Sidebar - 3D Scatter options
# -----------------------
st.sidebar.title("3D Scatter Options")
method = st.sidebar.selectbox("Dimensionality Reduction Method", ["UMAP", "t-SNE", "PCA"])
dim = st.sidebar.slider("Dimensions", 2, 3, 3)
color_option = st.sidebar.selectbox("Color by", ["numberOfComments", "upVoteRatio", "upVotes", "createdAt"], index=2)

# -----------------------
# Sidebar - Marker size
# -----------------------
st.sidebar.title("Marker Size Options")
marker_mode = st.sidebar.radio("Marker size mode", ["Static", "Variable"])
marker_size_static = st.sidebar.slider("Static marker size", 1, 10, 2)
marker_size_variable_column = st.sidebar.selectbox("Variable size based on", ["numberOfComments", "upVotes"], index=0)

# -----------------------
# Sidebar - Timeframe filter
# -----------------------
st.sidebar.title("Timeframe Filter")
min_date = df['createdAt'].min().date()
max_date = df['createdAt'].max().date()
date_range = st.sidebar.date_input("Select date range", [min_date, max_date], min_value=min_date, max_value=max_date)

# -----------------------
# Filter by timeframe
# -----------------------
start = pd.to_datetime(date_range[0]).tz_localize("UTC")
end = pd.to_datetime(date_range[1]).tz_localize("UTC")
mask = (df['createdAt'] >= start) & (df['createdAt'] <= end)
df_filtered = df[mask].reset_index(drop=True)
embeddings_filtered = embeddings[mask.values]

st.write(f"Filtered dataset: {len(df_filtered)} rows")

# -----------------------
# Dimensionality reduction
# -----------------------
st.write(f"## 3D Scatter Plot ({method})")
if method == "UMAP":
    reducer = umap.UMAP(n_components=dim, random_state=42)
elif method == "t-SNE":
    reducer = TSNE(n_components=dim, random_state=42, init='pca')
elif method == "PCA":
    reducer = PCA(n_components=dim)

reduced = reducer.fit_transform(embeddings_filtered[:, :])
df_filtered['x'] = reduced[:, 0]
df_filtered['y'] = reduced[:, 1]
df_filtered['z'] = reduced[:, 2] if dim == 3 else 0

# -----------------------
# Color scaling
# -----------------------
if color_option == "numberOfComments":
    df_filtered['color'] = np.log1p(df_filtered['numberOfComments'])
elif color_option == "upVotes":
    df_filtered['color'] = np.log1p(df_filtered['upVotes'])
elif color_option == "upVoteRatio":
    df_filtered['color'] = df_filtered['upVoteRatio'].clip(lower=0.8)
    df_filtered['color'] = (df_filtered['color'] - 0.8) / (1 - 0.8)
elif color_option == "createdAt":
    timestamp = df_filtered['createdAt'].astype(np.int64)
    df_filtered['color'] = (timestamp - timestamp.min()) / (timestamp.max() - timestamp.min())

# -----------------------
# Marker size scaling
# -----------------------
min_size, max_size = 2, 80  

if marker_mode == "Static":
    df_filtered['marker_size'] = marker_size_static
else:
    values = df_filtered[marker_size_variable_column].fillna(0)
    if values.max() > 0:
        df_filtered['marker_size'] = (
            (values - values.min()) / (values.max() - values.min())
        ) * (max_size - min_size) + min_size
    else:
        df_filtered['marker_size'] = min_size

# -----------------------
# 3D scatter plot
# -----------------------
fig = px.scatter_3d(
    df_filtered,
    x='x', y='y', z='z',
    color='color',
    size='marker_size' if marker_mode == "Variable" else None,
    color_continuous_scale='Reds',
    hover_data={'category': True, 'shortsum': True, 'numberOfComments': True, 'upVotes': True, 'createdAt': True, 'x': False, 'y': False, 'z': False, 'color': False, 'marker_size': False}
)

# Update marker for static mode
if marker_mode == "Static":
    fig.update_traces(marker=dict(size=marker_size_static, opacity=0.8))
else:
    fig.update_traces(marker=dict(opacity=0.8))

fig.update_layout(
    scene=dict(
        xaxis=dict(backgroundcolor="black", gridcolor="gray", showbackground=True, zerolinecolor="gray"),
        yaxis=dict(backgroundcolor="black", gridcolor="gray", showbackground=True, zerolinecolor="gray"),
        zaxis=dict(backgroundcolor="black", gridcolor="gray", showbackground=True, zerolinecolor="gray")
    ),
    paper_bgcolor="black",
    plot_bgcolor="black",
    font=dict(color="white")
)
st.plotly_chart(fig, use_container_width=True)
