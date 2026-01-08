import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Dark theme CSS
st.markdown("""
<style>
    .main {background-color: #0e1117;}
    .stMetric {background-color: #1f2937; border: 1px solid #374151;}
    h1, h2, h3 {color: #ffffff !important;}
    .stMarkdown {color: #e8e8e8;}
    .dataframe {background-color: #1a1d24;}
    .dataframe th {background-color: #2a2e3b; color: #ffffff;}
    .dataframe td {background-color: #1a1d24; color: #e8e8e8;}
    .stButton > button {background: linear-gradient(45deg, #1f77b4, #00d4aa); color: white;}
</style>
""", unsafe_allow_html=True)

st.title("🔍 DBSCAN Clustering Dashboard")

# Upload dataset
uploaded_file = st.file_uploader("📁 Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Dataset metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", len(df))
    col2.metric("Numeric Columns", df.select_dtypes(include=["float64", "int64"]).shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # Select numeric columns only
    numeric_df = df.select_dtypes(include=["float64", "int64"]).fillna(0)

    if numeric_df.shape[1] < 2:
        st.error("❌ Dataset must have at least 2 numeric columns for clustering.")
    else:
        st.subheader("🔧 Selected Features")
        st.success(f"✅ Using {numeric_df.shape[1]} numeric features")

        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_df)

        # Sidebar parameters
        st.sidebar.header("⚙️ DBSCAN Parameters")
        eps = st.sidebar.slider("EPS (Neighborhood radius)", 0.1, 5.0, 0.5, 0.1)
        min_samples = st.sidebar.slider("Min Samples", 1, 20, 5)

        # Scale toggle
        scale_features = st.sidebar.checkbox("🔄 Auto-scale Features", value=True)

        data = X_scaled if scale_features else numeric_df.values

        # DBSCAN model
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(data)

        df["Cluster"] = labels

        st.subheader("📈 Clustered Data Preview")
        preview_cols = ["Cluster"] + numeric_df.columns.tolist()[:3]
        st.dataframe(df[preview_cols].head(10), use_container_width=True)

        # Enhanced plot with dark theme
        st.subheader("🎨 Cluster Visualization")
        fig, ax = plt.subplots(figsize=(10, 8), facecolor='#1a1d24')
        unique_labels = sorted(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            if label == -1:
                color = 'lightgray'
                label_name = 'Noise'
            else:
                color = colors[i]
                label_name = f'Cluster {label}'
            
            mask = labels == label
            ax.scatter(numeric_df.iloc[:, 0][mask], numeric_df.iloc[:, 1][mask], 
                      c=[color], label=label_name, s=60, alpha=0.8, edgecolors='white')

        ax.set_facecolor('#0e1117')
        ax.set_xlabel(numeric_df.columns[0], fontsize=12, color='white')
        ax.set_ylabel(numeric_df.columns[1], fontsize=12, color='white')
        ax.set_title(f"DBSCAN Clustering (eps={eps:.2f}, min_samples={min_samples})", 
                    fontsize=14, color='white', pad=20)
        ax.legend(facecolor='#1a1d24', edgecolor='white')
        ax.grid(True, alpha=0.3, color='gray')
        plt.tight_layout()
        st.pyplot(fig)

        # FIXED Cluster info metrics
        st.subheader("📊 Cluster Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        num_clusters = len(set(labels) - {-1})
        col1.metric("Total Clusters", num_clusters)
        
        noise_count = list(labels).count(-1)
        col2.metric("Noise Points", noise_count)
        
        # Safe largest cluster
        positive_labels = labels[labels >= 0]
        largest_cluster = max(np.bincount(positive_labels)) if len(positive_labels) > 0 else 0
        col3.metric("Largest Cluster", largest_cluster)
        
        # Safe avg cluster size
        if num_clusters > 0:
            cluster_sizes = [np.sum(labels == i) for i in set(labels) if i >= 0]
            avg_size = f"{np.mean(cluster_sizes):.0f}"
        else:
            avg_size = "0"
        col4.metric("Avg Cluster Size", avg_size)

        # Cluster distribution
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        st.subheader("📈 Cluster Distribution")
        st.bar_chart(cluster_counts)

        # Download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Clustered Dataset",
            data=csv,
            file_name="dbscan_clustered_data.csv",
            mime="text/csv"
        )
