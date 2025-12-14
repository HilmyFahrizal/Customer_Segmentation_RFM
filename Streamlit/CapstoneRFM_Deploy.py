import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import datetime as dt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.metrics import davies_bouldin_score

st.set_page_config(page_title="Customer Analytics", layout="wide")

def preprocess_data(path):
    df_raw = pd.read_csv(path)

    df = df_raw[['Customer ID', 'InvoiceDate', 'Quantity', 'Price', 'Invoice']].copy()
    df = df.dropna(subset=['Customer ID'])
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]
    df = df.drop_duplicates()

    df['Revenue'] = df['Quantity'] * df['Price']
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    return df

def plot_top_customer(df, top_n=10):
    customer_revenue = df.groupby('Customer ID')['Revenue'].sum().reset_index()
    total_revenue = customer_revenue['Revenue'].sum()

    top_customers = customer_revenue.sort_values(by='Revenue', ascending=False).head(top_n)
    top_customers['Revenue_pct'] = top_customers['Revenue'] / total_revenue * 100

    fig, ax = plt.subplots(figsize=(10,6))
    bars = ax.barh(
        top_customers['Customer ID'].astype(str),
        top_customers['Revenue_pct'],
        color='steelblue'
    )

    ax.invert_yaxis()
    ax.set_xlim(0, max(top_customers['Revenue_pct']) * 1.2)

    for bar, pct in zip(bars, top_customers['Revenue_pct']):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, f'{pct:.2f}%', va='center')

    ax.set_xlabel("Revenue Contribution (%)")
    ax.set_ylabel("Customer ID")
    ax.set_title("Top 10 Customers by Revenue Contribution")
    plt.tight_layout()

    st.pyplot(fig)

def plot_daily_revenue(df):
    daily_revenue = (
        df.groupby("InvoiceDate")["Revenue"]
        .sum()
        .reset_index()
        .sort_values("InvoiceDate")
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=daily_revenue["InvoiceDate"],
            y=daily_revenue["Revenue"],
            mode="lines+markers",
            line=dict(width=3, color="#1f77b4"),
            marker=dict(size=6),
            name="Daily Revenue"
        )
    )
    fig.update_layout(
        title="Daily Revenue",
        xaxis_title="Tanggal",
        yaxis_title="Revenue",
        height=450,
        margin=dict(l=60, r=40, t=60, b=50),
        plot_bgcolor="white",
        hovermode="x unified"
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.1)"
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.1)"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_weekly_revenue(df):
    weekly_revenue = (
        df.groupby('InvoiceDate')['Revenue']
        .sum()
        .resample('W')
        .sum()
        .reset_index()
        .sort_values('InvoiceDate')
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=weekly_revenue['InvoiceDate'],
            y=weekly_revenue['Revenue'],
            mode='lines+markers',
            line=dict(width=3, color='#1f77b4'),
            marker=dict(size=8),
            name='Weekly Revenue'
        )
    )
    fig.update_layout(
        title='Weekly Revenue',
        xaxis_title='Week',
        yaxis_title='Revenue',
        height=450,
        margin=dict(l=60, r=40, t=60, b=50),
        plot_bgcolor='white',
        hovermode='x unified'
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor='rgba(0,0,0,0.1)'
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor='rgba(0,0,0,0.1)',
        tickformat=','
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_monthly_revenue(df):
    monthly_revenue = (
        df.groupby('InvoiceDate')['Revenue']
        .sum()
        .resample('ME')
        .sum()
        .reset_index()
        .sort_values('InvoiceDate')
    )

    # Buat figure Plotly
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=monthly_revenue['InvoiceDate'],
            y=monthly_revenue['Revenue'],
            mode='lines+markers',
            line=dict(width=3, color='#1f77b4'),
            marker=dict(size=8),
            name='Monthly Revenue'
        )
    )

    fig.update_layout(
        title='Monthly Revenue',
        xaxis_title='Month',
        yaxis_title='Revenue',
        height=450,
        margin=dict(l=60, r=40, t=60, b=50),
        plot_bgcolor='white',
        hovermode='x unified'
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor='rgba(0,0,0,0.1)'
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor='rgba(0,0,0,0.1)',
        tickformat=','
    )
    st.plotly_chart(fig, use_container_width=True)

def get_rfm_raw(df, reference_date):
    rfm = (
        df.groupby('Customer ID')
          .agg({
              'InvoiceDate': lambda x: (reference_date - x.max()).days,
              'Invoice': 'nunique',
              'Revenue': 'sum'
          })
          .rename(columns={
              'InvoiceDate': 'Recency',
              'Invoice': 'Frequency',
              'Revenue': 'Monetary'
          })
          .reset_index()
    )
    return rfm

def get_rfm_score(rfm_raw):
    R_Score = pd.qcut(rfm_raw['Recency'], 5, labels=[5,4,3,2,1])
    F_Score = pd.qcut(rfm_raw['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5])
    M_Score = pd.qcut(rfm_raw['Monetary'], 5, labels=[1,2,3,4,5])

    rfm_score = pd.DataFrame({
        'Customer ID': rfm_raw['Customer ID'],
        'R_Score': R_Score,
        'F_Score': F_Score,
        'M_Score': M_Score
    })

    rfm_score['RFM_Score'] = (
        rfm_score['R_Score'].astype(str) +
        rfm_score['F_Score'].astype(str) +
        rfm_score['M_Score'].astype(str)
    )

    return rfm_score


def calculate_rfm(df):
    reference_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'Invoice': 'nunique',
        'Revenue': 'sum'
    }).reset_index()

    rfm.columns = ['Customer ID', 'Recency', 'Frequency', 'Monetary']
    return rfm

def clustering_rfm(rfm_raw, k=4):
    rfm = rfm_raw.copy()
    
    scaler = MinMaxScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    return rfm

def clustering_rfm_custom(rfm, scaler='minmax', algorithm='kmeans', k=4, eps=0.5, min_samples=5, return_scaled=False):
    if scaler == 'minmax':
        scaler_model = MinMaxScaler()
    elif scaler == 'robust':
        scaler_model = RobustScaler()
    else:
        raise ValueError("Scaler must be 'minmax' or 'robust'")

    rfm_scaled = scaler_model.fit_transform(rfm[['Recency','Frequency','Monetary']])

    if algorithm == 'kmeans':
        model = KMeans(n_clusters=k, random_state=42)
        rfm['Cluster'] = model.fit_predict(rfm_scaled)
    elif algorithm == 'dbscan':
        model = DBSCAN(eps=eps, min_samples=min_samples)
        rfm['Cluster'] = model.fit_predict(rfm_scaled)
    else:
        raise ValueError("Algorithm must be 'kmeans' or 'dbscan'")

    if return_scaled:
        return rfm, rfm_scaled
    return rfm

def get_cluster_profile(rfm_clustered):
    cluster_profile = rfm_clustered.groupby('Cluster')[['Recency','Frequency','Monetary']].mean().round(2)
    
    rfm_profile = cluster_profile.rename(columns={
        'Recency': 'Recency (days)',
        'Frequency': 'Frequency (transactions)',
        'Monetary': 'Monetary (£)'
    })
    
    return rfm_profile

def lighten_color(color, amount=0.5):
    import colorsys
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = mcolors.to_rgb(c)
    c = colorsys.rgb_to_hls(*c)
    lightened = colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
    return lightened

def autopct_format(pct, all_vals):
    absolute = int(round(pct/100.*sum(all_vals)))
    return f"{pct:.1f}%\n({absolute})"

def plot_cluster_distribution(rfm_clustered):
    cluster_size = rfm_clustered['Cluster'].value_counts().sort_index()
    clusters = cluster_size.index
    counts = cluster_size.values

    base_colors = plt.cm.tab10.colors[:len(clusters)]
    colors = [lighten_color(c, 0.5) for c in base_colors]

    fig, ax = plt.subplots(figsize=(6,6))
    pie_result = ax.pie(
        counts,
        labels=None,
        autopct=lambda pct: autopct_format(pct, counts),
        startangle=90,
        colors=colors,
        textprops={'fontsize': 12, 'color': 'black', 'weight': 'bold'}
    )

    wedges = pie_result[0]
    ax.legend(wedges, clusters, title="Clusters", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    ax.set_title(f"Customer Distribution per Cluster\nTotal Customers: {counts.sum()}", fontsize=14)
    plt.tight_layout()

    st.pyplot(fig)

def plot_rfm_distribution(rfm_clustered, cluster_profile_mean, cluster_profile_median, metric='Recency'):
    fig, ax = plt.subplots(figsize=(10,5))

    for cluster in cluster_profile_mean.index:
        cluster_data = rfm_clustered[rfm_clustered['Cluster'] == cluster]

        if cluster_data.empty:
            continue

        sns.kdeplot(
            cluster_data[metric],
            label=f'Cluster {cluster}',
            fill=True,
            alpha=0.4,
            ax=ax
        )

        ax.axvline(cluster_profile_mean.loc[cluster, metric], color='b', linestyle='--', alpha=0.7)
        ax.axvline(cluster_profile_median.loc[cluster, metric], color='r', linestyle=':', alpha=0.7)

    ax.set_title(f'Distribution of {metric} per Cluster')
    ax.set_xlabel(metric)
    ax.set_ylabel('Density')
    ax.legend(title='Cluster')
    plt.tight_layout()

    st.pyplot(fig)

def plot_rfm_radar(cluster_profile):
    rfm_columns = ['Recency', 'Frequency', 'Monetary']
    num_metrics = len(rfm_columns)

    cluster_normalized = (cluster_profile - cluster_profile.min()) / (cluster_profile.max() - cluster_profile.min())

    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]

    colors = plt.cm.tab10.colors

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    for idx, row in cluster_normalized.iterrows():
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, marker='o', linewidth=2, label=f'Cluster {idx}', color=colors[idx % len(colors)])
        ax.fill(angles, values, color=colors[idx % len(colors)], alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(rfm_columns, fontsize=12)
    ax.set_yticks(np.linspace(0, 1, 5))
    ax.set_yticklabels([f"{y:.1f}" for y in np.linspace(0, 1, 5)], fontsize=10)
    ax.set_title("RFM Cluster Profile Radar Chart", fontsize=16, fontweight='bold')

    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    st.pyplot(fig)

def plot_rfm_3d(rfm_clustered):
    num_clusters = rfm_clustered['Cluster'].nunique()
    colors = sns.color_palette("tab10", n_colors=num_clusters)

    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(111, projection='3d')

    for cluster in range(num_clusters):
        cluster_data = rfm_clustered[rfm_clustered['Cluster'] == cluster]
        ax.scatter(
            cluster_data['Recency'],
            cluster_data['Frequency'],
            cluster_data['Monetary'],
            s=70,
            color=colors[cluster],
            label=f'Cluster {cluster}'
        )

    ax.set_xlabel('Recency (days since last purchase)', fontsize=12, labelpad=12)
    ax.set_ylabel('Frequency (transactions)', fontsize=12, labelpad=12)
    ax.set_zlabel('Monetary (£)', fontsize=12, labelpad=12)
    ax.set_title('3D Scatter Plot of RFM Clusters', fontsize=14, pad=20)

    z_max = rfm_clustered['Monetary'].max()
    z_ticks = np.linspace(0, z_max, 6)
    ax.set_zticks(z_ticks)
    ax.set_zlim(0, z_max*1.05)
    ax.set_zticklabels([f'{int(tick/1000)}k' for tick in z_ticks])

    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='z', labelsize=10)

    ax.legend(title='Cluster', title_fontsize=12, fontsize=11, loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.subplots_adjust(right=0.8)

    st.pyplot(fig)

def plot_rfm_bars_side_by_side(rfm_profile):
    col1, col2, col3 = st.columns(3)

    with col1:
        fig, ax = plt.subplots(figsize=(4,3))
        rfm_profile[['Recency (days)']].plot(kind='bar', ax=ax, color='skyblue', legend=False, rot=0)
        ax.set_title('Average Recency per Cluster')
        ax.set_ylabel('Days since Last Purchase')
        ax.set_xlabel('Cluster')
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(4,3))
        rfm_profile[['Frequency (transactions)']].plot(kind='bar', ax=ax, color='salmon', legend=False, rot=0)
        ax.set_title('Average Frequency per Cluster')
        ax.set_ylabel('Transactions')
        ax.set_xlabel('Cluster')
        plt.tight_layout()
        st.pyplot(fig)

    with col3:
        fig, ax = plt.subplots(figsize=(4,3))
        rfm_profile[['Monetary (£)']].plot(kind='bar', ax=ax, color='lightgreen', legend=False, rot=0)
        ax.set_title('Average Monetary per Cluster')
        ax.set_ylabel('£')
        ax.set_xlabel('Cluster')
        plt.tight_layout()
        st.pyplot(fig)

def segment_clusters(rfm_profile):
    cluster_profile_unit = rfm_profile.copy()

    rec_q1, rec_q2 = cluster_profile_unit['Recency (days)'].quantile([0.33, 0.66])
    freq_q1, freq_q2 = cluster_profile_unit['Frequency (transactions)'].quantile([0.33, 0.66])
    mon_q1, mon_q2 = cluster_profile_unit['Monetary (£)'].quantile([0.33, 0.66])

    def get_level(value, q1, q2):
        if value <= q1:
            return "Low"
        elif value <= q2:
            return "Medium"
        else:
            return "High"

    def rec_desc(level):
        return {
            "Low": "Recent / Frequent customer",
            "Medium": "Moderate recency",
            "High": "Long time since last purchase / Inactive"
        }[level]

    def freq_desc(level):
        return {
            "Low": "Purchases rarely",
            "Medium": "Moderate purchase frequency",
            "High": "Purchases frequently"
        }[level]

    def mon_desc(level):
        return {
            "Low": "Low transaction value",
            "Medium": "Moderate transaction value",
            "High": "High transaction value"
        }[level]

    cluster_profile_unit['Recency_level'] = cluster_profile_unit['Recency (days)'].apply(lambda x: get_level(x, rec_q1, rec_q2))
    cluster_profile_unit['Frequency_level'] = cluster_profile_unit['Frequency (transactions)'].apply(lambda x: get_level(x, freq_q1, freq_q2))
    cluster_profile_unit['Monetary_level'] = cluster_profile_unit['Monetary (£)'].apply(lambda x: get_level(x, mon_q1, mon_q2))

    cluster_profile_unit['Recency_desc'] = cluster_profile_unit['Recency_level'].apply(rec_desc)
    cluster_profile_unit['Frequency_desc'] = cluster_profile_unit['Frequency_level'].apply(freq_desc)
    cluster_profile_unit['Monetary_desc'] = cluster_profile_unit['Monetary_level'].apply(mon_desc)

    def assign_segment(row):
        R, F, M = row['Recency_level'], row['Frequency_level'], row['Monetary_level']

        if R=="Low" and F=="High" and M=="High":
            return "Champion / Loyal Customers"
        elif R=="High" and F=="Low" and M=="Low":
            return "Lost / At Risk"
        elif R in ["Medium","High"] and F in ["Low","Medium"]:
            return "Hibernating / Needs Attention"
        else:
            return "Potential / Promising"

    cluster_profile_unit['Segment'] = cluster_profile_unit.apply(assign_segment, axis=1)

    cols_order = [
        'Segment', 'Recency (days)','Recency_level','Recency_desc',
        'Frequency (transactions)','Frequency_level','Frequency_desc',
        'Monetary (£)','Monetary_level','Monetary_desc'
    ]
    
    return cluster_profile_unit[cols_order]

segment_order = [
    'Champion / Loyal Customers',
    'Hibernating / Needs Attention',
    'Potential / Promising',
    'Lost / At Risk'
]

segment_colors = {
    'Champion / Loyal Customers': '#FFA500',
    'Hibernating / Needs Attention': '#2ca02c',
    'Potential / Promising': '#1f77b4',
    'Lost / At Risk': '#d62728'
}

def plot_monthly_segment_trend(monthly_customers):
    if monthly_customers is None or monthly_customers.empty:
        st.warning("⚠️ No monthly customer data available.")
        return

    monthly_pivot = monthly_customers.pivot(
        index='YearMonth',
        columns='Segment',
        values='TotalCustomers'
    ).fillna(0)

    valid_segments = [s for s in segment_order if s in monthly_pivot.columns]
    if len(valid_segments) == 0:
        st.warning("⚠️ No valid segments available for plotting.")
        return

    monthly_pivot = monthly_pivot[valid_segments[::-1]]
    colors = [segment_colors[s] for s in monthly_pivot.columns]

    fig, ax = plt.subplots(figsize=(12,6))

    ax.stackplot(
        monthly_pivot.index.to_timestamp(),
        *[monthly_pivot[col].values for col in monthly_pivot.columns],
        labels=monthly_pivot.columns,
        colors=colors,
        alpha=0.75
    )

    ax.set_title("Monthly Trend of Customers per Segment", fontsize=14)
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Customers")

    ax.legend(
        title="Customer Segment",
        loc="upper left",
        frameon=True
    )

    plt.tight_layout()
    st.pyplot(fig)

def plot_customer_pyramid(df_segment: pd.DataFrame, segment_col: str = 'Segment'):
    customer_count = df_segment[segment_col].value_counts().reindex(segment_order[::-1], fill_value=0)
    
    percent = customer_count.values / customer_count.sum()
    segments = segment_order[::-1]
    
    fig, ax = plt.subplots(figsize=(6,8))

    bottom = 0
    for s, p in zip(segments, percent):
        left = 0.5 - (1 - bottom) / 2
        right = 0.5 + (1 - bottom) / 2
        xs = [left, right, right, left]
        ys = [bottom, bottom, bottom + p, bottom + p]
        ax.fill(xs, ys, color=segment_colors.get(s, '#888888'))
        ax.text(
            0.5, bottom + p/2,
            f"{s}\n{p*100:.1f}%",
            ha='center', va='center', color='white', fontsize=10
        )
        bottom += p

    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.axis('off')
    ax.set_title('Customer Pyramid (% of Users by Segment)')
    
    st.pyplot(fig)

menu = st.sidebar.selectbox(
    "Menu",
    ["Data", "RFM & Clustering", "Segmentation & Insight", "Recommendation per Customer"]
)

@st.cache_data
def load_all_data(url):
    df = preprocess_data(url)
    reference_date = df['InvoiceDate'].max() + dt.timedelta(days=1)
    
    rfm_raw = get_rfm_raw(df, reference_date)
    rfm_score = get_rfm_score(rfm_raw)
    
    return df, rfm_raw, rfm_score, reference_date

file_id = "1tw0gEG9wy5QnIYhz0-1_v75Fh7YbDrBN"
url = f"https://drive.google.com/uc?id={file_id}"

df, rfm_raw, rfm_score, reference_date = load_all_data(url)

@st.cache_data
def get_rfm_profile(rfm_clustered):
    rfm_profile = rfm_clustered.groupby('Cluster')[['Recency','Frequency','Monetary']].mean().round(2)
    rfm_profile = rfm_profile.rename(columns={
        'Recency': 'Recency (days)',
        'Frequency': 'Frequency (transactions)',
        'Monetary': 'Monetary (£)'
    })
    return rfm_profile

@st.cache_data
def prepare_monthly_customers(rfm_clustered, df_transactions):
    df = rfm_clustered.copy()

    df = df[(df['Cluster'] != -1) & (df['Segment'].notna())]

    if df.empty:
        return pd.DataFrame()

    df = df.merge(
        df_transactions[['Customer ID', 'InvoiceDate']],
        on='Customer ID',
        how='left'
    )

    df['YearMonth'] = df['InvoiceDate'].dt.to_period('M')

    monthly_customers = (
        df.groupby(['YearMonth', 'Segment'])['Customer ID']
          .nunique()
          .reset_index(name='TotalCustomers')
    )
    return monthly_customers

if menu == "Data":
    st.title("📊 Data Preview")

    st.subheader("Preview Data")
    st.dataframe(df.head())

    st.write("Unique Customers:", df['Customer ID'].nunique())
    st.write("Total Transactions:", df.shape[0])

    st.subheader("Data Statistics")
    st.write(df.describe())

    st.subheader("📈 Data Visualization")

    plot_option = st.selectbox(
        "Choose Visualization",
        (
            "Choose Basic Report",
            "Top 10 Customer by Revenue",
            "Daily Revenue",
            "Weekly Revenue",
            "Monthly Revenue"
        )
    )

    if plot_option == "Top 10 Customer by Revenue":
        plot_top_customer(df)
    elif plot_option == "Daily Revenue":
        plot_daily_revenue(df)
    elif plot_option == "Weekly Revenue":
        plot_weekly_revenue(df)
    elif plot_option == "Monthly Revenue":
        plot_monthly_revenue(df)

elif menu == "RFM & Clustering":
    st.subheader("Raw RFM Values")
    st.dataframe(rfm_raw.head())

    st.subheader("RFM Scores")
    st.dataframe(rfm_score.head())

    rfm_plot = rfm_raw.join(rfm_score.set_index('Customer ID'), on='Customer ID')
    heatmap_data = rfm_plot.groupby(['R_Score','F_Score'], observed=True)['Monetary'].mean().unstack()

    st.subheader("RFM Heatmap (Recency vs Frequency, Monetary as color)")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax)
    ax.set_title("RFM Heatmap")
    ax.set_xlabel("Frequency Score")
    ax.set_ylabel("Recency Score")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    st.pyplot(fig)

    if 'clustering_algo' not in st.session_state:
        st.session_state['clustering_algo'] = 'KMeans'

    algorithm = st.selectbox(
        "Clustering Algorithm", 
        ['KMeans', 'DBSCAN'], 
        index=['KMeans', 'DBSCAN'].index(st.session_state['clustering_algo'])
    )

    st.session_state['clustering_algo'] = algorithm

    if algorithm == 'KMeans':
        scaler = st.selectbox("Scaler", ['MinMaxScaler', 'RobustScaler'])
    elif algorithm == 'DBSCAN':
        scaler = st.selectbox("Scaler", ['RobustScaler'])

    if algorithm == 'KMeans':
        k = st.slider("Number of Clusters (KMeans)", 2, 10, 4)
        rfm_clustered, X_scaled = clustering_rfm_custom(
            rfm_raw,
            scaler=scaler.lower().replace('scaler',''),
            algorithm='kmeans',
            k=k,
            return_scaled=True
        )
    elif algorithm == 'DBSCAN':
        eps = st.number_input("DBSCAN eps", 0.1, 5.0, 0.5)
        min_samples = st.number_input("DBSCAN min_samples", 1, 20, 5)
        rfm_clustered, X_scaled = clustering_rfm_custom(
            rfm_raw,
            scaler=scaler.lower().replace('scaler',''),
            algorithm='dbscan',
            eps=eps,
            min_samples=min_samples,
            return_scaled=True
        )
    
    if algorithm == 'KMeans':
        sil_score = silhouette_score(X_scaled, rfm_clustered['Cluster'])
        st.write(f"**Silhouette Score:** {sil_score:.4f}")

        dbi_score = davies_bouldin_score(X_scaled, rfm_clustered['Cluster'])
        st.write(f"**Davies-Bouldin Index (DBI):** {dbi_score:.4f}")

        wcss = []
        for i in range(2, 11):
            km = KMeans(n_clusters=i, random_state=42)
            km.fit(X_scaled)
            wcss.append(km.inertia_)

        fig, ax = plt.subplots()
        ax.plot(range(2, 11), wcss, marker='o')
        ax.set_title("Elbow Method (WCSS vs Number of Clusters)")
        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("WCSS")
        st.pyplot(fig)

    st.session_state['rfm_clustered'] = rfm_clustered
    st.session_state['clustering_algo'] = algorithm
    st.session_state['scaler'] = scaler

    st.subheader("RFM Clustering Result")
    st.dataframe(rfm_clustered.head())

    st.subheader("Cluster Profile")
    cluster_profile = get_cluster_profile(rfm_clustered)
    st.dataframe(cluster_profile)

    st.subheader("Customer Distribution per Cluster")
    plot_cluster_distribution(rfm_clustered)

    cluster_profile_mean = rfm_clustered.groupby('Cluster')[['Recency','Frequency','Monetary']].mean().round(2)
    cluster_profile_median = rfm_clustered.groupby('Cluster')[['Recency','Frequency','Monetary']].median().round(2)

    metric_option = st.selectbox("Choose RFM Metric to Plot", ['Recency', 'Frequency', 'Monetary'])

    cluster_profile_mean = rfm_clustered.groupby('Cluster')[['Recency','Frequency','Monetary']].mean().round(2)
    cluster_profile_median = rfm_clustered.groupby('Cluster')[['Recency','Frequency','Monetary']].median().round(2)

    st.subheader(f"{metric_option} Distribution per Cluster")
    plot_rfm_distribution(rfm_clustered, cluster_profile_mean, cluster_profile_median, metric=metric_option)

    cluster_profile = get_cluster_profile(rfm_clustered)

    st.subheader("RFM Cluster Radar Chart")
    plot_rfm_radar(cluster_profile)

    st.subheader("3D Scatter Plot of RFM Clusters")
    plot_rfm_3d(rfm_clustered)

    rfm_profile = get_rfm_profile(rfm_clustered)

    st.session_state['rfm_profile'] = rfm_profile

    st.subheader("RFM Cluster Bar Charts")
    plot_rfm_bars_side_by_side(rfm_profile)

elif menu == "Segmentation & Insight":
    st.title("🔍 Customer Segmentation Insight")

    if 'rfm_clustered' not in st.session_state:
        k = 4
        rfm_clustered = clustering_rfm(rfm_raw, k)
        st.session_state['rfm_clustered'] = rfm_clustered
    else:
        rfm_clustered = st.session_state['rfm_clustered']

    rfm_profile = get_rfm_profile(rfm_clustered)

    if 'cluster_profile_unit' not in st.session_state:
        cluster_profile_unit = segment_clusters(rfm_profile).reset_index()
        st.session_state['cluster_profile_unit'] = cluster_profile_unit
    else:
        cluster_profile_unit = st.session_state['cluster_profile_unit']

    if 'Segment' not in rfm_clustered.columns:
        rfm_clustered = rfm_clustered.merge(
            cluster_profile_unit[['Cluster','Segment']],
            on='Cluster',
            how='left'
        )
        st.session_state['rfm_clustered'] = rfm_clustered

    st.subheader("Cluster Insight Table")
    cluster_profile_unit.index = range(1, len(cluster_profile_unit) + 1)

    st.dataframe(cluster_profile_unit)

    st.subheader("📈 Monthly Trend of Customers per Segment")
    monthly_customers = prepare_monthly_customers(
        rfm_clustered,
        df_transactions=df
    )
    plot_monthly_segment_trend(monthly_customers)

    st.subheader("👥 Customer Pyramid")
    plot_customer_pyramid(rfm_clustered, segment_col='Segment')

    df_segment_customers = rfm_clustered[['Customer ID', 'Segment']].copy()
    df_segment_customers = df_segment_customers.sort_values(['Segment', 'Customer ID']).reset_index(drop=True)

    df_segment_counts = rfm_clustered.groupby('Segment')['Customer ID'].nunique().reset_index()
    df_segment_counts = df_segment_counts.rename(columns={'Customer ID': 'Customer Count'}).sort_values('Segment').reset_index(drop=True)

    st.subheader("Customer ID per Segment")
    st.dataframe(df_segment_customers)

    st.subheader("Customer Count per Segment")
    st.dataframe(df_segment_counts)

elif menu == "Recommendation per Customer":
    st.title("🛍️ Customer Recommendation")

    if 'rfm_clustered' not in st.session_state:
        st.error("RFM segmentation not computed. Please run the Insight menu first.")
    else:
        rfm_clustered = st.session_state['rfm_clustered']

        if 'Segment' not in rfm_clustered.columns:
            if 'cluster_profile_unit' in st.session_state:
                rfm_clustered = rfm_clustered.merge(
                    st.session_state['cluster_profile_unit'][['Cluster', 'Segment']],
                    on='Cluster',
                    how='left'
                )
                st.session_state['rfm_clustered'] = rfm_clustered
            else:
                st.error("Segment info missing. Please run the Insight menu first.")
                st.stop()

        customer_ids = rfm_clustered['Customer ID'].unique()
        customer_id = st.selectbox("Select Customer ID:", options=customer_ids)

        if customer_id:
            customer_row = rfm_clustered[rfm_clustered['Customer ID'] == customer_id]
            segment = customer_row['Segment'].values[0]

            recommendation_table = {
                "Champion / Loyal Customers": {
                    "Persona": "Frequent, high-spending, recent purchasers. Core revenue contributors.",
                    "Recommendation": "VIP programs, loyalty rewards, early access to products. Focus on retention."
                },
                "Potential / Promising": {
                    "Persona": "Recent or moderately frequent buyers, mid-value transactions. Growth opportunity.",
                    "Recommendation": "Upsell/cross-sell campaigns, personalized recommendations, incentives to move to Champions."
                },
                "Hibernating / Needs Attention": {
                    "Persona": "Infrequent buyers, mid-low transaction value, inactive for a while.",
                    "Recommendation": "Re-engagement campaigns: email reminders, discounts, special offers to prevent churn."
                },
                "Lost / At Risk": {
                    "Persona": "Rare or no recent purchases, high AOV historically.",
                    "Recommendation": "Win-back campaigns: exclusive offers, targeted communication, highlight new products."
                }
            }

            st.subheader(f"Customer {customer_id} Segment: {segment}")
            st.markdown(f"**Persona:** {recommendation_table[segment]['Persona']}")
            st.markdown(f"**Key Recommendation:** {recommendation_table[segment]['Recommendation']}")