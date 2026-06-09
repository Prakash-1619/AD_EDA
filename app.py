import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Page Config ---
st.set_page_config(page_title="Real Estate Market Dashboard", page_icon="🏢", layout="wide")

# --- Data Loading & Caching ---
@st.cache_data
def load_data():
    # Replace with your actual data path or dynamic upload
    # df = pd.read_csv('/content/drive/MyDrive/TruEstates.ai/ABU_DHABI/Transactions/recent_sales_09_06.csv')
    df = pd.read_csv('recent_sales_09_06.csv')
    # For demonstration, creating a dummy dataframe mirroring your schema
    np.random.seed(42)
    dates = pd.date_range(start='2019-01-01', end='2026-06-01', freq='D')
    districts = ['Yas Island', 'Al Reem Island', 'Al Rahah', 'Ghayathi', 'Al Faqa\'']
    prop_types = ['apartment', 'villa', 'townhouse', 'plot']
    
    df = pd.DataFrame({
        'Sale Application Date': np.random.choice(dates, 10000),
        'Property Type': np.random.choice(prop_types, 10000),
        'District': np.random.choice(districts, 10000),
        'Sale Sequence': np.random.choice(['primary', 'secondary'], 10000),
        'Property Sold Area (SQM)': np.random.exponential(scale=100, size=10000) + 20,
        'Property Sale Price (AED)': np.random.exponential(scale=1500000, size=10000) + 500000,
    })
    df['Rate (AED per SQM)'] = df['Property Sale Price (AED)'] / df['Property Sold Area (SQM)']
    df['Sale Application Date'] = pd.to_datetime(df['Sale Application Date'])
    df['YearMonth'] = df['Sale Application Date'].dt.to_period('M').astype(str)
    return df

df_raw = load_data()

# --- Sidebar Filters ---
st.sidebar.title("🔍 Filters")

# 1. Outlier Removal (1% and 99%)
st.sidebar.subheader("Outlier Treatment")
remove_outliers = st.sidebar.checkbox("Remove 1% & 99% Outliers", value=True)

if remove_outliers:
    rate_1 = df_raw['Rate (AED per SQM)'].quantile(0.01)
    rate_99 = df_raw['Rate (AED per SQM)'].quantile(0.99)
    area_1 = df_raw['Property Sold Area (SQM)'].quantile(0.01)
    area_99 = df_raw['Property Sold Area (SQM)'].quantile(0.99)
    
    df = df_raw[
        (df_raw['Rate (AED per SQM)'] >= rate_1) & (df_raw['Rate (AED per SQM)'] <= rate_99) &
        (df_raw['Property Sold Area (SQM)'] >= area_1) & (df_raw['Property Sold Area (SQM)'] <= area_99)
    ]
else:
    df = df_raw.copy()

# 2. Dynamic Data Filters
st.sidebar.subheader("Data Filters")
selected_districts = st.sidebar.multiselect("Select Districts", options=df['District'].unique(), default=df['District'].unique()[:3])
selected_types = st.sidebar.multiselect("Property Types", options=df['Property Type'].unique(), default=df['Property Type'].unique())
selected_seq = st.sidebar.multiselect("Sale Sequence", options=df['Sale Sequence'].unique(), default=df['Sale Sequence'].unique())

# Apply filters
if selected_districts:
    df = df[df['District'].isin(selected_districts)]
if selected_types:
    df = df[df['Property Type'].isin(selected_types)]
if selected_seq:
    df = df[df['Sale Sequence'].isin(selected_seq)]

# --- Main Dashboard Title ---
st.title("🏢 Real Estate Market Analysis")
st.markdown("Comprehensive insights into property transactions, trends, and distributions.")

# Top KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{len(df):,}")
col2.metric("Median Rate (AED/SQM)", f"{df['Rate (AED per SQM)'].median():,.0f}")
col3.metric("Median Area (SQM)", f"{df['Property Sold Area (SQM)'].median():,.0f}")
col4.metric("Total Value (AED)", f"{df['Property Sale Price (AED)'].sum() / 1e9:,.2f} B")

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Trends", "📊 Distributions", "🔗 Relationships", "🗺️ District Map", "📋 Data & Summary"
])

# ==========================================
# TAB 1: TRENDS
# ==========================================
with tab1:
    st.subheader("Time Series Trends: Median Rate & Transaction Volume")
    
    trend_group_col = st.selectbox("Group Trends By:", ["Property Type", "District", "Sale Sequence"], index=0)
    
    # Aggregate data
    trend_df = df.groupby(['YearMonth', trend_group_col]).agg(
        Median_Rate=('Rate (AED per SQM)', 'median'),
        Tx_Count=('Rate (AED per SQM)', 'count')
    ).reset_index()
    
    # Create Subplots (Dual Axis)
    fig_trends = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add Bar for Transaction Counts (Stacked)
    for category in trend_df[trend_group_col].unique():
        cat_df = trend_df[trend_df[trend_group_col] == category]
        fig_trends.add_trace(
            go.Bar(x=cat_df['YearMonth'], y=cat_df['Tx_Count'], name=f"Count: {category}", opacity=0.6),
            secondary_y=False,
        )
        # Add Line for Median Rate
        fig_trends.add_trace(
            go.Scatter(x=cat_df['YearMonth'], y=cat_df['Median_Rate'], name=f"Rate: {category}", mode='lines+markers'),
            secondary_y=True,
        )

    fig_trends.update_layout(
        title_text="Transaction Volume (Bars) vs Median Rate (Lines)",
        barmode='stack',
        height=600,
        hovermode="x unified"
    )
    fig_trends.update_yaxes(title_text="Transaction Count", secondary_y=False)
    fig_trends.update_yaxes(title_text="Median Rate (AED per SQM)", secondary_y=True)
    
    st.plotly_chart(fig_trends, use_container_width=True)

# ==========================================
# TAB 2: DISTRIBUTIONS
# ==========================================
with tab2:
    st.subheader("Data Distributions")
    
    dist_col, color_col = st.columns(2)
    with dist_col:
        target_col = st.selectbox("Select Metric to Visualize:", 
                                  ["Rate (AED per SQM)", "Property Sold Area (SQM)", "Property Sale Price (AED)"])
    with color_col:
        dist_color = st.selectbox("Split Distribution By:", ["None", "Property Type", "District", "Sale Sequence"])
    
    color_arg = None if dist_color == "None" else dist_color
    
    fig_dist = px.histogram(
        df, 
        x=target_col, 
        color=color_arg, 
        marginal="box", # Adds a boxplot on top
        barmode="overlay",
        title=f"Distribution of {target_col}",
        opacity=0.7
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# ==========================================
# TAB 3: RELATIONSHIPS
# ==========================================
with tab3:
    st.subheader("Multivariate Relationships")
    st.markdown("Explore how different variables interact. Choose multiple columns to build a Scatter Plot or Heat Map.")
    
    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    with r_col1:
        x_axis = st.selectbox("X-Axis", ["Property Sold Area (SQM)", "Rate (AED per SQM)", "Property Sale Price (AED)"], index=0)
    with r_col2:
        y_axis = st.selectbox("Y-Axis", ["Rate (AED per SQM)", "Property Sold Area (SQM)", "Property Sale Price (AED)"], index=1)
    with r_col3:
        color_dim = st.selectbox("Color (Categorical)", ["Property Type", "District", "Sale Sequence"])
    with r_col4:
        size_dim = st.selectbox("Bubble Size (Numeric)", ["None", "Property Sold Area (SQM)", "Property Sale Price (AED)"])

    chart_type = st.radio("Chart Type", ["Scatter Plot", "Density Heat Map"], horizontal=True)

    size_arg = None if size_dim == "None" else size_dim

    if chart_type == "Scatter Plot":
        fig_rel = px.scatter(
            df, x=x_axis, y=y_axis, color=color_dim, size=size_arg,
            hover_data=['District', 'Sale Application Date'],
            title=f"Scatter: {y_axis} vs {x_axis}",
            opacity=0.6,
            height=600
        )
    else:
        fig_rel = px.density_heatmap(
            df, x=x_axis, y=y_axis, facet_col=color_dim,
            title=f"Heatmap: {y_axis} vs {x_axis} by {color_dim}",
            height=600
        )
        
    st.plotly_chart(fig_rel, use_container_width=True)

# ==========================================
# TAB 4: MAP VIEW
# ==========================================
with tab4:
    st.subheader("Geographical Distribution")
    
    # Mock Coordinates Mapping for Abu Dhabi Districts (Replace with actual data if available)
    coord_mapping = {
        'Yas Island': {'lat': 24.496, 'lon': 54.603},
        'Al Reem Island': {'lat': 24.494, 'lon': 54.406},
        'Al Rahah': {'lat': 24.437, 'lon': 54.582},
        'Ghayathi': {'lat': 23.844, 'lon': 52.810},
        "Al Faqa'": {'lat': 24.733, 'lon': 55.616}
    }
    
    # Aggregate data for map
    map_df = df.groupby('District').agg(
        Median_Rate=('Rate (AED per SQM)', 'median'),
        Transactions=('Rate (AED per SQM)', 'count')
    ).reset_index()
    
    # Map coordinates safely
    map_df['lat'] = map_df['District'].map(lambda x: coord_mapping.get(x, {'lat': 24.4539})['lat'])
    map_df['lon'] = map_df['District'].map(lambda x: coord_mapping.get(x, {'lon': 54.3773})['lon'])

    fig_map = px.scatter_mapbox(
        map_df, 
        lat="lat", 
        lon="lon", 
        hover_name="District", 
        hover_data={"Median_Rate": ":.0f", "Transactions": True, "lat": False, "lon": False},
        color="Median_Rate",
        size="Transactions",
        color_continuous_scale=px.colors.sequential.Viridis,
        zoom=9, 
        height=600,
        title="Districts Overview (Bubble Size = Transactions, Color = Median Rate)"
    )
    
    fig_map.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_map, use_container_width=True)
    st.info("💡 Note: Ensure your raw dataset contains 'lat' and 'lon' columns for exact project coordinates. This visualization aggregates by District utilizing predefined coordinate mappings.")

# ==========================================
# TAB 5: DATA & SUMMARY
# ==========================================
with tab5:
    st.subheader("Filtered Data Summary")
    
    st.markdown("**Descriptive Statistics**")
    st.dataframe(df.describe().T.style.format("{:,.2f}"))
    
    st.markdown("**Raw Data (First 1000 Rows)**")
    st.dataframe(df.head(1000))
    
    # Allow CSV Download
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name='filtered_real_estate_data.csv',
        mime='text/csv',
    )
