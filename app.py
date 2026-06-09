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
    try:
        # Try reading the actual CSV from the repository root
        df = pd.read_csv('recent_sales_09_06.csv')
        df['Sale Application Date'] = pd.to_datetime(df['Sale Application Date'])
        df['YearMonth'] = df['Sale Application Date'].dt.to_period('M').astype(str)
        return df
    except FileNotFoundError:
        # Fallback to dummy data if the CSV is missing
        st.warning("⚠️ 'recent_sales_09_06.csv' not found. Loading sample data instead.")
        np.random.seed(42)
        dates = pd.date_range(start='2019-01-01', end='2026-06-01', freq='D')
        districts = ['Yas Island', 'Al Reem Island', 'Al Rahah', 'Ghayathi', "Al Faqa'"]
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
df = df_raw.copy()

# --- Sidebar Filters ---
st.sidebar.title("🔍 Filters")

# 1. Outlier Removal (1% and 99%)
st.sidebar.subheader("Outlier Treatment")
remove_outliers = st.sidebar.checkbox("Remove 1% & 99% Outliers", value=True)

if remove_outliers:
    # Safely check if columns exist before applying outlier logic
    if 'Rate (AED per SQM)' in df.columns and 'Property Sold Area (SQM)' in df.columns:
        rate_1 = df_raw['Rate (AED per SQM)'].quantile(0.01)
        rate_99 = df_raw['Rate (AED per SQM)'].quantile(0.99)
        area_1 = df_raw['Property Sold Area (SQM)'].quantile(0.01)
        area_99 = df_raw['Property Sold Area (SQM)'].quantile(0.99)
        
        df = df[
            (df['Rate (AED per SQM)'] >= rate_1) & (df['Rate (AED per SQM)'] <= rate_99) &
            (df['Property Sold Area (SQM)'] >= area_1) & (df['Property Sold Area (SQM)'] <= area_99)
        ]

# 2. Dynamic Cascading Filters
st.sidebar.subheader("Dynamic Data Filters")

# Automatically grab all categorical/object columns
filter_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
columns_to_exclude = ['Sale Application Date', 'YearMonth']
filter_columns = [col for col in filter_columns if col not in columns_to_exclude]

# Loop through each column to create a cascading filter
for col in filter_columns:
    # Get the unique values based on the CURRENT state of the dynamically filtered 'df'
    available_options = sorted(df[col].dropna().astype(str).unique().tolist())
    
    selected_values = st.sidebar.multiselect(
        f"Filter by {col}", 
        options=available_options, 
        default=[] 
    )
    
    # If the user makes a selection, filter the dataframe immediately for the next loop iteration
    if selected_values:
        df = df[df[col].astype(str).isin(selected_values)]

# --- Main Dashboard Title ---
st.title("🏢 Real Estate Market Analysis")
st.markdown("Comprehensive insights into property transactions, trends, and distributions.")

# Top KPIs (Safeguarded against empty dataframes)
if not df.empty:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{len(df):,}")
    
    if 'Rate (AED per SQM)' in df.columns:
        col2.metric("Median Rate (AED/SQM)", f"{df['Rate (AED per SQM)'].median():,.0f}")
    if 'Property Sold Area (SQM)' in df.columns:
        col3.metric("Median Area (SQM)", f"{df['Property Sold Area (SQM)'].median():,.0f}")
    if 'Property Sale Price (AED)' in df.columns:
        col4.metric("Total Value (AED)", f"{df['Property Sale Price (AED)'].sum() / 1e9:,.2f} B")
else:
    st.warning("⚠️ No data matches the current filter criteria. Please adjust your filters.")

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Trends", "📊 Distributions", "🔗 Relationships", "🗺️ District Map", "📋 Data & Summary"
])

if not df.empty:
    # ==========================================
    # TAB 1: TRENDS
    # ==========================================
    with tab1:
        st.subheader("Time Series Trends: Median Rate & Transaction Volume")
        
        # We use the categorical columns we found earlier for the grouping dropdown
        grouping_options = filter_columns if filter_columns else ["Property Type"]
        trend_group_col = st.selectbox("Group Trends By:", grouping_options, index=0)
        
        if 'Rate (AED per SQM)' in df.columns:
            trend_df = df.groupby(['YearMonth', trend_group_col]).agg(
                Median_Rate=('Rate (AED per SQM)', 'median'),
                Tx_Count=('Rate (AED per SQM)', 'count')
            ).reset_index()
            
            fig_trends = make_subplots(specs=[[{"secondary_y": True}]])
            
            for category in trend_df[trend_group_col].unique():
                cat_df = trend_df[trend_df[trend_group_col] == category]
                # Bars for volume
                fig_trends.add_trace(
                    go.Bar(x=cat_df['YearMonth'], y=cat_df['Tx_Count'], name=f"Count: {category}", opacity=0.6),
                    secondary_y=False,
                )
                # Lines for rates
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
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        if numeric_cols:
            dist_col, color_col = st.columns(2)
            with dist_col:
                target_col = st.selectbox("Select Metric to Visualize:", numeric_cols)
            with color_col:
                color_options = ["None"] + filter_columns
                dist_color = st.selectbox("Split Distribution By:", color_options)
            
            color_arg = None if dist_color == "None" else dist_color
            
            fig_dist = px.histogram(
                df, 
                x=target_col, 
                color=color_arg, 
                marginal="box", 
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
        if numeric_cols and filter_columns:
            r_col1, r_col2, r_col3, r_col4 = st.columns(4)
            with r_col1:
                x_axis = st.selectbox("X-Axis", numeric_cols, index=0)
            with r_col2:
                # Try to set a different default for Y if possible
                default_y_idx = 1 if len(numeric_cols) > 1 else 0
                y_axis = st.selectbox("Y-Axis", numeric_cols, index=default_y_idx)
            with r_col3:
                color_dim = st.selectbox("Color (Categorical)", filter_columns)
            with r_col4:
                size_dim = st.selectbox("Bubble Size (Numeric)", ["None"] + numeric_cols)

            chart_type = st.radio("Chart Type", ["Scatter Plot", "Density Heat Map"], horizontal=True)
            size_arg = None if size_dim == "None" else size_dim

            if chart_type == "Scatter Plot":
                fig_rel = px.scatter(
                    df, x=x_axis, y=y_axis, color=color_dim, size=size_arg,
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
        
        if 'District' in df.columns and 'Rate (AED per SQM)' in df.columns:
            # Mock Coordinates Mapping for Abu Dhabi Districts
            coord_mapping = {
                'Yas Island': {'lat': 24.496, 'lon': 54.603},
                'Al Reem Island': {'lat': 24.494, 'lon': 54.406},
                'Al Rahah': {'lat': 24.437, 'lon': 54.582},
                'Ghayathi': {'lat': 23.844, 'lon': 52.810},
                "Al Faqa'": {'lat': 24.733, 'lon': 55.616}
            }
            
            map_df = df.groupby('District').agg(
                Median_Rate=('Rate (AED per SQM)', 'median'),
                Transactions=('Rate (AED per SQM)', 'count')
            ).reset_index()
            
            # Map coordinates safely - defaults to Abu Dhabi center if not found
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
            st.info("💡 Note: This visualization aggregates by District using predefined coordinate mappings. If you have exact latitude/longitude columns in your raw dataset in the future, you can map exact project locations.")
        else:
            st.warning("Map requires 'District' and 'Rate (AED per SQM)' columns to render.")

    # ==========================================
    # TAB 5: DATA & SUMMARY
    # ==========================================
    with tab5:
        st.subheader("Filtered Data Summary")
        
        st.markdown("**Descriptive Statistics**")
        st.dataframe(df.describe().T.style.format("{:,.2f}"))
        
        st.markdown("**Raw Data** (Showing up to 1000 rows)")
        st.dataframe(df.head(1000))
        
        # Allow CSV Download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name='filtered_real_estate_data.csv',
            mime='text/csv',
        )
