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
        df = pd.read_csv('recent_sales_09_06.csv')
        df['Sale Application Date'] = pd.to_datetime(df['Sale Application Date'])
        df['YearMonth'] = df['Sale Application Date'].dt.to_period('M').astype(str)
        return df
    except FileNotFoundError:
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

# Automatically categorize columns
num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
filter_columns = [col for col in cat_cols if col not in ['Sale Application Date', 'YearMonth']]

# ==========================================
# SIDEBAR: DYNAMIC OUTLIERS
# ==========================================
st.sidebar.title("🛠️ Settings & Filters")
st.sidebar.subheader("1. Outlier Configuration")

# Choose when to apply outliers
outlier_scope = st.sidebar.radio(
    "Apply Outliers To:", 
    ["Overall Data (Before Filters)", "Filtered Data (After Filters)"],
    help="Overall applies rules globally. Filtered applies rules only to the data left after your category selections."
)

outlier_bounds = {}
st.sidebar.markdown("**Numerical Percentiles**")
for col in num_cols:
    with st.sidebar.expander(f"⚙️ {col} Limits"):
        if st.checkbox(f"Filter {col}", value=False, key=f"check_{col}"):
            lower, upper = st.slider(f"Percentile Range", 0.0, 100.0, (1.0, 99.0), 0.5, key=f"slider_{col}")
            outlier_bounds[col] = (lower / 100.0, upper / 100.0)

st.sidebar.markdown("**Categorical Thresholds**")
min_cat_freq = st.sidebar.number_input(
    "Minimum Category Frequency", 
    min_value=1, value=1, 
    help="Drops categories with fewer than this many transactions."
)

# Helper function to apply outlier logic
def apply_outliers(data):
    d = data.copy()
    # 1. Apply numerical bounds
    for col, (low, high) in outlier_bounds.items():
        l_val = d[col].quantile(low)
        h_val = d[col].quantile(high)
        d = d[(d[col] >= l_val) & (d[col] <= h_val)]
    # 2. Apply categorical thresholds
    if min_cat_freq > 1:
        for col in filter_columns:
            counts = d[col].value_counts()
            valid_cats = counts[counts >= min_cat_freq].index
            d = d[d[col].isin(valid_cats)]
    return d

# Apply BEFORE filters if selected
if outlier_scope == "Overall Data (Before Filters)":
    df = apply_outliers(df)

# ==========================================
# SIDEBAR: DYNAMIC CASCADING FILTERS
# ==========================================
st.sidebar.subheader("2. Dynamic Filters")
for col in filter_columns:
    available_options = sorted(df[col].dropna().astype(str).unique().tolist())
    selected_values = st.sidebar.multiselect(f"Filter by {col}", options=available_options, default=[])
    if selected_values:
        df = df[df[col].astype(str).isin(selected_values)]

# Apply AFTER filters if selected
if outlier_scope == "Filtered Data (After Filters)":
    df = apply_outliers(df)


# ==========================================
# MAIN DASHBOARD
# ==========================================
st.title("🏢 Real Estate Market Analysis")
st.markdown("Comprehensive insights into property transactions, trends, and distributions.")

# Top KPIs
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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Trends", "📊 Distributions", "🔗 Relationships", "🗺️ District Map", "🧩 Correlations", "📋 Data & Summary"
])

if not df.empty:
    # ==========================================
    # TAB 1: TRENDS
    # ==========================================
    with tab1:
        st.subheader("Time Series Trends: Median Rate & Transaction Volume")
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
                fig_trends.add_trace(
                    go.Bar(x=cat_df['YearMonth'], y=cat_df['Tx_Count'], name=f"Count: {category}", opacity=0.6),
                    secondary_y=False,
                )
                fig_trends.add_trace(
                    go.Scatter(x=cat_df['YearMonth'], y=cat_df['Median_Rate'], name=f"Rate: {category}", mode='lines+markers'),
                    secondary_y=True,
                )

            fig_trends.update_layout(title_text="Transaction Volume (Bars) vs Median Rate (Lines)", barmode='stack', height=600, hovermode="x unified")
            fig_trends.update_yaxes(title_text="Transaction Count", secondary_y=False)
            fig_trends.update_yaxes(title_text="Median Rate (AED per SQM)", secondary_y=True)
            st.plotly_chart(fig_trends, use_container_width=True)

    # ==========================================
    # TAB 2: DISTRIBUTIONS (Enhanced)
    # ==========================================
    with tab2:
        st.subheader("Data Distributions")
        dist_type = st.radio("Choose Distribution Type:", ["Categorical (Categories vs Values)", "Numerical (Histograms)"], horizontal=True)
        
        if dist_type == "Numerical (Histograms)":
            if num_cols:
                d_col1, d_col2 = st.columns(2)
                with d_col1:
                    target_col = st.selectbox("Select Numeric Metric:", num_cols)
                with d_col2:
                    dist_color = st.selectbox("Split Distribution By:", ["None"] + filter_columns)
                
                color_arg = None if dist_color == "None" else dist_color
                fig_dist = px.histogram(df, x=target_col, color=color_arg, marginal="box", barmode="overlay", opacity=0.7)
                st.plotly_chart(fig_dist, use_container_width=True)
                
        else:
            # Categorical Distribution (Dual Axis Top/Bottom)
            c_col1, c_col2, c_col3, c_col4 = st.columns(4)
            with c_col1:
                cat_col = st.selectbox("Categorical Column:", filter_columns)
            with c_col2:
                num_target = st.selectbox("Line Metric (Median):", num_cols, index=num_cols.index('Rate (AED per SQM)') if 'Rate (AED per SQM)' in num_cols else 0)
            with c_col3:
                sort_type = st.selectbox("Sort By Volume:", ["Top", "Bottom"])
            with c_col4:
                n_records = st.selectbox("Number of Records (N):", [5, 10, 20, 50, 100], index=1)

            # Aggregate
            cat_agg = df.groupby(cat_col).agg(
                Count=(cat_col, 'count'),
                Median_Val=(num_target, 'median')
            ).reset_index()

            # Filter Top or Bottom N
            if sort_type == "Top":
                cat_agg = cat_agg.nlargest(n_records, 'Count')
            else:
                cat_agg = cat_agg.nsmallest(n_records, 'Count')
                # Optional: drop 0 counts for bottoms
                cat_agg = cat_agg[cat_agg['Count'] > 0]

            # Re-sort to display largest -> smallest nicely
            cat_agg = cat_agg.sort_values(by='Count', ascending=False)

            fig_cat = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Y1: Bars (Count)
            fig_cat.add_trace(
                go.Bar(x=cat_agg[cat_col], y=cat_agg['Count'], name="Transaction Count", marker_color='royalblue', opacity=0.7),
                secondary_y=False,
            )
            # Y2: Line (Median Num Val)
            fig_cat.add_trace(
                go.Scatter(x=cat_agg[cat_col], y=cat_agg['Median_Val'], name=f"Median {num_target}", mode='lines+markers', line=dict(color='firebrick', width=3)),
                secondary_y=True,
            )

            fig_cat.update_layout(title_text=f"{sort_type} {n_records} {cat_col} (Volume vs {num_target})", height=500, hovermode="x unified")
            fig_cat.update_yaxes(title_text="Transaction Count", secondary_y=False)
            fig_cat.update_yaxes(title_text=f"Median {num_target}", secondary_y=True)
            
            st.plotly_chart(fig_cat, use_container_width=True)

    # ==========================================
    # TAB 3: RELATIONSHIPS
    # ==========================================
    with tab3:
        st.subheader("Multivariate Relationships")
        if num_cols and filter_columns:
            r_col1, r_col2, r_col3, r_col4 = st.columns(4)
            with r_col1:
                x_axis = st.selectbox("X-Axis", num_cols, index=0)
            with r_col2:
                y_axis = st.selectbox("Y-Axis", num_cols, index=1 if len(num_cols) > 1 else 0)
            with r_col3:
                color_dim = st.selectbox("Color (Categorical)", filter_columns)
            with r_col4:
                size_dim = st.selectbox("Bubble Size (Numeric)", ["None"] + num_cols)

            chart_type = st.radio("Chart Type", ["Scatter Plot", "Density Heat Map"], horizontal=True)
            size_arg = None if size_dim == "None" else size_dim

            if chart_type == "Scatter Plot":
                fig_rel = px.scatter(df, x=x_axis, y=y_axis, color=color_dim, size=size_arg, opacity=0.6, height=600)
            else:
                fig_rel = px.density_heatmap(df, x=x_axis, y=y_axis, facet_col=color_dim, height=600)
                
            st.plotly_chart(fig_rel, use_container_width=True)

    # ==========================================
    # TAB 4: MAP VIEW
    # ==========================================
    with tab4:
        st.subheader("Geographical Distribution")
        if 'District' in df.columns and 'Rate (AED per SQM)' in df.columns:
            coord_mapping = {
                'Yas Island': {'lat': 24.496, 'lon': 54.603},
                'Al Reem Island': {'lat': 24.494, 'lon': 54.406},
                'Al Rahah': {'lat': 24.437, 'lon': 54.582},
                'Ghayathi': {'lat': 23.844, 'lon': 52.810},
                "Al Faqa'": {'lat': 24.733, 'lon': 55.616}
            }
            map_df = df.groupby('District').agg(Median_Rate=('Rate (AED per SQM)', 'median'), Transactions=('Rate (AED per SQM)', 'count')).reset_index()
            map_df['lat'] = map_df['District'].map(lambda x: coord_mapping.get(x, {'lat': 24.4539})['lat'])
            map_df['lon'] = map_df['District'].map(lambda x: coord_mapping.get(x, {'lon': 54.3773})['lon'])

            fig_map = px.scatter_mapbox(
                map_df, lat="lat", lon="lon", hover_name="District", 
                hover_data={"Median_Rate": ":.0f", "Transactions": True, "lat": False, "lon": False},
                color="Median_Rate", size="Transactions", color_continuous_scale=px.colors.sequential.Viridis, zoom=9, height=600
            )
            fig_map.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("Map requires 'District' and 'Rate (AED per SQM)' columns.")

    # ==========================================
    # TAB 5: CORRELATIONS (New Tab!)
    # ==========================================
    with tab5:
        st.subheader("Numerical Correlation Matrix")
        st.markdown("Understand how numerical fields interact. Values closer to **1** or **-1** indicate strong correlations.")
        
        if len(num_cols) > 1:
            # Calculate correlation matrix
            corr_matrix = df[num_cols].corr()
            
            fig_corr = px.imshow(
                corr_matrix, 
                text_auto=".2f", 
                aspect="auto", 
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1
            )
            fig_corr.update_layout(height=600)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Not enough numerical columns to generate a correlation matrix.")

    # ==========================================
    # TAB 6: DATA & SUMMARY
    # ==========================================
    with tab6:
        st.subheader("Filtered Data Summary")
        st.markdown("**Descriptive Statistics**")
        st.dataframe(df.describe().T.style.format("{:,.2f}"))
        
        st.markdown("**Raw Data** (Showing up to 1000 rows)")
        st.dataframe(df.head(1000))
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Filtered Data as CSV", data=csv, file_name='filtered_real_estate_data.csv', mime='text/csv')
