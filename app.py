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
        # Introduce some mock null values for demonstration
        df.loc[df.sample(frac=0.1).index, 'Property Sold Area (SQM)'] = np.nan
        df.loc[df.sample(frac=0.05).index, 'District'] = np.nan
        
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

st.sidebar.title("🛠️ Settings & Filters")

# ==========================================
# SIDEBAR: DATE FILTER (NEW)
# ==========================================
st.sidebar.subheader("📅 Time Frame Filter")
min_date = df_raw['Sale Application Date'].min().date()
max_date = df_raw['Sale Application Date'].max().date()

# Streamlit date_input returns a tuple of selected dates
selected_dates = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Apply the date filter immediately so all subsequent outliers/dropdowns use this time frame
if len(selected_dates) == 2:
    start_date, end_date = selected_dates
    df = df[(df['Sale Application Date'].dt.date >= start_date) & (df['Sale Application Date'].dt.date <= end_date)]
elif len(selected_dates) == 1:
    start_date = selected_dates[0]
    df = df[df['Sale Application Date'].dt.date >= start_date]

# ==========================================
# SIDEBAR: DYNAMIC OUTLIERS
# ==========================================
st.sidebar.subheader("1. Outlier Configuration")

outlier_scope = st.sidebar.radio(
    "Apply Outliers To:", 
    ["None (Keep All Data)", "Overall Data (Before Filters)", "Filtered Data (After Filters)"],
    index=0,
    help="None bypasses outliers. Overall applies rules globally. Filtered applies rules only to the data left after your category selections."
)

outlier_bounds = {}
st.sidebar.markdown("**Numerical Percentiles**")
for col in num_cols:
    with st.sidebar.expander(f"⚙️ {col} Limits"):
        if st.checkbox(f"Filter {col}", value=False, key=f"check_{col}"):
            lower, upper = st.slider(f"Percentile Range", 0.0, 100.0, (1.0, 99.0), 0.5, key=f"slider_{col}")
            outlier_bounds[col] = (lower / 100.0, upper / 100.0)

def apply_outliers(data):
    d = data.copy()
    for col, (low, high) in outlier_bounds.items():
        l_val = d[col].quantile(low)
        h_val = d[col].quantile(high)
        d = d[(d[col] >= l_val) & (d[col] <= h_val)]
    return d

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

if outlier_scope == "Filtered Data (After Filters)":
    df = apply_outliers(df)


# ==========================================
# MAIN DASHBOARD
# ==========================================
st.title("🏢 Real Estate Market Analysis")
st.markdown("Comprehensive insights into property transactions, trends, and distributions.")

# Top KPIs
if not df.empty:
    # Row 1 Metrics: Core Stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{len(df):,}")
    if 'Rate (AED per SQM)' in df.columns:
        col2.metric("Median Rate (AED/SQM)", f"{df['Rate (AED per SQM)'].median():,.0f}")
    if 'Property Sold Area (SQM)' in df.columns:
        col3.metric("Median Area (SQM)", f"{df['Property Sold Area (SQM)'].median():,.0f}")
    if 'Property Sale Price (AED)' in df.columns:
        col4.metric("Total Value (AED)", f"{df['Property Sale Price (AED)'].sum() / 1e9:,.2f} B")
        
    # Row 2 Metrics: Time Frame Boundaries (NEW)
    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
    if 'Sale Application Date' in df.columns:
        first_tx = df['Sale Application Date'].min().strftime('%d %b %Y')
        last_tx = df['Sale Application Date'].max().strftime('%d %b %Y')
        d_col1.metric("🗓️ First Recorded Sale", first_tx)
        d_col2.metric("⏱️ Latest Recorded Sale", last_tx)
else:
    st.warning("⚠️ No data matches the current filter criteria. Please adjust your filters.")

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📈 Trends", "📊 Distributions", "🔗 Relationships", "🗺️ District Map", "🧩 Correlations", "❓ Null Values", "🕒 First Transactions", "📋 Data & Summary"
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
    # TAB 2: DISTRIBUTIONS 
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
            c_col1, c_col2, c_col3, c_col4 = st.columns(4)
            with c_col1:
                cat_col = st.selectbox("Categorical Column:", filter_columns)
            with c_col2:
                num_target = st.selectbox("Line Metric (Median):", num_cols, index=num_cols.index('Rate (AED per SQM)') if 'Rate (AED per SQM)' in num_cols else 0)
            with c_col3:
                sort_type = st.selectbox("Sort By Volume:", ["Top", "Bottom"])
            with c_col4:
                n_records = st.selectbox("Number of Records (N):", [5, 10, 20, 50, 100], index=1)

            cat_agg = df.groupby(cat_col).agg(
                Count=(cat_col, 'count'),
                Median_Val=(num_target, 'median')
            ).reset_index()

            if sort_type == "Top":
                cat_agg = cat_agg.nlargest(n_records, 'Count')
            else:
                cat_agg = cat_agg.nsmallest(n_records, 'Count')
                cat_agg = cat_agg[cat_agg['Count'] > 0]

            cat_agg = cat_agg.sort_values(by='Count', ascending=False)
            fig_cat = make_subplots(specs=[[{"secondary_y": True}]])
            
            fig_cat.add_trace(go.Bar(x=cat_agg[cat_col], y=cat_agg['Count'], name="Transaction Count", marker_color='royalblue', opacity=0.7), secondary_y=False)
            fig_cat.add_trace(go.Scatter(x=cat_agg[cat_col], y=cat_agg['Median_Val'], name=f"Median {num_target}", mode='lines+markers', line=dict(color='firebrick', width=3)), secondary_y=True)

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
                x_axis = st.selectbox("X-Axis", filter_columns, index=0)
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
                'Al Reem Island': {'lat': 24.494, 'lon': 54.406},
                'Yas Island': {'lat': 24.496, 'lon': 54.603},
                'Al Saadiyat Island': {'lat': 24.539, 'lon': 54.437},
                'Al Maryah Island': {'lat': 24.502, 'lon': 54.389},
                'Al Rahah': {'lat': 24.437, 'lon': 54.582},
                'Khalifa City': {'lat': 24.417, 'lon': 54.583},
                'Al Reef': {'lat': 24.463, 'lon': 54.654},
                'Mohamed Bin Zayed City': {'lat': 24.348, 'lon': 54.549},
                'Al Shamkhah': {'lat': 24.394, 'lon': 54.717},
                'Bani Yas': {'lat': 24.305, 'lon': 54.625},
                'Zayed City': {'lat': 23.655, 'lon': 53.704},
                'Al Shahamah': {'lat': 24.531, 'lon': 54.685},
                'Al Bahyah': {'lat': 24.551, 'lon': 54.654},
                'Ramhan Island': {'lat': 24.501, 'lon': 54.501},
                'Fahid Island': {'lat': 24.512, 'lon': 54.536},
                'Al Jubail Island': {'lat': 24.542, 'lon': 54.485},
                'Al Bateen': {'lat': 24.453, 'lon': 54.345},
                'Al Khalidiyah': {'lat': 24.471, 'lon': 54.341},
                'Al Nahyan': {'lat': 24.475, 'lon': 54.381},
                'Al Mushrif': {'lat': 24.448, 'lon': 54.395},
                'Musaffah': {'lat': 24.356, 'lon': 54.515},
                'Ghayathi': {'lat': 23.844, 'lon': 52.810},
                "Al Faqa'": {'lat': 24.733, 'lon': 55.616},
                'Hili': {'lat': 24.283, 'lon': 55.772},
                'Al Jimi': {'lat': 24.246, 'lon': 55.748},
                'Al Danah': {'lat': 24.488, 'lon': 54.361},
                'Al Zahiyah': {'lat': 24.494, 'lon': 54.372},
                'Madinat Zayed': {'lat': 23.655, 'lon': 53.704},
                'Shakhbout City': {'lat': 24.397, 'lon': 54.636},
                'Nourai Island': {'lat': 24.615, 'lon': 54.481}
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
    # TAB 5: CORRELATIONS
    # ==========================================
    with tab5:
        st.subheader("Numerical Correlation Matrix")
        if len(num_cols) > 1:
            corr_matrix = df[num_cols].corr()
            fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            fig_corr.update_layout(height=600)
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Not enough numerical columns to generate a correlation matrix.")

    # ==========================================
    # TAB 6: NULL VALUES
    # ==========================================
    with tab6:
        st.subheader("Missing (Null) Values Analysis")
        st.markdown("This tab displays the percentage and count of missing values across all columns for the **currently filtered dataset**.")
        
        null_counts = df.isnull().sum().reset_index()
        null_counts.columns = ['Column', 'Missing Count']
        null_counts['% Missing'] = (null_counts['Missing Count'] / len(df)) * 100
        null_counts = null_counts[null_counts['Missing Count'] > 0].sort_values(by='% Missing', ascending=False)
        
        if not null_counts.empty:
            fig_nulls = px.bar(
                null_counts, x='Column', y='% Missing', text_auto='.2f', 
                title="Percentage of Missing Values by Column", color='% Missing', color_continuous_scale='Reds'
            )
            fig_nulls.update_layout(height=500)
            st.plotly_chart(fig_nulls, use_container_width=True)
            st.markdown("**Detailed Missing Value Table**")
            st.dataframe(null_counts.style.format({'% Missing': '{:.2f}%'}))
        else:
            st.success("✅ There are zero missing values in the currently selected data slice!")

    # ==========================================
    # TAB 7: FIRST TRANSACTIONS
    # ==========================================
    with tab7:
        st.subheader("First Transactions Analysis")
        st.markdown("Find the **very first (oldest) transaction** ever recorded for every category, and display the top newest ones among them.")
        
        rt_col1, rt_col2 = st.columns(2)
        with rt_col1:
            recent_col = st.selectbox("Group by (Category):", filter_columns, index=0)
        with rt_col2:
            limit_n_str = st.selectbox("Show Top N Newest Among Them:", ["5", "10", "20", "50", "All"], index=1)
        
        if 'Sale Application Date' in df.columns:
            first_df = df.sort_values(by='Sale Application Date', ascending=True).drop_duplicates(subset=[recent_col], keep='first')
            first_df = first_df.sort_values(by='Sale Application Date', ascending=False)
            
            if limit_n_str != "All":
                first_df = first_df.head(int(limit_n_str))
                
            if 'Rate (AED per SQM)' in df.columns:
                st.markdown(f"**Timeline of the First-Ever '{recent_col}' Sales**")
                fig_first_time = px.scatter(
                    first_df, x='Sale Application Date', y='Rate (AED per SQM)', color=recent_col,
                    hover_data=['Property Sale Price (AED)', 'Property Sold Area (SQM)', 'Property Type'],
                    title=f"First Transaction Recorded per {recent_col}", opacity=0.8
                )
                fig_first_time.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
                st.plotly_chart(fig_first_time, use_container_width=True)
            
            st.markdown(f"**Detailed Data: First transaction details for the selected '{recent_col}'**")
            display_cols = [recent_col, 'Sale Application Date'] + [c for c in ['Property Sale Price (AED)', 'Rate (AED per SQM)', 'Property Type', 'Sale Sequence'] if c in df.columns]
            st.dataframe(first_df[display_cols].reset_index(drop=True))
        else:
            st.warning("Cannot find 'Sale Application Date' column to calculate the first transactions.")

    # ==========================================
    # TAB 8: DATA & SUMMARY
    # ==========================================
    with tab8:
        st.subheader("Filtered Data Summary")
        st.markdown("**Descriptive Statistics**")
        st.dataframe(df.describe().T.style.format("{:,.2f}"))
        
        st.markdown("**Raw Data** (Showing up to 1000 rows)")
        st.dataframe(df.head(1000))
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Filtered Data as CSV", data=csv, file_name='filtered_real_estate_data.csv', mime='text/csv')
