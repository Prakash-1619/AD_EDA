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
        df.loc[df.sample(frac=0.1).index, 'Property Sold Area (SQM)'] = np.nan
        df.loc[df.sample(frac=0.05).index, 'District'] = np.nan
        
        df['Rate (AED per SQM)'] = df['Property Sale Price (AED)'] / df['Property Sold Area (SQM)']
        df['Sale Application Date'] = pd.to_datetime(df['Sale Application Date'])
        return df

df_raw = load_data()
df = df_raw.copy()

# Automatically categorize structural attributes
num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
filter_columns = [col for col in cat_cols if col not in ['Sale Application Date', 'YearMonth']]

st.sidebar.title("🛠️ Settings & Filters")

# ==========================================
# SIDEBAR: DATE FILTER
# ==========================================
st.sidebar.subheader("📅 Time Frame Filter")
min_date = df_raw['Sale Application Date'].min().date()
max_date = df_raw['Sale Application Date'].max().date()

selected_dates = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(selected_dates) == 2:
    start_date, end_date = selected_dates
    df = df[(df['Sale Application Date'].dt.date >= start_date) & (df['Sale Application Date'].dt.date <= end_date)]
elif len(selected_dates) == 1:
    start_date = selected_dates[0]
    df = df[df['Sale Application Date'].dt.date >= start_date]

# ==========================================
# SIDEBAR: DYNAMIC OUTLIERS (GROUPED CAPABLE)
# ==========================================
st.sidebar.subheader("1. Outlier Configuration")

outlier_scope = st.sidebar.radio(
    "Apply Outliers To:", 
    ["None (Keep All Data)", "Overall Data (Before Filters)", "Filtered Data (After Filters)"],
    index=0,
    help="None bypasses outliers. Overall applies rules globally. Filtered applies rules only to the data left after category selections."
)

outlier_grouping = st.sidebar.multiselect(
    "Calculate Outliers Within Categories (Optional):",
    options=filter_columns,
    default=[],
    help="Select categories to group by (e.g., 'Property Type' profiles outliers per item layout class independently)."
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
        if outlier_grouping:
            q_low = d.groupby(outlier_grouping)[col].transform(lambda x: x.quantile(low))
            q_high = d.groupby(outlier_grouping)[col].transform(lambda x: x.quantile(high))
            d = d[(d[col] >= q_low) & (d[col] <= q_high)]
        else:
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
# MAIN DASHBOARD ENTRY
# ==========================================
st.title("🏢 Real Estate Market Analysis Engine")
st.markdown("Dynamic predictive modeling framework sandbox and exploratory analysis platform.")

# Universal Top-Line Metrics Banner
if not df.empty:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", f"{len(df):,}")
    if 'Rate (AED per SQM)' in df.columns:
        col2.metric("Median Rate (AED/SQM)", f"{df['Rate (AED per SQM)'].median():,.0f}")
    if 'Property Sold Area (SQM)' in df.columns:
        col3.metric("Median Area (SQM)", f"{df['Property Sold Area (SQM)'].median():,.0f}")
    if 'Property Sale Price (AED)' in df.columns:
        col4.metric("Total Market Value", f"AED {df['Property Sale Price (AED)'].sum() / 1e9:,.2f} B")
else:
    st.warning("⚠️ No records match current parameters. Soften filter restrictions.")

# Define Re-ordered System Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📋 Summary Profile", "📈 Trends", "📊 Distributions", "🔗 Relationships", "🗺️ District Map", "🧩 Correlations", "❓ Null Footprint", "🕒 First Transactions", "📋 Raw Snapshot"
])

if not df.empty:
    # ==========================================
    # TAB 1: SUMMARY PROFILE (NEW)
    # ==========================================
    with tab1:
        st.subheader("Dataset Architectural Blueprint")
        st.markdown("High-level overview of structural properties, time-span baselines, and modeling parameters.")
        
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            st.markdown("### Operational Dimensions")
            metrics_summary = pd.DataFrame({
                "Property Dimension Metric": ["Total Volume Records", "Feature Attribute Count", "Temporal Start Date", "Temporal End Date"],
                "Current Filtered State": [
                    f"{len(df):,}", 
                    f"{len(df.columns)} Columns", 
                    df['Sale Application Date'].min().strftime('%d %b %Y'), 
                    df['Sale Application Date'].max().strftime('%d %b %Y')
                ],
                "Raw Global Baseline": [
                    f"{len(df_raw):,}", 
                    f"{len(df_raw.columns)} Columns", 
                    df_raw['Sale Application Date'].min().strftime('%d %b %Y'), 
                    df_raw['Sale Application Date'].max().strftime('%d %b %Y')
                ]
            })
            st.table(metrics_summary)
            
        with s_col2:
            st.markdown("### Feature Classification Targets")
            target_framework = pd.DataFrame({
                "Variable Paradigm Role": ["Primary Modeling Target Variable", "Secondary Normalization Target", "Temporal Index Axis"],
                "Field Column Assigned": ["Property Sale Price (AED)", "Rate (AED per SQM)", "Sale Application Date"],
                "Data Matrix Classification": ["Continuous Numerical Float", "Derived Unit Continuous Float", "Datetime Temporal Stamp"]
            })
            st.table(target_framework)
            
        st.markdown("### Field Level Schema & Structural Sparsity Profile")
        schema_df = pd.DataFrame({
            "Data Type (dtype)": df.dtypes.astype(str),
            "Non-Null Observations": df.notnull().sum(),
            "Null Records Count": df.isnull().sum(),
            "Sparsity Percent (% Missing)": (df.isnull().sum() / len(df)) * 100,
            "Distinct Unique Elements": df.nunique()
        })
        st.dataframe(schema_df.style.format({'Sparsity Percent (% Missing)': '{:.2f}%', 'Non-Null Observations': '{:,}', 'Null Records Count': '{:,}', 'Distinct Unique Elements': '{:,}'}))

    # ==========================================
    # TAB 2: TRENDS ENGINE (ENHANCED)
    # ==========================================
    with tab2:
        st.subheader("Temporal Market Vectors")
        
        t_cfg1, t_cfg2 = st.columns(2)
        with t_cfg1:
            time_grain = st.selectbox(
                "Time Frame Granularity Scale:", 
                options=["Yearly", "Monthly", "Weekly", "Daily"], 
                index=1
            )
        with t_cfg2:
            trend_mode = st.radio(
                "Trend Aggregation Strategy:", 
                options=["Overall Market (Unified Track)", "By Categorical Feature Breakdown"], 
                horizontal=True
            )
            
        # Compute corresponding period labels mapping dynamically
        t_df = df.copy()
        if time_grain == "Yearly":
            t_df['Period_Axis'] = t_df['Sale Application Date'].dt.to_period('Y').astype(str)
        elif time_grain == "Monthly":
            t_df['Period_Axis'] = t_df['Sale Application Date'].dt.to_period('M').astype(str)
        elif time_grain == "Weekly":
            t_df['Period_Axis'] = t_df['Sale Application Date'].dt.to_period('W').astype(str).str.split('/').str[0]
        else:
            t_df['Period_Axis'] = t_df['Sale Application Date'].dt.date.astype(str)

        if 'Rate (AED per SQM)' in t_df.columns:
            fig_trends = make_subplots(specs=[[{"secondary_y": True}]])
            
            if trend_mode == "Overall Market (Unified Track)":
                trend_agg = t_df.groupby('Period_Axis').agg(
                    Median_Rate=('Rate (AED per SQM)', 'median'),
                    Tx_Count=('Rate (AED per SQM)', 'count')
                ).reset_index().sort_values(by='Period_Axis')
                
                fig_trends.add_trace(
                    go.Bar(x=trend_agg['Period_Axis'], y=trend_agg['Tx_Count'], name="Volume Count", opacity=0.4, marker_color='grey'),
                    secondary_y=False
                )
                fig_trends.add_trace(
                    go.Scatter(x=trend_agg['Period_Axis'], y=trend_agg['Median_Rate'], name="Market Median Rate", mode='lines+markers', line=dict(color='indigo', width=3)),
                    secondary_y=True
                )
            else:
                grouping_options = filter_columns if filter_columns else ["Property Type"]
                trend_group_col = st.selectbox("Group Sub-Trends Axis By:", grouping_options, index=0)
                
                trend_agg = t_df.groupby(['Period_Axis', trend_group_col]).agg(
                    Median_Rate=('Rate (AED per SQM)', 'median'),
                    Tx_Count=('Rate (AED per SQM)', 'count')
                ).reset_index().sort_values(by='Period_Axis')
                
                for category in trend_agg[trend_group_col].unique():
                    cat_df = trend_agg[trend_agg[trend_group_col] == category]
                    fig_trends.add_trace(
                        go.Bar(x=cat_df['Period_Axis'], y=cat_df['Tx_Count'], name=f"Vol: {category}", opacity=0.5),
                        secondary_y=False,
                    )
                    fig_trends.add_trace(
                        go.Scatter(x=cat_df['Period_Axis'], y=cat_df['Median_Rate'], name=f"Rate: {category}", mode='lines+markers'),
                        secondary_y=True,
                    )

            fig_trends.update_layout(title_text=f"Transaction Volume Metrics vs Valuation Index ({time_grain} Resolution)", barmode='stack', height=600, hovermode="x unified")
            fig_trends.update_yaxes(title_text="Transaction Volume (Count)", secondary_y=False)
            fig_trends.update_yaxes(title_text="Valuation Index (Median AED/SQM)", secondary_y=True)
            st.plotly_chart(fig_trends, use_container_width=True)

    # ==========================================
    # TAB 3: DISTRIBUTIONS
    # ==========================================
    with tab2: # Note: st.tabs structural containers maintain separate block assignment matching indices seamlessly
        pass
    with tab3:
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

            cat_agg = df.groupby(cat_col).agg(Count=(cat_col, 'count'), Median_Val=(num_target, 'median')).reset_index()

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
    # TAB 4: MAP VIEW
    # ==========================================
    with tab4:
        st.subheader("Geographical Distribution")
        if 'District' in df.columns and 'Rate (AED per SQM)' in df.columns:
            coord_mapping = {
                "Al Faqa'": {'lat': 24.7165, 'lon': 55.6216}, 'Yas Island': {'lat': 24.4864, 'lon': 54.6091},
                'Ghayathi': {'lat': 23.8993, 'lon': 52.8018}, 'Al Rahah': {'lat': 24.4553, 'lon': 54.6154},
                'Al Saadiyat Island': {'lat': 24.5451, 'lon': 54.4397}, 'Al Shamkhah': {'lat': 24.3783, 'lon': 54.7043},
                'Al Maryah Island': {'lat': 24.5030, 'lon': 54.3900}, 'Al Reem Island': {'lat': 24.4954, 'lon': 54.4052},
                'Mohamed Bin Zayed City': {'lat': 24.3404, 'lon': 54.5516}, 'Al Reef': {'lat': 24.4664, 'lon': 54.6569},
                'Rabdan': {'lat': 24.4055, 'lon': 54.5058}, 'Al Aamerah': {'lat': 24.2332, 'lon': 55.5543},
                'Khalifa City': {'lat': 24.4201, 'lon': 54.5750}, 'Ain Al Faydah': {'lat': 24.0777, 'lon': 55.7116},
                'Al Khibeesi': {'lat': 24.2298, 'lon': 55.6990}, 'Al Bahyah': {'lat': 24.5565, 'lon': 54.6642},
                'Zayed City': {'lat': 23.6515, 'lon': 53.7020}, 'Al Jimi': {'lat': 24.2510, 'lon': 55.7355},
                'Al Hidayriyyat': {'lat': 24.3665, 'lon': 54.4096}, 'Al Danah': {'lat': 24.4853, 'lon': 54.3699},
                'Al Tiwayya': {'lat': 24.2512, 'lon': 55.6965}, 'Ramhan Island': {'lat': 24.5429, 'lon': 54.5270},
                'Al Layyan': {'lat': 24.7789, 'lon': 54.9822}, 'Al Khalidiyah': {'lat': 24.4693, 'lon': 54.3488},
                'Al Samhah': {'lat': 24.6660, 'lon': 54.7466}, 'Fahid Island': {'lat': 24.5091, 'lon': 54.5485},
                'Hili': {'lat': 24.3044, 'lon': 55.7751}, 'Al Shahamah': {'lat': 24.5522, 'lon': 54.6810},
                'Nourai Island': {'lat': 24.6147, 'lon': 54.4868}, 'Sweihan': {'lat': 24.3826, 'lon': 55.4677}
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

    # ==========================================
    # TAB 6: NULL FOOTPRINT
    # ==========================================
    with tab6:
        st.subheader("Missing (Null) Values Analysis")
        null_counts = df.isnull().sum().reset_index()
        null_counts.columns = ['Column', 'Missing Count']
        null_counts['% Missing'] = (null_counts['Missing Count'] / len(df)) * 100
        null_counts = null_counts[null_counts['Missing Count'] > 0].sort_values(by='% Missing', ascending=False)
        
        if not null_counts.empty:
            fig_nulls = px.bar(null_counts, x='Column', y='% Missing', text_auto='.2f', color='% Missing', color_continuous_scale='Reds')
            st.plotly_chart(fig_nulls, use_container_width=True)
            st.dataframe(null_counts.style.format({'% Missing': '{:.2f}%'}))
        else:
            st.success("✅ Clean Dataset Matrix! No missing fields identified in current filter state.")

    # ==========================================
    # TAB 7: FIRST TRANSACTIONS
    # ==========================================
    with tab7:
        st.subheader("First Transactions Analysis")
        rt_col1, rt_col2 = st.columns(2)
        with rt_col1:
            recent_col = st.selectbox("Group Axis Categories:", filter_columns, index=0)
        with rt_col2:
            limit_n_str = st.selectbox("Show Top N Newest Establishments among First Transactions:", ["5", "10", "20", "50", "All"], index=1)
        
        if 'Sale Application Date' in df.columns:
            first_df = df.sort_values(by='Sale Application Date', ascending=True).drop_duplicates(subset=[recent_col], keep='first')
            first_df = first_df.sort_values(by='Sale Application Date', ascending=False)
            
            if limit_n_str != "All":
                first_df = first_df.head(int(limit_n_str))
                
            if 'Rate (AED per SQM)' in df.columns:
                fig_first_time = px.scatter(
                    first_df, x='Sale Application Date', y='Rate (AED per SQM)', color=recent_col,
                    hover_data=['Property Sale Price (AED)', 'Property Sold Area (SQM)'], title="Establishment Sale Trajectory Baseline Blueprint"
                )
                fig_first_time.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
                st.plotly_chart(fig_first_time, use_container_width=True)
            
            display_cols = [recent_col, 'Sale Application Date'] + [c for c in ['Property Sale Price (AED)', 'Rate (AED per SQM)', 'Property Type'] if c in df.columns]
            st.dataframe(first_df[display_cols].reset_index(drop=True))

    # ==========================================
    # TAB 8 & 9: RELATIONSHIPS & RAW SNAPSHOT
    # ==========================================
    with tab8:
        st.subheader("Multivariate Cross Relationships")
        if num_cols and filter_columns:
            r_col1, r_col2, r_col3, r_col4 = st.columns(4)
            with r_col1: x_axis = st.selectbox("X Axis Select:", filter_columns, index=0)
            with r_col2: y_axis = st.selectbox("Y Axis Select:", num_cols, index=1 if len(num_cols) > 1 else 0)
            with r_col3: color_dim = st.selectbox("Color Mapping Layer:", filter_columns)
            with r_col4: size_dim = st.selectbox("Marker Dimension Bubble:", ["None"] + num_cols)
            size_arg = None if size_dim == "None" else size_dim
            fig_rel = px.scatter(df, x=x_axis, y=y_axis, color=color_dim, size=size_arg, opacity=0.6, height=600)
            st.plotly_chart(fig_rel, use_container_width=True)

    with tab9:
        st.subheader("Data Export Matrix Hub")
        st.dataframe(df.describe().T.style.format("{:,.2f}"))
        st.dataframe(df.head(1000))
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Engineered File Slice (CSV)", data=csv, file_name='filtered_real_estate_data.csv', mime='text/csv')
