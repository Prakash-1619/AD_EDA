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
            'Asset Class': np.random.choice(['residential', 'commercial'], 10000, p=[0.9, 0.1]),
            'Sale Application Date': np.random.choice(dates, 10000),
            'Property Type': np.random.choice(prop_types, 10000),
            'District': np.random.choice(districts, 10000),
            'Community': np.random.choice(['Comm A', 'Comm B', 'Comm C'], 10000),
            'Sale Sequence': np.random.choice(['primary', 'secondary'], 10000),
            'Property Sold Area (SQM)': np.random.exponential(scale=100, size=10000) + 20,
            'Property Sale Price (AED)': np.random.exponential(scale=1500000, size=10000) + 500000,
            'Other Numeric': np.random.randint(1, 100, 10000)
        })
        df.loc[df.sample(frac=0.1).index, 'Property Sold Area (SQM)'] = np.nan
        df.loc[df.sample(frac=0.05).index, 'District'] = np.nan
        
        df['Rate (AED per SQM)'] = df['Property Sale Price (AED)'] / df['Property Sold Area (SQM)']
        df['Sale Application Date'] = pd.to_datetime(df['Sale Application Date'])
        return df

df_raw = load_data()

# Automatically categorize structural attributes (Base)
all_num_cols = df_raw.select_dtypes(include=['float64', 'int64']).columns.tolist()
cat_cols = df_raw.select_dtypes(include=['object', 'category']).columns.tolist()
all_filter_cols = [col for col in cat_cols if col not in ['Sale Application Date', 'YearMonth']]

# ==========================================
# SIDEBAR: MASTER VIEW TOGGLE
# ==========================================
st.sidebar.title("👁️ Dashboard Mode")
view_mode = st.sidebar.radio(
    "Select Operating View:",
    ["Abu Dhabi Real Estate", "Dynamic EDA"],
    help="View 1 applies strict default outlier hygiene grouped by asset/type/district. View 2 unlocks dynamic configuration."
)
st.sidebar.markdown("---")

# ==========================================
# SIDEBAR: UNIVERSAL FILTERS (Date & Cascading)
# ==========================================
st.sidebar.subheader("📅 Time Frame Filter")
min_date = df_raw['Sale Application Date'].min().date()
max_date = df_raw['Sale Application Date'].max().date()

selected_dates = st.sidebar.date_input("Select Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

# Apply Date Filter
df_base = df_raw.copy()
if len(selected_dates) == 2:
    df_base = df_base[(df_base['Sale Application Date'].dt.date >= selected_dates[0]) & (df_base['Sale Application Date'].dt.date <= selected_dates[1])]
elif len(selected_dates) == 1:
    df_base = df_base[df_base['Sale Application Date'].dt.date >= selected_dates[0]]

st.sidebar.subheader("🔍 Cascading Filters")
for col in all_filter_cols:
    available_options = sorted(df_base[col].dropna().astype(str).unique().tolist())
    selected_values = st.sidebar.multiselect(f"Filter {col}", options=available_options, default=[])
    if selected_values:
        df_base = df_base[df_base[col].astype(str).isin(selected_values)]

# ==========================================
# REUSABLE DASHBOARD ENGINE (TABS 1-8)
# ==========================================
def render_dashboard(df_active, df_raw_ref, num_cols, filter_cols, is_view_1):
    # Top-Line Metrics
    col1, col2,  = st.columns(4) # col3, col4
    col1.metric("Total Transactions", f"{len(df_active):,}")
    if 'Property Sale Price (AED)' in df_active.columns:
        col4.metric("Total Market Value", f"AED {df_active['Property Sale Price (AED)'].sum() / 1e9:,.2f} B")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📋 Summary Profile", "📈 Trends", "📊 Distributions", "🗺️ District Map", "🧩 Correlations", "🕒 First Transactions", "🔗 Relationships", "📋 Raw Snapshot"
    ])

    # --- TAB 1: SUMMARY ---
    with tab1:
        st.subheader("Dataset Architectural Blueprint")
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            st.markdown("### Operational Dimensions")
            metrics_summary = pd.DataFrame({
                "Property Dimension Metric": ["Total Volume Records", "Feature Attribute Count", "Temporal Start Date", "Temporal End Date"],
                "Current Filtered State": [f"{len(df_active):,}", f"{len(df_active.columns)} Columns", df_active['Sale Application Date'].min().strftime('%d %b %Y'), df_active['Sale Application Date'].max().strftime('%d %b %Y')],
                "Raw Global Baseline": [f"{len(df_raw_ref):,}", f"{len(df_raw_ref.columns)} Columns", df_raw_ref['Sale Application Date'].min().strftime('%d %b %Y'), df_raw_ref['Sale Application Date'].max().strftime('%d %b %Y')]
            })
            st.table(metrics_summary)
            
            if is_view_1:
                st.info(f"**Outlier Impact Summary:** The default 1%/99% grouped cleaning rules removed **{len(df_raw_ref) - len(df_active):,}** extreme outlier records from your selected time frame and filters.")
                
        with s_col2:
            st.markdown("### Feature Classification Targets")
            target_framework = pd.DataFrame({
                "Variable Paradigm Role": ["Primary Modeling Target Variable", "Secondary Normalization Target", "Temporal Index Axis"],
                "Field Column Assigned": ["Rate (AED per SQM)", "Property Sale Price (AED)", "Sale Application Date"],
            })
            st.table(target_framework)
            
        st.markdown("### Data Summary")
        schema_df = pd.DataFrame({
            "Data Type": df_active.dtypes.astype(str),
            "Non-Null": df_active.notnull().sum(),
            "Null Records": df_active.isnull().sum(),
            "% Missing": (df_active.isnull().sum() / len(df_active)) * 100,
            "Unique Elements": df_active.nunique()
        })
        st.dataframe(schema_df.style.format({'% Missing': '{:.2f}%'}))

    # --- TAB 2: TRENDS ---
    with tab2:
        st.subheader("Temporal Market Vectors & Dynamic Growth Trackers")
        t_cfg1, t_cfg2, t_cfg3 = st.columns(3)
        with t_cfg1:
            time_grain = st.selectbox("Granularity Scale:", ["Yearly", "Monthly", "Weekly", "Daily"], index=1)
        with t_cfg2:
            trend_mode = st.radio("Aggregation Strategy:", ["Overall Market (Unified Track)", "By Categorical Feature Breakdown"], horizontal=True)
        with t_cfg3:
            label_suffix = "YoY" if time_grain == "Yearly" else "MoM" if time_grain == "Monthly" else "WoW" if time_grain == "Weekly" else "DoD"
            calc_growth = st.checkbox(f"Show Growth Rate (% {label_suffix})", value=False)
            
        t_df = df_active.copy()
        if time_grain == "Yearly": t_df['Period_Axis'] = t_df['Sale Application Date'].dt.to_period('Y').astype(str)
        elif time_grain == "Monthly": t_df['Period_Axis'] = t_df['Sale Application Date'].dt.to_period('M').astype(str)
        elif time_grain == "Weekly": t_df['Period_Axis'] = t_df['Sale Application Date'].dt.to_period('W').astype(str).str.split('/').str[0]
        else: t_df['Period_Axis'] = t_df['Sale Application Date'].dt.date.astype(str)

        if 'Rate (AED per SQM)' in t_df.columns:
            fig_trends = make_subplots(specs=[[{"secondary_y": True}]])
            if trend_mode == "Overall Market (Unified Track)":
                trend_agg = t_df.groupby('Period_Axis').agg(Median_Rate=('Rate (AED per SQM)', 'median'), Tx_Count=('Rate (AED per SQM)', 'count')).reset_index().sort_values(by='Period_Axis')
                if calc_growth:
                    trend_agg['Rate_Metric'] = trend_agg['Median_Rate'].pct_change() * 100
                    trend_agg['Count_Metric'] = trend_agg['Tx_Count'].pct_change() * 100
                    y1_title, y2_title = f"Volume Growth (% {label_suffix})", f"Rate Growth (% {label_suffix})"
                else:
                    trend_agg['Rate_Metric'] = trend_agg['Median_Rate']
                    trend_agg['Count_Metric'] = trend_agg['Tx_Count']
                    y1_title, y2_title = "Transaction Volume (Count)", "Valuation Index (Median Rate)"
                
                fig_trends.add_trace(go.Bar(x=trend_agg['Period_Axis'], y=trend_agg['Count_Metric'], name=y1_title, opacity=0.4, marker_color='grey'), secondary_y=False)
                fig_trends.add_trace(go.Scatter(x=trend_agg['Period_Axis'], y=trend_agg['Rate_Metric'], name=y2_title, mode='lines+markers', line=dict(color='indigo', width=3)), secondary_y=True)
            else:
                trend_group_col = st.selectbox("Group Sub-Trends By:", filter_cols, index=0)
                trend_agg = t_df.groupby(['Period_Axis', trend_group_col]).agg(Median_Rate=('Rate (AED per SQM)', 'median'), Tx_Count=('Rate (AED per SQM)', 'count')).reset_index().sort_values(by='Period_Axis')
                
                if calc_growth:
                    trend_agg['Rate_Metric'] = trend_agg.groupby(trend_group_col)['Median_Rate'].pct_change() * 100
                    trend_agg['Count_Metric'] = trend_agg.groupby(trend_group_col)['Tx_Count'].pct_change() * 100
                    y1_title, y2_title = f"Volume Growth (% {label_suffix})", f"Rate Growth (% {label_suffix})"
                else:
                    trend_agg['Rate_Metric'] = trend_agg['Median_Rate']
                    trend_agg['Count_Metric'] = trend_agg['Tx_Count']
                    y1_title, y2_title = "Transaction Volume (Count)", "Valuation Index (Median Rate)"
                
                for category in trend_agg[trend_group_col].unique():
                    cat_df = trend_agg[trend_agg[trend_group_col] == category]
                    fig_trends.add_trace(go.Bar(x=cat_df['Period_Axis'], y=cat_df['Count_Metric'], name=f"{y1_title.split(' ')[0]}: {category}", opacity=0.5), secondary_y=False)
                    fig_trends.add_trace(go.Scatter(x=cat_df['Period_Axis'], y=cat_df['Rate_Metric'], name=f"{y2_title.split(' ')[0]}: {category}", mode='lines+markers'), secondary_y=True)

            fig_trends.update_layout(title_text=f"Market Metric Trajectory ({time_grain})", barmode='stack', height=600, hovermode="x unified")
            st.plotly_chart(fig_trends, use_container_width=True)

    # --- TAB 3: DISTRIBUTIONS ---
    with tab3:
        st.subheader("Structural Data Distributions")
        dist_type = st.radio("Choose Distribution Type:", ["Pareto Analysis", "Categorical (Categories vs Values)", "Numerical (Histograms)"], horizontal=True, index=0)
        
        if dist_type == "Pareto Analysis":
            pareto_opts = [c for c in ['District', 'Community'] if c in filter_cols] + [c for c in filter_cols if c not in ['District', 'Community']]
            p_col = st.selectbox("Dimension For Pareto (80/20 Rule):", options=pareto_opts)
            
            p_agg = df_active[p_col].value_counts().reset_index()
            p_agg.columns = [p_col, 'Count']
            p_agg['Cumulative_Percentage'] = (p_agg['Count'].cumsum() / p_agg['Count'].sum()) * 100
            
            fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
            fig_pareto.add_trace(go.Bar(x=p_agg[p_col], y=p_agg['Count'], name="Volume", marker_color='teal'), secondary_y=False)
            fig_pareto.add_trace(go.Scatter(x=p_agg[p_col], y=p_agg['Cumulative_Percentage'], name="Cumulative %", marker=dict(color='orange'), mode='lines+markers', line=dict(width=3)), secondary_y=True)
            fig_pareto.add_shape(type="line", x0=0, x1=len(p_agg)-1, y0=80, y1=80, line=dict(color="red", width=2, dash="dash"), secondary_y=True)
            
            fig_pareto.update_layout(title_text=f"Pareto Vector: {p_col}", height=500, hovermode="x unified")
            fig_pareto.update_yaxes(title_text="Cumulative Impact (%)", secondary_y=True, range=[0, 105])
            st.plotly_chart(fig_pareto, use_container_width=True)

        elif dist_type == "Numerical (Histograms)":
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                target_col = st.selectbox("Select Numeric Metric:", num_cols)
            with d_col2:
                dist_color = st.selectbox("Split Distribution By:", ["None"] + filter_cols)
            
            color_arg = None if dist_color == "None" else dist_color
            
            if is_view_1:
                show_before = st.checkbox("Overlay 'Before Cleaning' Baseline", value=False, help="Show the raw data histogram behind the cleaned data to see exactly what the 1%/99% rules removed.")
                if show_before:
                    df_comp_clean = df_active[[target_col] + ([color_arg] if color_arg else [])].copy()
                    df_comp_clean['State'] = 'After Cleaning (Cleaned)'
                    df_comp_raw = df_raw_ref[[target_col] + ([color_arg] if color_arg else [])].copy()
                    df_comp_raw['State'] = 'Before Cleaning (Raw)'
                    df_combined = pd.concat([df_comp_raw, df_comp_clean])
                    
                    # Ensure color mapping remains correct if splitting
                    color_mapping = "State" if color_arg is None else color_arg
                    facet_mapping = None if color_arg is None else "State"
                    
                    fig_dist = px.histogram(df_combined, x=target_col, color=color_mapping, facet_row=facet_mapping, marginal="box", barmode="overlay", opacity=0.6)
                else:
                    fig_dist = px.histogram(df_active, x=target_col, color=color_arg, marginal="box", barmode="overlay", opacity=0.7)
            else:
                fig_dist = px.histogram(df_active, x=target_col, color=color_arg, marginal="box", barmode="overlay", opacity=0.7)
                
            st.plotly_chart(fig_dist, use_container_width=True)
                
        else:
            c_col1, c_col2, c_col3, c_col4 = st.columns(4)
            with c_col1: cat_col = st.selectbox("Categorical Column:", filter_cols)
            with c_col2: num_target = st.selectbox("Line Metric (Median):", num_cols)
            with c_col3: sort_type = st.selectbox("Sort By Volume:", ["Top", "Bottom"])
            with c_col4: n_records = st.selectbox("Number of Records:", [5, 10, 20, 50, 100], index=1)

            cat_agg = df_active.groupby(cat_col).agg(Count=(cat_col, 'count'), Median_Val=(num_target, 'median')).reset_index()
            cat_agg = cat_agg.nlargest(n_records, 'Count') if sort_type == "Top" else cat_agg.nsmallest(n_records, 'Count')
            cat_agg = cat_agg[cat_agg['Count'] > 0].sort_values(by='Count', ascending=False)
            
            fig_cat = make_subplots(specs=[[{"secondary_y": True}]])
            fig_cat.add_trace(go.Bar(x=cat_agg[cat_col], y=cat_agg['Count'], name="Volume", marker_color='royalblue'), secondary_y=False)
            fig_cat.add_trace(go.Scatter(x=cat_agg[cat_col], y=cat_agg['Median_Val'], name=f"Median {num_target}", mode='lines+markers', line=dict(color='firebrick', width=3)), secondary_y=True)
            fig_cat.update_layout(title_text=f"{sort_type} {n_records} {cat_col} (Volume vs {num_target})", height=500, hovermode="x unified")
            st.plotly_chart(fig_cat, use_container_width=True)

    # --- TAB 4: MAP VIEW ---
    with tab4:
        st.subheader("Geographical Distribution")
        if 'District' in df_active.columns and 'Rate (AED per SQM)' in df_active.columns:
            coord_mapping = {"Al Faqa'": {'lat': 24.7165, 'lon': 55.6216}, 'Yas Island': {'lat': 24.4864, 'lon': 54.6091}, 'Ghayathi': {'lat': 23.8993, 'lon': 52.8018}, 'Al Rahah': {'lat': 24.4553, 'lon': 54.6154}, 'Al Saadiyat Island': {'lat': 24.5451, 'lon': 54.4397}, 'Al Shamkhah': {'lat': 24.3783, 'lon': 54.7043}, 'Al Maryah Island': {'lat': 24.5030, 'lon': 54.3900}, 'Al Reem Island': {'lat': 24.4954, 'lon': 54.4052}, 'Mohamed Bin Zayed City': {'lat': 24.3404, 'lon': 54.5516}, 'Al Reef': {'lat': 24.4664, 'lon': 54.6569}, 'Rabdan': {'lat': 24.4055, 'lon': 54.5058}, 'Khalifa City': {'lat': 24.4201, 'lon': 54.5750}}
            map_df = df_active.groupby('District').agg(Median_Rate=('Rate (AED per SQM)', 'median'), Transactions=('Rate (AED per SQM)', 'count')).reset_index()
            map_df['lat'] = map_df['District'].map(lambda x: coord_mapping.get(x, {'lat': 24.4539})['lat'])
            map_df['lon'] = map_df['District'].map(lambda x: coord_mapping.get(x, {'lon': 54.3773})['lon'])

            fig_map = px.scatter_mapbox(map_df, lat="lat", lon="lon", hover_name="District", hover_data={"Median_Rate": ":.0f", "Transactions": True, "lat": False, "lon": False}, color="Median_Rate", size="Transactions", color_continuous_scale=px.colors.sequential.Viridis, zoom=9, height=600)
            fig_map.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig_map, use_container_width=True)

    # --- TAB 5: CORRELATIONS ---
    with tab5:
        st.subheader("Numerical Correlation Matrix")
        if len(num_cols) > 1:
            corr_matrix = df_active[num_cols].corr()
            fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
            st.plotly_chart(fig_corr, use_container_width=True)

    # --- TAB 6: FIRST TRANSACTIONS ---
    with tab6:
        st.subheader("First Transactions Analysis")
        rt_col1, rt_col2 = st.columns(2)
        with rt_col1: recent_col = st.selectbox("Group Axis Categories:", filter_cols, index=0)
        with rt_col2: limit_n_str = st.selectbox("Show Top N Newest Establishments:", ["5", "10", "20", "50", "All"], index=1)
        
        if 'Sale Application Date' in df_active.columns:
            first_df = df_active.sort_values(by='Sale Application Date', ascending=True).drop_duplicates(subset=[recent_col], keep='first').sort_values(by='Sale Application Date', ascending=False)
            if limit_n_str != "All": first_df = first_df.head(int(limit_n_str))
                
            if 'Rate (AED per SQM)' in df_active.columns:
                fig_first_time = px.scatter(first_df, x='Sale Application Date', y='Rate (AED per SQM)', color=recent_col, hover_data=['Property Sale Price (AED)'])
                fig_first_time.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
                st.plotly_chart(fig_first_time, use_container_width=True)

    # --- TAB 7: RELATIONSHIPS ---
    with tab7:
        st.subheader("Multivariate Cross Relationships")
        if num_cols and filter_cols:
            r_col1, r_col2, r_col3, r_col4 = st.columns(4)
            with r_col1: x_axis = st.selectbox("X Axis Select:", filter_cols, index=0)
            with r_col2: y_axis = st.selectbox("Y Axis Select:", num_cols, index=1 if len(num_cols) > 1 else 0)
            with r_col3: color_dim = st.selectbox("Color Mapping Layer:", filter_cols)
            with r_col4: size_dim = st.selectbox("Marker Dimension Bubble:", ["None"] + num_cols)
            size_arg = None if size_dim == "None" else size_dim
            fig_rel = px.scatter(df_active, x=x_axis, y=y_axis, color=color_dim, size=size_arg, opacity=0.6, height=600)
            st.plotly_chart(fig_rel, use_container_width=True)

    # --- TAB 8: RAW SNAPSHOT ---
    with tab8:
        st.subheader("Data Export Matrix Hub")
        st.dataframe(df_active.describe().T.style.format("{:,.2f}"))
        csv = df_active.to_csv(index=False).encode('utf-8')
        st.download_button(label="Download Engineered File Slice (CSV)", data=csv, file_name='filtered_real_estate_data.csv', mime='text/csv')

# ==========================================
# MAIN EXECUTION ROUTING
# ==========================================
if df_base.empty:
    st.warning("⚠️ No records match your current Date and Sidebar Filters.")
else:
    if "Main View 1" in view_mode:
        st.title("Abu Dhabi Real Estate")
        st.markdown("Automated standard hygiene applied: 1% & 99% quantiles removed for precise property evaluation.")
        
        # 1. Define View 1 specific numerical columns
        num_cols_v1 = [c for c in ['Rate (AED per SQM)', 'Property Sold Area (SQM)', 'Property Sale Price (AED)'] if c in df_base.columns]
        
        # 2. Hardcoded Outlier Logic (Grouping by Asset Class & Property Type / District)
        df_v1_clean = df_base.copy()
        
        if all(c in df_v1_clean.columns for c in ['Asset Class', 'Property Type', 'Property Sold Area (SQM)']):
            area_low = df_v1_clean.groupby(['Asset Class', 'Property Type'])['Property Sold Area (SQM)'].transform(lambda x: x.quantile(0.01))
            area_high = df_v1_clean.groupby(['Asset Class', 'Property Type'])['Property Sold Area (SQM)'].transform(lambda x: x.quantile(0.99))
            df_v1_clean = df_v1_clean[(df_v1_clean['Property Sold Area (SQM)'] >= area_low) & (df_v1_clean['Property Sold Area (SQM)'] <= area_high)]
            
        if all(c in df_v1_clean.columns for c in ['Asset Class', 'District', 'Rate (AED per SQM)']):
            rate_low = df_v1_clean.groupby(['Asset Class', 'District'])['Rate (AED per SQM)'].transform(lambda x: x.quantile(0.01))
            rate_high = df_v1_clean.groupby(['Asset Class', 'District'])['Rate (AED per SQM)'].transform(lambda x: x.quantile(0.99))
            df_v1_clean = df_v1_clean[(df_v1_clean['Rate (AED per SQM)'] >= rate_low) & (df_v1_clean['Rate (AED per SQM)'] <= rate_high)]

        # 3. View 1 "Before/After" Interactive Toggle
        data_state = st.radio(
            "Toggle Data State for Dashboard:", 
            ["Cleaned Data (Auto-Outliers Removed)", "Raw Data (Before Outlier Cleaning)"], 
            horizontal=True
        )
        
        df_active = df_v1_clean if "Cleaned" in data_state else df_base
        
        # 4. Render Dashboard
        render_dashboard(df_active, df_base, num_cols_v1, all_filter_cols, is_view_1=True)

    elif "Main View 2" in view_mode:
        st.title("Dynamic filteration EDA")
        
        # 1. Show Dynamic Outliers Sidebar (Only in View 2)
        st.sidebar.subheader("🛠️ Dynamic Outlier Configuration")
        outlier_scope = st.sidebar.radio("Apply Outliers To:", ["None (Keep All Data)", "Filtered Data (Apply Now)"], index=0)
        outlier_grouping = st.sidebar.multiselect("Calculate Outliers Within Categories:", options=all_filter_cols, default=[])
        
        outlier_bounds = {}
        st.sidebar.markdown("**Numerical Percentiles**")
        for col in all_num_cols:
            with st.sidebar.expander(f"⚙️ {col} Limits"):
                if st.checkbox(f"Filter {col}", value=False, key=f"v2_check_{col}"):
                    lower, upper = st.slider(f"Percentile Range", 0.0, 100.0, (1.0, 99.0), 0.5, key=f"v2_slider_{col}")
                    outlier_bounds[col] = (lower / 100.0, upper / 100.0)

        # 2. Apply Dynamic Logic
        df_v2_clean = df_base.copy()
        if outlier_scope == "Filtered Data (Apply Now)":
            for col, (low, high) in outlier_bounds.items():
                if outlier_grouping:
                    q_low = df_v2_clean.groupby(outlier_grouping)[col].transform(lambda x: x.quantile(low))
                    q_high = df_v2_clean.groupby(outlier_grouping)[col].transform(lambda x: x.quantile(high))
                    df_v2_clean = df_v2_clean[(df_v2_clean[col] >= q_low) & (df_v2_clean[col] <= q_high)]
                else:
                    l_val, h_val = df_v2_clean[col].quantile(low), df_v2_clean[col].quantile(high)
                    df_v2_clean = df_v2_clean[(df_v2_clean[col] >= l_val) & (df_v2_clean[col] <= h_val)]
        
        # 3. Render Dashboard
        render_dashboard(df_v2_clean, df_base, all_num_cols, all_filter_cols, is_view_1=False)
