import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import shapiro, f_oneway, kruskal, ttest_ind, mannwhitneyu, normaltest
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scikit_posthocs as sp
import io
import csv
import json
from datetime import datetime

st.set_page_config(
    page_title="Benefit-med Data Explorer", 
    layout="wide",
    page_icon="📊"
)

st.markdown("""
    <style>
        .sticky-image {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            background-color: white;
            z-index: 100;
            border-bottom: 2px solid orange;
            padding: 80px 20;
            text-align: center;
        }

        .main-content {
            padding-top: 100px;  /* Ajuste cette valeur si l’image est plus grande ou plus petite */
        }

        /* Ton CSS existant (optionnel à conserver ici) */
        .main {padding-top: 2rem;}
        .st-b7 {background-color: #ff8c00;}
        .st-c0 {background-color: #ffffff;}
        .sidebar .sidebar-content {background-color: blue;}
        .reportview-container .main .block-container {padding-top: 2rem;}
        h1 {color: #2c3e50; border-bottom: 2px solid orange; padding-bottom: 10px;}
        h3 {color: #2980b9;}
        .st-bb {background-color: transparent;}
        .st-at {background-color: orange;}
        .st-ae {background-color: none;}
        .st-af {color: black;}
        .stButton>button {border-radius: 4px; padding: 0.5rem 1rem;}
        .stDownloadButton>button {background-color: #27ae60; color: white;}
        .st-eb {padding: 0.5rem;}
        .st-df {border-radius: 4px;}
        .stAlert {border-radius: 4px;}
        .stat-test {background-color: lightblue; padding: 15px; border-radius: 5px; margin-bottom: 15px;}
        .posthoc-table {margin-top: 20px;}
    </style>

    <div class="sticky-image">
        <img src="https://i.imgur.com/F0dqfM8.png" style="height:80px;" />
    </div>
""", unsafe_allow_html=True)

# Contenu principal décalé
st.markdown("<div class='main-content'>", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1 style='color: #2c3e50; text-decoration: none; margin-bottom: 10px;'>
            DATA EXPLORER
        </h1>
        <h3 style='color: #7f8c8d; font-weight: normal;'>
            Interactive statistical analysis and visualization
        </h3>
    </div>
""", unsafe_allow_html=True)


st.markdown("""
    <style>
        .main {padding-top: 2rem;}
        .st-b7 {background-color: #ff8c00;}
        .st-c0 {background-color: #ffffff;}
        .sidebar .sidebar-content {background-color: blue;}
        .reportview-container .main .block-container {padding-top: 2rem;}
        h1 {color: #2c3e50; border-bottom: 2px solid orange; padding-bottom: 10px;}
        h3 {color: #2980b9;}
        .st-bb {background-color: transparent;}
        .st-at {background-color: orange;}
        .st-ae {background-color: none;}
        .st-af {color: black;}
        .stButton>button {border-radius: 4px; padding: 0.5rem 1rem;}
        .stDownloadButton>button {background-color: #27ae60; color: white;}
        .st-eb {padding: 0.5rem;}
        .st-df {border-radius: 4px;}
        .stAlert {border-radius: 4px;}
        .stat-test {background-color: lightblue; padding: 15px; border-radius: 5px; margin-bottom: 15px;}
        .posthoc-table {margin-top: 20px;}
    </style>
""", unsafe_allow_html=True)

def detect_separator(file):
    """Detect the separator used in a CSV file."""
    sample = file.read(1024).decode('utf-8')
    file.seek(0)
    sniffer = csv.Sniffer()
    dialect = sniffer.sniff(sample)
    return dialect.delimiter

# Creation of a sample dataset
@st.cache_data
def generate_sample_data():
    """Generate a comprehensive metabolite dataset with realistic features."""
    np.random.seed(42)
    n = 300 
    genotypes = ['Wild Type', 'Mutant A', 'Mutant B', 'Transgenic']
    tissues = ['Leaf', 'Root', 'Stem', 'Flower']
    treatments = ['Control', 'Drought', 'Salt', 'Cold', 'Heat']
    time_points = ['0h', '6h', '12h', '24h', '48h', '72h']
    
    secondary_metabolites = [
        'Chlorogenic Acid', 'Rutin', 'Quercetin', 'Kaempferol', 
        'Caffeic Acid', 'Ferulic Acid', 'Sinapic Acid', 'Anthocyanins'
    ]
    
    df = pd.DataFrame({
        "Genotype": np.random.choice(genotypes, size=n, p=[0.4, 0.2, 0.2, 0.2]),
        "Tissue": np.random.choice(tissues, size=n, p=[0.4, 0.3, 0.2, 0.1]),
        "Treatment": np.random.choice(treatments, size=n, p=[0.3, 0.2, 0.2, 0.15, 0.15]),
        "Time_Point": np.random.choice(time_points, size=n, p=[0.2, 0.15, 0.15, 0.2, 0.2, 0.1]),
        "Growth_Condition": np.random.choice(['Field', 'Greenhouse', 'Growth Chamber'], size=n),
    })
    
    for met in secondary_metabolites:
        if met in ['Anthocyanins']:
            base = np.random.normal(5, 2, n)
            stress_effect = np.where(df['Treatment'].isin(['Cold', 'Drought']), np.random.normal(10, 3, n), 0)
            genotype_effect = np.where(df['Genotype'] == 'Mutant A', np.random.normal(-3, 1, n), 0)
            df[met] = np.clip(base + stress_effect + genotype_effect, 0.5, 30).round(2)
            
        elif met in ['Chlorogenic Acid', 'Rutin']:
            base = np.random.normal(8, 3, n)
            tissue_effect = np.where(df['Tissue'].isin(['Leaf', 'Flower']), np.random.normal(5, 2, n), 0)
            df[met] = np.clip(base + tissue_effect, 0.5, 25).round(2)
            
        else:
            df[met] = np.clip(np.random.lognormal(1.5, 0.5, n), 0.1, 15).round(2)
    
    for col in secondary_metabolites:
        mask = np.random.rand(n) < 0.05
        df.loc[mask, col] = np.nan
    
    return df

def check_normality(data, alpha=0.05):
    """Check if data is normally distributed using Shapiro-Wilk test."""
    if len(data) < 3:
        return False, "Not enough data points (n < 3)"
    
    stat, p = shapiro(data)
    if p > alpha:
        return True, f"Normally distributed (Shapiro-Wilk p={p:.2f})"
    else:
        return False, f"Not normally distributed (Shapiro-Wilk p={p:.2f})"

def check_variance_homogeneity(group_data, alpha=0.05):
    """Check if groups have equal variances using Levene's test."""
    from scipy.stats import levene
    try:
        stat, p = levene(*group_data)
        if p > alpha:
            return True, f"Equal variances (Levene's p={p:.2f})"
        else:
            return False, f"Unequal variances (Levene's p={p:.2f})"
    except:
        return False, "Could not perform variance test"

def perform_statistical_test(df, x_col, y_col):
    """Perform appropriate statistical test based on data characteristics."""
    groups = df[x_col].dropna().unique()
    group_data = [df[df[x_col] == group][y_col].dropna() 
                 for group in groups if len(df[df[x_col] == group]) >= 3]

    if len(group_data) < 2:
        return None, "Not enough groups with sufficient data (need at least 2 groups with ≥3 samples each)"

    
    normality_results = [check_normality(g) for g in group_data]
    all_normal = all([result[0] for result in normality_results])
    variance_result = check_variance_homogeneity(group_data)
    equal_var = variance_result[0]

    test_results = {
        "x_variable": x_col,
        "y_variable": y_col,
        "groups": groups.tolist(),
        "group_sizes": [len(g) for g in group_data],
        "normality": [result[1] for result in normality_results],
        "variance": variance_result[1],
        "all_normal": all_normal,
        "equal_var": equal_var
    }

    if len(groups) > 2:
        if all_normal and equal_var:
            stat, p = f_oneway(*group_data)
            test_used = "ANOVA"
            posthoc_method = "Tukey HSD"
        else:
            stat, p = kruskal(*group_data)
            test_used = "Kruskal-Wallis"
            posthoc_method = "Dunn's test"
    else:
        g1, g2 = group_data[:2]
        if all_normal and equal_var:
            stat, p = ttest_ind(g1, g2, equal_var=True)
            test_used = "Independent t-test"
            posthoc_method = None
        else:
            stat, p = mannwhitneyu(g1, g2)
            test_used = "Mann-Whitney U"
            posthoc_method = None

    if p < 0.1:
        p_str = "p < 0.1"
    else:
        p_str = f"p = {p:.2f}"
    
    interpretation = "Significant difference" if p < 0.1 else "No significant difference"
    
    test_results.update({
        "test": test_used,
        "statistic": stat,
        "p_value": p,
        "p_str": p_str,
        "interpretation": interpretation,
        "posthoc_method": posthoc_method
    })

    return test_results, None

def perform_posthoc_test(df, x_col, y_col, test_results):
    """Perform appropriate post-hoc test based on the main test."""
    if test_results["posthoc_method"] is None:
        return None
    
    if test_results["posthoc_method"] == "Tukey HSD":
        
        data = df[[x_col, y_col]].dropna()
        tukey = pairwise_tukeyhsd(endog=data[y_col], groups=data[x_col], alpha=0.05)
        
        tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
        tukey_df = tukey_df[tukey_df['reject']].sort_values('meandiff', ascending=False)
        
        return {
            "method": "Tukey HSD",
            "results": tukey_df,
            "summary": tukey.summary()
        }
    
    elif test_results["posthoc_method"] == "Dunn's test":
        
        data = df[[x_col, y_col]].dropna()
        dunn_results = sp.posthoc_dunn(data, val_col=y_col, group_col=x_col, p_adjust='holm')
        
        significant_pairs = []
        for i in range(len(dunn_results)):
            for j in range(i+1, len(dunn_results)):
                if dunn_results.iloc[i,j] < 0.05:
                    significant_pairs.append({
                        'group1': dunn_results.index[i],
                        'group2': dunn_results.columns[j],
                        'p_value': dunn_results.iloc[i,j]
                    })
        
        return {
            "method": "Dunn's test",
            "results": dunn_results,
            "significant_pairs": significant_pairs
        }
    
    return None

def plot_statistical_results(df, x_col, y_col, test_results):
    """Create visualization with statistical annotations."""
    fig = px.box(df, x=x_col, y=y_col, color=x_col, points="all",
                title=f"{y_col} by {x_col}<br>{test_results['test']}: {test_results['p_str']} ({test_results['interpretation']})")

    if "posthoc" in test_results and test_results["posthoc"] is not None:
        if test_results["posthoc"]["method"] == ["Tukey HSD", "Dunn's test"]:
             
            sig_pairs = test_results["posthoc"]["results"]
            if len(sig_pairs) > 0:
                annotations = []
                y_max = df[y_col].max()
                y_step = (df[y_col].max() - df[y_col].min()) * 0.05
                
                for i, (_, row) in enumerate(sig_pairs.iterrows()):
                    group1 = row['group1']
                    group2 = row['group2']
                    p_value = row['p-adj']
                    
                    annotations.append(dict(
                        xref='paper', yref='y',
                        x=0.05, y=y_max + (i+1)*y_step,
                        xanchor='left', yanchor='middle',
                        text=f"{group1} > {group2} (p={p_value:.2f})",
                        showarrow=False,
                        font=dict(size=12)
                    ))
                
                fig.update_layout(annotations=annotations)
    
    return fig

with st.sidebar:
    side_image_url = "https://i.imgur.com/hoAtwrV.jpeg" 
    st.sidebar.image(side_image_url, use_container_width = True)
    st.header("📂 Data Upload")
    uploaded_file = st.file_uploader("Upload File", type=["csv", "xlsx", "json"], 
                                   help="Upload your dataset in CSV, Excel, or JSON format",
                                   label_visibility="collapsed")
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                sep = detect_separator(uploaded_file)
                df = pd.read_csv(uploaded_file, sep=sep)
                st.session_state["separator"] = sep
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.json'):
                df = pd.read_json(uploaded_file)
            else:
                raise ValueError("Unsupported file format")
            
            st.session_state["uploaded_file"] = uploaded_file.name
            
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except (ValueError, TypeError):
                        continue
            
            st.session_state["df"] = df
            st.success(f"✅ {uploaded_file.name} loaded successfully")
            if uploaded_file.name.endswith('.csv'):
                st.info(f"Detected separator: `{sep}`")
                
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            df = generate_sample_data()
            st.session_state["df"] = df
            st.info("💡 Using sample data instead.")

    else:
        df = generate_sample_data()
        st.info("💡 Using a sample dataset. Upload your own data to analyze.")
    
    cat_vars = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_vars = df.select_dtypes(include=["number"]).columns.tolist()
    date_vars = df.select_dtypes(include=["datetime"]).columns.tolist()
    
    st.header("🔍 Data Filters")
    with st.expander("Categorical Filters"):
        for col in cat_vars:
            options = st.multiselect(
                f"Filter {col}",
                df[col].dropna().unique(),
                default=df[col].dropna().unique(),
                key=f"filter_{col}"
            )
            df = df[df[col].isin(options)]
    
    st.header("📊 Analysis Settings")
    analysis_type = st.radio(
        "Analysis Type",
        ["Descriptive", "Comparative", "Correlational"],
        index=1,
        help="Select the type of analysis to perform"
    )
    
    if analysis_type == "Comparative":
        x_var = st.selectbox(
            "Grouping Variable (Categorical)",
            options=cat_vars,
            help="Select categorical variable to group by"
        )
        
        y_var = st.selectbox(
            "Outcome Variable (Numerical)",
            options=num_vars,
            help="Select numerical variable to compare"
        )
        
        graph_type = st.selectbox(
            "Graph Type",
            ["Box Plot", "Violin Plot", "Strip Plot", "Bar Plot"],
            help="Select the type of visualization to display"
        )
        color_var = st.selectbox(
            "Color by (Optional)",
            options=["None"] + cat_vars,
            help="Color the graph by another categorical variable"
        )
        color_var = None if color_var == "None" else color_var
        
        show_posthoc = st.checkbox(
            "Show post-hoc tests",
            value=True,
            help="Show pairwise comparisons if overall test is significant"
        )
    
    elif analysis_type == "Correlational":
        x_var_corr = st.selectbox(
            "X Variable (Numerical)",
            options=num_vars,
            help="Select first numerical variable"
        )
        
        y_var_corr = st.selectbox(
            "Y Variable (Numerical)",
            options=[v for v in num_vars if v != x_var_corr],
            help="Select second numerical variable"
        )
        
        graph_type_corr = st.selectbox(
            "Graph Type",
            ["Scatter Plot", "Line Plot", "Hexbin Plot", "Density Contour"],
            help="Select the type of visualization to display"
        )
        
        color_var_corr = st.selectbox(
            "Color by (Optional)",
            options=["None"] + cat_vars,
            help="Color the graph by another categorical variable"
        )
        color_var_corr = None if color_var_corr == "None" else color_var_corr
        
        trendline = st.checkbox(
            "Show trendline",
            value=True,
            help="Add regression line to scatter plot"
        )
st.subheader("🔍 Data Preview")
with st.expander("View Data", expanded=True):
    st.dataframe(df.head(100), height=300)
    st.caption(f"Showing {len(df)} rows from original dataset")

st.subheader("📈 Statistical Analysis")

if analysis_type == "Comparative":
    st.markdown(f"### Comparative Analysis: {y_var} by {x_var}")
    
    test_results, error = perform_statistical_test(df, x_var, y_var)
    
    if error:
        st.error(error)
    else:
        
        st.markdown("""
        <div class="stat-test">
            <h4>Statistical Test Results</h4>
            <p><strong>Test used:</strong> {test}</p>
            <p><strong>Test statistic:</strong> {statistic:.2f}</p>
            <p><strong>P-value:</strong> {p_str}</p>
            <p><strong>Interpretation:</strong> {interpretation}</p>
            <p><strong>Normality:</strong> {normality}</p>
            <p><strong>Variance homogeneity:</strong> {variance}</p>
        </div>
        """.format(
            test=test_results["test"],
            statistic=test_results["statistic"],
            p_str=test_results["p_str"],
            interpretation=test_results["interpretation"],
            normality=", ".join(test_results["normality"]),
            variance=test_results["variance"]
        ), unsafe_allow_html=True)
        
        if show_posthoc and test_results["p_value"] < 0.05 and test_results["posthoc_method"] is not None:
            posthoc_results = perform_posthoc_test(df, x_var, y_var, test_results)
            test_results["posthoc"] = posthoc_results
            
            if posthoc_results is not None:
                st.markdown(f"#### Post-hoc Analysis ({posthoc_results['method']})")
                
                if posthoc_results["method"] == "Tukey HSD":
                    st.write("Significant pairwise comparisons:")
                    st.dataframe(posthoc_results["results"])
                    
                    with st.expander("Show full Tukey HSD results"):
                        st.text(posthoc_results["summary"])
                
                elif posthoc_results["method"] == "Dunn's test":
                    if len(posthoc_results["significant_pairs"]) > 0:
                        st.write("Significant pairwise comparisons (adjusted p-values):")
                        posthoc_df = pd.DataFrame(posthoc_results["significant_pairs"])
                        st.dataframe(posthoc_df.sort_values("p_value"))
                    else:
                        st.info("No significant pairwise comparisons found after adjustment")
                    
                    with st.expander("Show full Dunn's test matrix"):
                        st.dataframe(posthoc_results["results"].style.format("{:.3f}"))
        
        title = f"{y_var} by {x_var}<br>{test_results['test']}: {test_results['p_str']} ({test_results['interpretation']})"
        
        try:
            if graph_type == "Box Plot":
                fig = px.box(df, x=x_var, y=y_var, color=color_var, points="all",
                            title=title, hover_data=df.columns)
            elif graph_type == "Violin Plot":
                fig = px.violin(df, x=x_var, y=y_var, color=color_var, box=True,
                               title=title, hover_data=df.columns)
            elif graph_type == "Strip Plot":
                fig = px.strip(df, x=x_var, y=y_var, color=color_var,
                              title=title, hover_data=df.columns)
            elif graph_type == "Bar Plot":
                if color_var:
                    df_agg = df.groupby([x_var, color_var])[y_var].agg(['mean', 'std', 'count']).reset_index()
                    df_agg['se'] = df_agg['std'] / np.sqrt(df_agg['count']) 

                    fig = px.bar(df_agg, x=x_var, y='mean', color=color_var,
                                 error_y='se', 
                                 title=title,
                                 labels={'mean': y_var},
                                 hover_data={'mean':':.2f', 'se':':.2f', 'std':':.2f', 'count': True})
                else:
                    df_agg = df.groupby(x_var)[y_var].agg(['mean', 'std', 'count']).reset_index()
                    
                    df_agg['se'] = df_agg['std'] / np.sqrt(df_agg['count'])

                    fig = px.bar(df_agg, x=x_var, y='mean',
                            error_y='se', title=title,
                            labels={'mean': y_var},
                            hover_data={'mean': ':.2f', 'se': ':.2f', 'std': ':.2f','count': True}) 
                fig.update_traces(error_y_thickness=1.5, error_y_color='black')
            
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not create {graph_type}: {str(e)}")

elif analysis_type == "Correlational":
    st.markdown(f"### Correlational Analysis: {y_var_corr} vs {x_var_corr}")

    corr = df[[x_var_corr, y_var_corr]].corr().iloc[0,1]
    
    st.markdown(f"""
    <div class="stat-test">
        <h4>Correlation Analysis</h4>
        <p><strong>Pearson correlation coefficient (r):</strong> {corr:.3f}</p>
        <p><strong>Interpretation:</strong> {"Positive" if corr > 0 else "Negative"} correlation, 
        {"strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.3 else "weak"}</p>
    </div>
    """, unsafe_allow_html=True)
    
    
    title = f"{y_var_corr} vs {x_var_corr} (r = {corr:.2f})"
    
    try:
        if graph_type_corr == "Scatter Plot":
            fig = px.scatter(df, x=x_var_corr, y=y_var_corr, 
                           color=color_var_corr,
                           trendline="ols" if trendline else None,
                           title=title,
                           hover_data=df.columns)
            
        elif graph_type_corr == "Line Plot":
            fig = px.line(df, x=x_var_corr, y=y_var_corr, 
                         color=color_var_corr,
                         title=title,
                         hover_data=df.columns)
            
        elif graph_type_corr == "Hexbin Plot":
            fig = px.density_heatmap(df, x=x_var_corr, y=y_var_corr,
                                   title=title + " (Density Heatmap)",
                                   hover_data=df.columns)
            
        elif graph_type_corr == "Density Contour":
            fig = px.density_contour(df, x=x_var_corr, y=y_var_corr,
                                   color=color_var_corr,
                                   title=title,
                                   hover_data=df.columns)
            fig.update_traces(contours_coloring="fill", contours_showlabels=True)
        
        if trendline and graph_type_corr == "Scatter Plot":
            
            results = px.get_trendline_results(fig)
            params = results.iloc[0]["px_fit_results"].params
            equation = f"{y_var_corr} = {params[0]:.2f} + {params[1]:.2f}*{x_var_corr}"
            fig.add_annotation(
                x=0.05, y=0.95,
                xref="paper", yref="paper",
                text=equation,
                showarrow=False,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        
        if trendline and graph_type_corr == "Scatter Plot":
            st.markdown("#### Residual Analysis")
            
            X = df[x_var_corr].values.reshape(-1, 1)
            Y = df[y_var_corr].values
            model = LinearRegression().fit(X, Y)
            residuals = Y - model.predict(X)
            
            fig_res = px.scatter(
                x=model.predict(X),
                y=residuals,
                color=df[color_var_corr] if color_var_corr else None,
                labels={'x': 'Predicted Values', 'y': 'Residuals'},
                title="Residual Plot",
                trendline="lowess"
            )
            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_res, use_container_width=True)
            
            normal, norm_msg = check_normality(residuals)
            st.markdown(f"**Residuals normality check:** {norm_msg}")
            
    except Exception as e:
        st.error(f"Could not create {graph_type_corr}: {str(e)}")

with st.expander("💾 Export Data", expanded=False):
    st.write("### Export Options")
    export_format = st.radio(
        "Export Format",
        ["CSV", "Excel", "JSON"],
        horizontal=True
    )
    export_filename = st.text_input(
        "Filename",
        value=f"data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    if st.button("Export Data"):
        try:
            buffer = io.BytesIO()
            
            if export_format == "CSV":
                df.to_csv(buffer, index=False)
                buffer.seek(0)
                st.download_button(
                    "Download CSV",
                    data=buffer,
                    file_name=f"{export_filename}.csv",
                    mime="text/csv"
                )
            elif export_format == "Excel":
                with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                    df.to_excel(writer, index=False, sheet_name='Data')
                    writer.close()
                st.download_button(
                    "Download Excel",
                    data=buffer,
                    file_name=f"{export_filename}.xlsx",
                    mime="application/vnd.ms-excel"
                )
            elif export_format == "JSON":
                json_data = df.to_json(orient='records', indent=2)
                st.download_button(
                    "Download JSON",
                    data=json_data,
                    file_name=f"{export_filename}.json",
                    mime="application/json"
                )
            
            st.success("Data exported successfully!")
        except Exception as e:
            st.error(f"Error exporting data: {str(e)}")

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: black; font-size: 0.9em;'>
        <p>For educational and research purposes</p>
        <p>© 2025 From the BENEFIT-MED Project supported by the PRIMA funding Program</p>
        <p>Contact: 
            <a href='mailto:loic.rajjou@inrae.fr'>loic.rajjou@inrae.fr</a>|
            <a href='mailto:alma.balestrazzi@unipv.it'>alma.balestrazzi@unipv.it</a>
            </p>
    </div>
""", unsafe_allow_html=True)
