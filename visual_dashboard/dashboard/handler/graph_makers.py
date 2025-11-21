"""
Graph generation functions for the safety events dashboard.
All functions are designed to accept filtered data to allow user customization.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from io import BytesIO
import base64
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# Helper Functions
# ============================================================================


def prepare_categorical_data(df, column, dropna=True, include_missing_as_category=False):
    """
    Prepare categorical data for visualization.
    
    Args:
        df: DataFrame
        column: Name of the categorical column
        dropna: Whether to drop NaN values (if False and include_missing_as_category=False, 
                missing values will still be excluded from counts)
        include_missing_as_category: If True, replace NaN with "Missing" category
    
    Returns:
        Series with value counts
    """
    if column not in df.columns:
        return pd.Series(dtype=int)
    
    data = df[column].copy()
    
    # Handle missing values
    if include_missing_as_category:
        # Replace NaN/None with "Missing" category
        data = data.fillna('Missing')
        return data.value_counts()
    elif dropna:
        # Drop missing values (current default behavior)
        data = data.dropna()
        return data.value_counts()
    else:
        # Keep missing values but value_counts will still exclude them by default
        # Use dropna=False to include them
        return data.value_counts(dropna=False)


def get_data_quality_info(df, column):
    """
    Get data quality information for a column.
    
    Args:
        df: DataFrame
        column: Name of the column
    
    Returns:
        Dictionary with:
            - total_rows: Total number of rows in dataframe
            - missing_count: Number of missing values in the column
            - valid_count: Number of non-missing values
            - missing_percentage: Percentage of missing values
    """
    if column not in df.columns:
        return {
            'total_rows': len(df),
            'missing_count': len(df),
            'valid_count': 0,
            'missing_percentage': 100.0
        }
    
    total_rows = len(df)
    missing_count = df[column].isna().sum()
    valid_count = total_rows - missing_count
    missing_percentage = (missing_count / total_rows * 100) if total_rows > 0 else 0
    
    return {
        'total_rows': total_rows,
        'missing_count': missing_count,
        'valid_count': valid_count,
        'missing_percentage': missing_percentage
    }


# ============================================================================
# 1. Statistical Summary Function
# ============================================================================

def generate_statistical_summary(df):
    """
    Generate statistical summary including total counts, averages, percentages, etc.
    This is used for the "Big Numbers" display.
    
    Args:
        df: DataFrame with incident data
    
    Returns:
        Dictionary with statistical metrics
    """
    summary = {}
    
    # Total number of incidents
    summary['total_incidents'] = len(df)
    
    # High-risk incidents: LOSS_POTENTIAL_SEVERITY >= 4
    if 'LOSS_POTENTIAL_SEVERITY' in df.columns:
        severity_values = pd.to_numeric(df['LOSS_POTENTIAL_SEVERITY'], errors='coerce')
        high_risk_incidents = len(df[severity_values >= 4])
        summary['high_risk_count'] = high_risk_incidents
        summary['high_risk_percentage'] = (high_risk_incidents / len(df) * 100) if len(df) > 0 else 0
    else:
        summary['high_risk_count'] = 0
        summary['high_risk_percentage'] = 0
    
    # Average severity score (using LOSS_POTENTIAL_SEVERITY)
    if 'LOSS_POTENTIAL_SEVERITY' in df.columns:
        severity_values = pd.to_numeric(df['LOSS_POTENTIAL_SEVERITY'], errors='coerce')
        summary['avg_severity'] = severity_values.mean()
        summary['min_severity'] = severity_values.min()
        summary['max_severity'] = severity_values.max()
        summary['median_severity'] = severity_values.median()
    else:
        summary['avg_severity'] = None
        summary['min_severity'] = None
        summary['max_severity'] = None
        summary['median_severity'] = None
    
    # Average likelihood score
    if 'LIKELIHOOD_VALUE' in df.columns:
        likelihood_values = pd.to_numeric(df['LIKELIHOOD_VALUE'], errors='coerce')
        summary['avg_likelihood'] = likelihood_values.mean()
        summary['min_likelihood'] = likelihood_values.min()
        summary['max_likelihood'] = likelihood_values.max()
        summary['median_likelihood'] = likelihood_values.median()
    else:
        summary['avg_likelihood'] = None
        summary['min_likelihood'] = None
        summary['max_likelihood'] = None
        summary['median_likelihood'] = None
    
    # Number of open corrective actions (STATUS == 'Open')
    if 'STATUS' in df.columns:
        open_actions = len(df[df['STATUS'] == 'Open'])
        summary['open_actions'] = open_actions
        summary['open_actions_percentage'] = (open_actions / len(df) * 100) if len(df) > 0 else 0
    else:
        summary['open_actions'] = 0
        summary['open_actions_percentage'] = 0
    
    return summary

# =============================================================================
# 2. Bar Chart - All Categories
# =============================================================================

def generate_bar_chart_all_categories(df, column, title=None, rotation=45, top_n=None, return_metadata=False):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")

    counts = df[column].value_counts().dropna()
    if top_n:
        counts = counts.head(top_n)

    fig = px.bar(
        x=counts.index,
        y=counts.values,
        text=counts.values,
        title=title or f"Bar Chart: {column}",
        labels={'x': column, 'y': 'Count'}
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(template='plotly_white', xaxis_tickangle=rotation, height=600)

    if return_metadata:
        return {'figure': fig, 'data_quality': get_data_quality_info(df, column)}
    return fig

# =============================================================================
# 3. Bar Chart - Top K
# =============================================================================

def generate_bar_chart_top_k(df, column, k=10, title=None, horizontal=False, return_metadata=False):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")

    counts = df[column].value_counts().head(k)
    fig = px.bar(
        x=counts.index if not horizontal else counts.values[::-1],
        y=counts.values if not horizontal else counts.index[::-1],
        orientation='h' if horizontal else 'v',
        title=title or f"Top {k}: {column}",
        text=counts.values,
        labels={'x': column if not horizontal else 'Count', 'y': 'Count' if not horizontal else column}
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(template='plotly_white', height=600)

    if return_metadata:
        return {'figure': fig, 'data_quality': get_data_quality_info(df, column)}
    return fig

# =============================================================================
# 4. Pie Chart
# =============================================================================

def generate_pie_chart(df, column, title=None, top_n=None, return_metadata=False):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")

    counts = df[column].value_counts()
    if top_n and top_n < len(counts):
        others = counts.iloc[top_n:].sum()
        counts = counts.head(top_n)
        counts['Others'] = others

    fig = px.pie(
        names=counts.index,
        values=counts.values,
        title=title or f"Pie Chart: {column}",
        hole=0.3
    )
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(template='plotly_white', height=600)

    if return_metadata:
        return {'figure': fig, 'data_quality': get_data_quality_info(df, column)}
    return fig

# =============================================================================
# 5. Heatmap
# =============================================================================

def generate_heatmap(df, x_column, y_column, title=None, return_metadata=False):
    if x_column not in df.columns or y_column not in df.columns:
        raise ValueError("One or both columns not found in dataframe")

    crosstab = pd.crosstab(df[y_column], df[x_column])
    fig = px.imshow(
        crosstab,
        text_auto=True,
        color_continuous_scale='YlOrRd',
        title=title or f"Heatmap: {x_column} vs {y_column}"
    )
    fig.update_layout(template='plotly_white', height=700)

    if return_metadata:
        return {'figure': fig, 'data_quality': get_data_quality_info(df, x_column)}
    return fig

# =============================================================================
# 6. Word Cloud
# =============================================================================

def generate_wordcloud(df, text_column, title=None, max_words=100, return_metadata=False):
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in dataframe")

    text_data = df[text_column].dropna().astype(str)
    combined_text = ' '.join(text_data.tolist())
    wc = WordCloud(width=800, height=400, max_words=max_words, background_color='white').generate(combined_text)

    # Convert to base64 image
    buffer = BytesIO()
    wc.to_image().save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode()
    img_uri = f"data:image/png;base64,{encoded}"

    # Display as Plotly image
    fig = go.Figure()
    fig.add_layout_image(dict(source=img_uri, xref="paper", yref="paper", x=0, y=1, sizex=1, sizey=1, xanchor="left", yanchor="top"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(title=title or f"Word Cloud: {text_column}", template='plotly_white', height=500)

    if return_metadata:
        return {'figure': fig, 'data_quality': get_data_quality_info(df, text_column)}
    return fig

# =============================================================================
# 7. Temporal Distribution
# =============================================================================

def generate_temporal_distribution(df, time_column='DATE_OF_INCIDENT', groupby='year', title=None, return_metadata=False):
    if time_column not in df.columns:
        raise ValueError(f"Column '{time_column}' not found in dataframe")

    df_temp = df.copy()
    
    # Check if column is already numeric (like YEAR_OF_INCIDENT)
    if pd.api.types.is_numeric_dtype(df_temp[time_column]):
        # Already numeric, use directly
        df_temp = df_temp.dropna(subset=[time_column])
        if groupby == 'year':
            df_temp['GROUP'] = df_temp[time_column].astype(int)
        else:
            # For numeric columns, only year grouping makes sense
            df_temp['GROUP'] = df_temp[time_column].astype(int)
    else:
        # Try to parse as datetime
        df_temp[time_column] = pd.to_datetime(df_temp[time_column], errors='coerce')
        df_temp = df_temp.dropna(subset=[time_column])

        if groupby == 'year':
            df_temp['GROUP'] = df_temp[time_column].dt.year
        elif groupby == 'month':
            df_temp['GROUP'] = df_temp[time_column].dt.to_period('M').astype(str)
        else:
            df_temp['GROUP'] = df_temp[time_column].dt.date

    counts = df_temp['GROUP'].value_counts().sort_index()

    fig = px.bar(
        x=counts.index,
        y=counts.values,
        text=counts.values,
        title=title or f"Incident Frequency by {groupby.capitalize()}",
        labels={'x': groupby.capitalize(), 'y': 'Number of Incidents'}
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(template='plotly_white', height=600)

    if return_metadata:
        return {'figure': fig, 'data_quality': get_data_quality_info(df, time_column)}
    return fig

# =============================================================================
# 8. Stacked Bar Chart
# =============================================================================

def generate_stacked_bar_chart(df, category_column, stack_column, title=None, return_metadata=False):
    if category_column not in df.columns or stack_column not in df.columns:
        raise ValueError(f"One or both columns not found in dataframe")

    crosstab = pd.crosstab(df[category_column], df[stack_column])
    melted = crosstab.reset_index().melt(id_vars=category_column, var_name=stack_column, value_name='Count')

    # Define color mapping for risk colors (matching impact type pie chart colors)
    # Environment (#fb8072) -> Red, Injury (#8dd3c7) -> Green, Injury/Illness (#ffffb3) -> Yellow
    color_discrete_map = None
    if stack_column.upper() == 'RISK_COLOR':
        color_discrete_map = {
            'Red': '#FF0000',
            'Yellow': '#FFFF33',
            'Green': '#4CBB17',
        }
        # Reorder categories for proper stacking: Green (bottom), Yellow, Red (top)
        # In Plotly, the order in color_discrete_map and category order affects stacking
        # We'll use category_order to ensure proper order
        unique_stack_values = melted[stack_column].unique()
        severity_order = ['Green', 'GREEN', 'green', 'Yellow', 'YELLOW', 'yellow', 'Red', 'RED', 'red']
        ordered_stack_values = [v for v in severity_order if v in unique_stack_values]
        ordered_stack_values.extend([v for v in unique_stack_values if v not in ordered_stack_values])
        
        fig = px.bar(
            melted,
            x=category_column,
            y='Count',
            color=stack_column,
            title=title or f"Stacked Bar Chart: {stack_column} by {category_column}",
            barmode='stack',
            color_discrete_map=color_discrete_map,
            category_orders={stack_column: ordered_stack_values}
        )
    else:
        fig = px.bar(
            melted,
            x=category_column,
            y='Count',
            color=stack_column,
            title=title or f"Stacked Bar Chart: {stack_column} by {category_column}",
            barmode='stack'
        )
    
    fig.update_layout(template='plotly_white', height=600)

    if return_metadata:
        return {'figure': fig, 'data_quality': get_data_quality_info(df, category_column)}
    return fig

def generate_event_cluster_plot(df, x_col='tsne_x', y_col='tsne_y',
                                title="Event Cluster Visualization",
                                sample_ratio=0.1, color_column=None,
                                return_metadata=False):
    """
    Generate event cluster scatter plot using tsne_x and tsne_y.

    Args:
        df: filtered dataframe
        x_col, y_col: columns for coordinates
        sample_ratio: percent of events to visualize (0~1)
        color_column: optional column used to color the nodes

    Returns:
        Plotly figure (or dict with metadata)
    """

    # 1. 必须存在 tsne_x / tsne_y
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError("tsne_x / tsne_y columns not found in dataframe.")

    # 2. 清理：跳过坐标为空的 event
    df = df.dropna(subset=[x_col, y_col])
    if len(df) == 0:
        raise ValueError("No events with valid tsne_x/tsne_y values.")

    # 3. 控制采样规模
    if sample_ratio < 1:
        n = int(len(df) * sample_ratio)
        df = df.sample(n=n, random_state=42)

    # 4. 画图
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=df[color_column] if color_column else None,
        title=title,
        opacity=0.7
    )
    fig.update_layout(template='plotly_white', height=700)

    if return_metadata:
        return {
            "figure": fig,
            "data_quality": get_data_quality_info(df, x_col)
        }

    return fig
