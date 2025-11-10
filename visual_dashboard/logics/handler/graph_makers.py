"""
Graph generation functions for the safety events dashboard.
All functions are designed to accept filtered data to allow user customization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


# ============================================================================
# Helper Functions
# ============================================================================

def filter_data_by_date_range(df, date_column, start_date=None, end_date=None):
    """
    Filter dataframe by date range.
    
    Args:
        df: DataFrame to filter
        date_column: Name of the date column
        start_date: Start date (inclusive), format: 'YYYY-MM-DD' or None
        end_date: End date (inclusive), format: 'YYYY-MM-DD' or None
    
    Returns:
        Filtered DataFrame
    """
    df_filtered = df.copy()
    
    if date_column not in df_filtered.columns:
        return df_filtered
    
    # Convert date column to datetime
    df_filtered[date_column] = pd.to_datetime(df_filtered[date_column], errors='coerce')
    
    if start_date:
        df_filtered = df_filtered[df_filtered[date_column] >= pd.to_datetime(start_date)]
    
    if end_date:
        df_filtered = df_filtered[df_filtered[date_column] <= pd.to_datetime(end_date)]
    
    return df_filtered


def filter_data_by_column(df, column, values):
    """
    Filter dataframe by column values.
    
    Args:
        df: DataFrame to filter
        column: Name of the column to filter by
        values: List of values to keep, or None to keep all
    
    Returns:
        Filtered DataFrame
    """
    if values is None or len(values) == 0:
        return df.copy()
    
    if column not in df.columns:
        return df.copy()
    
    return df[df[column].isin(values)].copy()


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


def format_number(value, decimals=1):
    """
    Format number for display.
    
    Args:
        value: Number to format
        decimals: Number of decimal places
    
    Returns:
        Formatted string
    """
    if pd.isna(value):
        return "N/A"
    
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    
    return f"{value:,.{decimals}f}"


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


# ============================================================================
# 2. Bar Chart - All Categories
# ============================================================================

def generate_bar_chart_all_categories(df, column, title=None, xlabel=None, ylabel="Count", 
                                       figsize=(12, 6), rotation=45, top_n=None, return_metadata=False):
    """
    Generate a bar chart showing all categories in a column.
    
    Args:
        df: DataFrame
        column: Name of the column to visualize
        title: Chart title (default: column name)
        xlabel: X-axis label (default: column name)
        ylabel: Y-axis label
        figsize: Figure size tuple
        rotation: Rotation angle for x-axis labels
        top_n: If specified, only show top N categories
        return_metadata: If True, return dict with 'figure' and 'data_quality' keys
    
    Returns:
        matplotlib Figure object, or dict with 'figure' and 'data_quality' if return_metadata=True
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    # Get data quality info
    data_quality = get_data_quality_info(df, column)
    
    # Prepare data
    value_counts = prepare_categorical_data(df, column)
    
    if len(value_counts) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title or f"Bar Chart: {column}")
        if return_metadata:
            return {'figure': fig, 'data_quality': data_quality}
        return fig
    
    # Limit to top N if specified
    if top_n and top_n < len(value_counts):
        value_counts = value_counts.head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot
    bars = ax.bar(range(len(value_counts)), value_counts.values, color='steelblue', alpha=0.7)
    
    # Customize
    ax.set_xticks(range(len(value_counts)))
    ax.set_xticklabels(value_counts.index, rotation=rotation, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel or column)
    
    # Add data quality info to title
    chart_title = title or f"Bar Chart: {column}"
    if data_quality['missing_count'] > 0:
        chart_title += f"\n(Valid: {data_quality['valid_count']:,}, Missing: {data_quality['missing_count']:,})"
    ax.set_title(chart_title)
    
    # Add value labels on bars
    for i, (idx, val) in enumerate(value_counts.items()):
        ax.text(i, val, f'{int(val):,}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if return_metadata:
        return {'figure': fig, 'data_quality': data_quality}
    return fig


# ============================================================================
# 3. Bar Chart - Top K
# ============================================================================

def generate_bar_chart_top_k(df, column, k=10, title=None, xlabel=None, ylabel="Count",
                             figsize=(12, 6), rotation=45, horizontal=False, return_metadata=False):
    """
    Generate a bar chart showing top K categories.
    
    Args:
        df: DataFrame
        column: Name of the column to visualize
        k: Number of top categories to show
        title: Chart title (default: "Top K {column}")
        xlabel: X-axis label (default: column name)
        ylabel: Y-axis label
        figsize: Figure size tuple
        rotation: Rotation angle for x-axis labels
        horizontal: If True, create horizontal bar chart
        return_metadata: If True, return dict with 'figure' and 'data_quality' keys
    
    Returns:
        matplotlib Figure object, or dict with 'figure' and 'data_quality' if return_metadata=True
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    # Get data quality info
    data_quality = get_data_quality_info(df, column)
    
    # Prepare data
    value_counts = prepare_categorical_data(df, column)
    
    if len(value_counts) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title or f"Top {k}: {column}")
        if return_metadata:
            return {'figure': fig, 'data_quality': data_quality}
        return fig
    
    # Get top K
    top_k = value_counts.head(k)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot
    if horizontal:
        bars = ax.barh(range(len(top_k)), top_k.values, color='steelblue', alpha=0.7)
        ax.set_yticks(range(len(top_k)))
        ax.set_yticklabels(top_k.index)
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel or column)
        
        # Add value labels
        for i, (idx, val) in enumerate(top_k.items()):
            ax.text(val, i, f' {int(val):,}', va='center', fontsize=8)
    else:
        bars = ax.bar(range(len(top_k)), top_k.values, color='steelblue', alpha=0.7)
        ax.set_xticks(range(len(top_k)))
        ax.set_xticklabels(top_k.index, rotation=rotation, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel or column)
        
        # Add value labels
        for i, (idx, val) in enumerate(top_k.items()):
            ax.text(i, val, f'{int(val):,}', ha='center', va='bottom', fontsize=8)
    
    # Add data quality info to title
    chart_title = title or f"Top {k}: {column}"
    if data_quality['missing_count'] > 0:
        chart_title += f"\n(Valid: {data_quality['valid_count']:,}, Missing: {data_quality['missing_count']:,})"
    ax.set_title(chart_title)
    plt.tight_layout()
    
    if return_metadata:
        return {'figure': fig, 'data_quality': data_quality}
    return fig


# ============================================================================
# 4. Pie Chart
# ============================================================================

def generate_pie_chart(df, column, title=None, figsize=(10, 8), top_n=None, 
                       autopct='%1.1f%%', startangle=90, return_metadata=False):
    """
    Generate a pie chart for categorical data.
    
    Args:
        df: DataFrame
        column: Name of the column to visualize
        title: Chart title (default: column name)
        figsize: Figure size tuple
        top_n: If specified, only show top N categories, group others as "Others"
        autopct: Format string for percentage labels
        startangle: Starting angle for the pie chart
        return_metadata: If True, return dict with 'figure' and 'data_quality' keys
    
    Returns:
        matplotlib Figure object, or dict with 'figure' and 'data_quality' if return_metadata=True
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    # Get data quality info
    data_quality = get_data_quality_info(df, column)
    
    # Prepare data
    value_counts = prepare_categorical_data(df, column)
    
    if len(value_counts) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title or f"Pie Chart: {column}")
        if return_metadata:
            return {'figure': fig, 'data_quality': data_quality}
        return fig
    
    # Limit to top N if specified
    if top_n and top_n < len(value_counts):
        top_categories = value_counts.head(top_n)
        others_count = value_counts.iloc[top_n:].sum()
        if others_count > 0:
            top_categories['Others'] = others_count
        value_counts = top_categories
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot with adjusted label distances to prevent overlap
    colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
    wedges, texts, autotexts = ax.pie(value_counts.values, labels=value_counts.index,
                                       autopct=autopct, startangle=startangle, colors=colors,
                                       pctdistance=0.8, labeldistance=1.1)
    
    # Customize text - make smaller to prevent overlap
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(9)  # Smaller font size
        autotext.set_fontweight('bold')
    
    # Adjust label text size and position to prevent overlap
    for text in texts:
        text.set_fontsize(8)  # Smaller label font size
        # Use rotation and offset for better spacing
        if len(value_counts) > 5:
            # For many categories, rotate labels slightly
            text.set_rotation(0)
    
    # Add data quality info to title
    chart_title = title or f"Pie Chart: {column}"
    if data_quality['missing_count'] > 0:
        chart_title += f"\n(Valid: {data_quality['valid_count']:,}, Missing: {data_quality['missing_count']:,})"
    ax.set_title(chart_title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if return_metadata:
        return {'figure': fig, 'data_quality': data_quality}
    return fig


# ============================================================================
# 5. Heatmap
# ============================================================================

def generate_heatmap(df, x_column, y_column, title=None, figsize=(12, 8), 
                     cmap='YlOrRd', annot=True, fmt='d', cbar_label="Count", return_metadata=False):
    """
    Generate a heatmap showing the relationship between two categorical variables.
    
    Args:
        df: DataFrame
        x_column: Name of the column for x-axis
        y_column: Name of the column for y-axis
        title: Chart title (default: "{x_column} vs {y_column}")
        figsize: Figure size tuple
        cmap: Colormap name
        annot: Whether to annotate cells with values
        fmt: Format string for annotations
        cbar_label: Label for colorbar
        return_metadata: If True, return dict with 'figure' and 'data_quality' keys
    
    Returns:
        matplotlib Figure object, or dict with 'figure' and 'data_quality' if return_metadata=True
    """
    if x_column not in df.columns or y_column not in df.columns:
        raise ValueError(f"One or both columns not found in dataframe")
    
    # Get data quality info for both columns
    x_quality = get_data_quality_info(df, x_column)
    y_quality = get_data_quality_info(df, y_column)
    # Combined quality info (rows with both columns valid)
    valid_both = df[[x_column, y_column]].dropna()
    combined_quality = {
        'total_rows': len(df),
        'valid_count': len(valid_both),
        'missing_count': len(df) - len(valid_both),
        'missing_percentage': ((len(df) - len(valid_both)) / len(df) * 100) if len(df) > 0 else 0,
        'x_column_quality': x_quality,
        'y_column_quality': y_quality
    }
    
    # Create crosstab (automatically drops rows with missing values in either column)
    crosstab = pd.crosstab(df[y_column], df[x_column])
    
    if crosstab.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title or f"Heatmap: {x_column} vs {y_column}")
        if return_metadata:
            return {'figure': fig, 'data_quality': combined_quality}
        return fig
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(crosstab, annot=annot, fmt=fmt, cmap=cmap, ax=ax,
                cbar_kws={'label': cbar_label}, linewidths=0.5, linecolor='gray')
    
    # Add data quality info to title
    chart_title = title or f"Heatmap: {x_column} vs {y_column}"
    if combined_quality['missing_count'] > 0:
        chart_title += f"\n(Valid: {combined_quality['valid_count']:,}, Missing: {combined_quality['missing_count']:,})"
    ax.set_title(chart_title, fontsize=12, fontweight='bold')
    ax.set_xlabel(x_column, fontsize=10)
    ax.set_ylabel(y_column, fontsize=10)
    
    plt.tight_layout()
    
    if return_metadata:
        return {'figure': fig, 'data_quality': combined_quality}
    return fig


# ============================================================================
# 6. Word Cloud
# ============================================================================

def generate_wordcloud(df, text_column, title=None, figsize=(12, 8), max_words=100,
                       width=800, height=400, background_color='white', 
                       colormap='viridis', stopwords=None, return_metadata=False):
    """
    Generate a word cloud from text data.
    
    Args:
        df: DataFrame
        text_column: Name of the column containing text
        title: Chart title (default: "Word Cloud: {text_column}")
        figsize: Figure size tuple
        max_words: Maximum number of words to display
        width: Width of the word cloud
        height: Height of the word cloud
        background_color: Background color
        colormap: Colormap for word colors
        stopwords: Set of stopwords to exclude (or None to use default)
        return_metadata: If True, return dict with 'figure' and 'data_quality' keys
    
    Returns:
        matplotlib Figure object, or dict with 'figure' and 'data_quality' if return_metadata=True
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in dataframe")
    
    # Get data quality info
    data_quality = get_data_quality_info(df, text_column)
    
    # Combine all text (dropna removes missing values)
    text_data = df[text_column].dropna().astype(str)
    
    # Update data quality with actual valid count (after filtering empty strings)
    text_data = text_data[text_data.str.strip() != '']
    valid_count = len(text_data)
    data_quality['valid_count'] = valid_count
    data_quality['missing_count'] = len(df) - valid_count
    data_quality['missing_percentage'] = (data_quality['missing_count'] / len(df) * 100) if len(df) > 0 else 0
    
    if len(text_data) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No text data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title or f"Word Cloud: {text_column}")
        if return_metadata:
            return {'figure': fig, 'data_quality': data_quality}
        return fig
    
    # Combine all text
    combined_text = ' '.join(text_data.tolist())
    
    # Default stopwords
    if stopwords is None:
        stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
                        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                        'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
                        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'])
    
    # Generate word cloud
    wordcloud = WordCloud(width=width, height=height, max_words=max_words,
                         background_color=background_color, colormap=colormap,
                         stopwords=stopwords, relative_scaling=0.5,
                         random_state=42).generate(combined_text)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    # Add data quality info to title
    chart_title = title or f"Word Cloud: {text_column}"
    if data_quality['missing_count'] > 0:
        chart_title += f"\n(Valid: {data_quality['valid_count']:,}, Missing: {data_quality['missing_count']:,})"
    ax.set_title(chart_title, fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if return_metadata:
        return {'figure': fig, 'data_quality': data_quality}
    return fig


# ============================================================================
# Specialized Functions for Dashboard Requirements
# ============================================================================

def generate_temporal_distribution(df, time_column='DATE_OF_INCIDENT', 
                                  groupby='year', title=None, figsize=(12, 6), return_metadata=False):
    """
    Generate temporal distribution bar chart (by year/month/date).
    
    Args:
        df: DataFrame
        time_column: Name of the date column
        groupby: 'year', 'month', or 'date'
        title: Chart title
        figsize: Figure size tuple
        return_metadata: If True, return dict with 'figure' and 'data_quality' keys
    
    Returns:
        matplotlib Figure object, or dict with 'figure' and 'data_quality' if return_metadata=True
    """
    if time_column not in df.columns:
        raise ValueError(f"Column '{time_column}' not found in dataframe")
    
    # Get data quality info
    data_quality = get_data_quality_info(df, time_column)
    
    # Convert to datetime
    df_temp = df.copy()
    df_temp[time_column] = pd.to_datetime(df_temp[time_column], errors='coerce')
    df_temp = df_temp.dropna(subset=[time_column])
    
    # Update data quality with actual valid count after datetime conversion
    valid_after_conversion = len(df_temp)
    data_quality['valid_count'] = valid_after_conversion
    data_quality['missing_count'] = len(df) - valid_after_conversion
    data_quality['missing_percentage'] = (data_quality['missing_count'] / len(df) * 100) if len(df) > 0 else 0
    
    if len(df_temp) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No date data available', ha='center', va='center', transform=ax.transAxes)
        if return_metadata:
            return {'figure': fig, 'data_quality': data_quality}
        return fig
    
    # Group by time period
    if groupby == 'year':
        if 'YEAR_OF_INCIDENT' in df.columns:
            time_data = df['YEAR_OF_INCIDENT'].dropna()
            xlabel = 'Year'
        else:
            time_data = df_temp[time_column].dt.year
            xlabel = 'Year'
    elif groupby == 'month':
        if 'MONTH_OF_INCIDENT' in df.columns:
            time_data = df['MONTH_OF_INCIDENT'].dropna()
            xlabel = 'Month'
        else:
            time_data = df_temp[time_column].dt.month
            xlabel = 'Month'
    else:  # date
        time_data = df_temp[time_column].dt.date
        xlabel = 'Date'
    
    # Count occurrences
    counts = time_data.value_counts().sort_index()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(counts)), counts.values, color='steelblue', alpha=0.7)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=45, ha='right')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Number of Incidents')
    
    # Add data quality info to title
    chart_title = title or f"Incident Frequency by {xlabel}"
    if data_quality['missing_count'] > 0:
        chart_title += f"\n(Valid: {data_quality['valid_count']:,}, Missing: {data_quality['missing_count']:,})"
    ax.set_title(chart_title)
    
    # Add value labels
    for i, val in enumerate(counts.values):
        ax.text(i, val, f'{int(val):,}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if return_metadata:
        return {'figure': fig, 'data_quality': data_quality}
    return fig


def generate_stacked_bar_chart(df, category_column, stack_column, title=None,
                               figsize=(12, 6), rotation=45, return_metadata=False):
    """
    Generate a stacked bar chart showing distribution of stack_column within categories.
    
    Args:
        df: DataFrame
        category_column: Column for x-axis categories
        stack_column: Column to stack (e.g., RISK_COLOR, SEVERITY)
        title: Chart title
        figsize: Figure size tuple
        rotation: Rotation angle for x-axis labels
        return_metadata: If True, return dict with 'figure' and 'data_quality' keys
    
    Returns:
        matplotlib Figure object, or dict with 'figure' and 'data_quality' if return_metadata=True
    """
    if category_column not in df.columns or stack_column not in df.columns:
        raise ValueError(f"One or both columns not found in dataframe")
    
    # Get data quality info for both columns
    cat_quality = get_data_quality_info(df, category_column)
    stack_quality = get_data_quality_info(df, stack_column)
    # Combined quality info (rows with both columns valid)
    valid_both = df[[category_column, stack_column]].dropna()
    combined_quality = {
        'total_rows': len(df),
        'valid_count': len(valid_both),
        'missing_count': len(df) - len(valid_both),
        'missing_percentage': ((len(df) - len(valid_both)) / len(df) * 100) if len(df) > 0 else 0,
        'category_column_quality': cat_quality,
        'stack_column_quality': stack_quality
    }
    
    # Create crosstab (automatically drops rows with missing values in either column)
    crosstab = pd.crosstab(df[category_column], df[stack_column])
    
    if crosstab.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        if return_metadata:
            return {'figure': fig, 'data_quality': combined_quality}
        return fig
    
    # Define color mapping for risk colors
    # Using colors from impact type pie chart for consistency:
    # Environment (#fb8072) -> Red, Injury (#8dd3c7) -> Green, Injury/Illness (#ffffb3) -> Yellow
    risk_color_map = {
        'RED': '#fb8072',      # Red (from Environment color in impact type pie)
        'Red': '#fb8072',
        'red': '#fb8072',
        'YELLOW': '#ffffb3',   # Yellow (from Injury/Illness color in impact type pie)
        'Yellow': '#ffffb3',
        'yellow': '#ffffb3',
        'GREEN': '#8dd3c7',    # Green (from Injury color in impact type pie)
        'Green': '#8dd3c7',
        'green': '#8dd3c7'
    }
    
    # Reorder columns based on severity (if stack_column is RISK_COLOR)
    # For stacked bars: first column is at bottom, last column is at top
    # We want: Green (lowest) at bottom, Yellow in middle, Red (highest) at top
    if stack_column.upper() == 'RISK_COLOR':
        # Define order: Green first (bottom), then Yellow, then Red (top)
        severity_order = ['Green', 'GREEN', 'green', 'Yellow', 'YELLOW', 'yellow', 'Red', 'RED', 'red']
        
        # Get available colors in the data, ordered by severity
        available_colors = [col for col in severity_order if col in crosstab.columns]
        # Add any other colors not in our mapping
        other_colors = [col for col in crosstab.columns if col not in available_colors]
        # Final order: Green (bottom), Yellow, Red (top), then others
        ordered_colors = available_colors + other_colors
        crosstab = crosstab[ordered_colors]
        
        # Create color list for plotting (matching the column order)
        plot_colors = [risk_color_map.get(col, '#CCCCCC') for col in crosstab.columns]
    else:
        # For other stack columns, use default colors
        plot_colors = None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot stacked bars with custom colors if available
    if plot_colors and len(plot_colors) == len(crosstab.columns):
        # Plot with custom colors (Green at bottom, Red at top)
        crosstab.plot(kind='bar', stacked=True, ax=ax, color=plot_colors, alpha=0.8)
    else:
        # Fallback to default colormap
        crosstab.plot(kind='bar', stacked=True, ax=ax, colormap='Set3', alpha=0.8)
    
    ax.set_xlabel(category_column)
    ax.set_ylabel('Count')
    
    # Add data quality info to title
    chart_title = title or f"Stacked Bar Chart: {stack_column} by {category_column}"
    if combined_quality['missing_count'] > 0:
        chart_title += f"\n(Valid: {combined_quality['valid_count']:,}, Missing: {combined_quality['missing_count']:,})"
    ax.set_title(chart_title)
    ax.legend(title=stack_column, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=rotation)
    
    plt.tight_layout()
    
    if return_metadata:
        return {'figure': fig, 'data_quality': combined_quality}
    return fig

