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


def prepare_categorical_data(df, column, dropna=True):
    """
    Prepare categorical data for visualization.
    
    Args:
        df: DataFrame
        column: Name of the categorical column
        dropna: Whether to drop NaN values
    
    Returns:
        Series with value counts
    """
    if column not in df.columns:
        return pd.Series(dtype=int)
    
    data = df[column].copy()
    if dropna:
        data = data.dropna()
    
    return data.value_counts()


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
    
    # High-risk (red) incidents percentage
    if 'RISK_COLOR' in df.columns:
        red_incidents = len(df[df['RISK_COLOR'].str.upper() == 'RED'])
        summary['high_risk_count'] = red_incidents
        summary['high_risk_percentage'] = (red_incidents / len(df) * 100) if len(df) > 0 else 0
    else:
        summary['high_risk_count'] = 0
        summary['high_risk_percentage'] = 0
    
    # Average severity score
    if 'SEVERITY_VALUE' in df.columns:
        severity_values = pd.to_numeric(df['SEVERITY_VALUE'], errors='coerce')
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
    
    # Number of open corrective actions
    if 'STATUS' in df.columns:
        open_actions = len(df[df['STATUS'].str.upper() == 'OPEN'])
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
                                       figsize=(12, 6), rotation=45, top_n=None):
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
    
    Returns:
        matplotlib Figure object
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    # Prepare data
    value_counts = prepare_categorical_data(df, column)
    
    if len(value_counts) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title or f"Bar Chart: {column}")
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
    ax.set_title(title or f"Bar Chart: {column}")
    
    # Add value labels on bars
    for i, (idx, val) in enumerate(value_counts.items()):
        ax.text(i, val, f'{int(val):,}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return fig


# ============================================================================
# 3. Bar Chart - Top K
# ============================================================================

def generate_bar_chart_top_k(df, column, k=10, title=None, xlabel=None, ylabel="Count",
                             figsize=(12, 6), rotation=45, horizontal=False):
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
    
    Returns:
        matplotlib Figure object
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    # Prepare data
    value_counts = prepare_categorical_data(df, column)
    
    if len(value_counts) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title or f"Top {k}: {column}")
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
    
    ax.set_title(title or f"Top {k}: {column}")
    plt.tight_layout()
    return fig


# ============================================================================
# 4. Pie Chart
# ============================================================================

def generate_pie_chart(df, column, title=None, figsize=(10, 8), top_n=None, 
                       autopct='%1.1f%%', startangle=90):
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
    
    Returns:
        matplotlib Figure object
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")
    
    # Prepare data
    value_counts = prepare_categorical_data(df, column)
    
    if len(value_counts) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title or f"Pie Chart: {column}")
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
    
    # Plot
    colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
    wedges, texts, autotexts = ax.pie(value_counts.values, labels=value_counts.index,
                                       autopct=autopct, startangle=startangle, colors=colors)
    
    # Customize text
    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
    
    ax.set_title(title or f"Pie Chart: {column}", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig


# ============================================================================
# 5. Heatmap
# ============================================================================

def generate_heatmap(df, x_column, y_column, title=None, figsize=(12, 8), 
                     cmap='YlOrRd', annot=True, fmt='d', cbar_label="Count"):
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
    
    Returns:
        matplotlib Figure object
    """
    if x_column not in df.columns or y_column not in df.columns:
        raise ValueError(f"One or both columns not found in dataframe")
    
    # Create crosstab
    crosstab = pd.crosstab(df[y_column], df[x_column])
    
    if crosstab.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title or f"Heatmap: {x_column} vs {y_column}")
        return fig
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    sns.heatmap(crosstab, annot=annot, fmt=fmt, cmap=cmap, ax=ax,
                cbar_kws={'label': cbar_label}, linewidths=0.5, linecolor='gray')
    
    ax.set_title(title or f"Heatmap: {x_column} vs {y_column}", fontsize=12, fontweight='bold')
    ax.set_xlabel(x_column, fontsize=10)
    ax.set_ylabel(y_column, fontsize=10)
    
    plt.tight_layout()
    return fig


# ============================================================================
# 6. Word Cloud
# ============================================================================

def generate_wordcloud(df, text_column, title=None, figsize=(12, 8), max_words=100,
                       width=800, height=400, background_color='white', 
                       colormap='viridis', stopwords=None):
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
    
    Returns:
        matplotlib Figure object
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in dataframe")
    
    # Combine all text
    text_data = df[text_column].dropna().astype(str)
    
    if len(text_data) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No text data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title or f"Word Cloud: {text_column}")
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
    ax.set_title(title or f"Word Cloud: {text_column}", fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig


# ============================================================================
# Specialized Functions for Dashboard Requirements
# ============================================================================

def generate_temporal_distribution(df, time_column='DATE_OF_INCIDENT', 
                                  groupby='year', title=None, figsize=(12, 6)):
    """
    Generate temporal distribution bar chart (by year/month/date).
    
    Args:
        df: DataFrame
        time_column: Name of the date column
        groupby: 'year', 'month', or 'date'
        title: Chart title
        figsize: Figure size tuple
    
    Returns:
        matplotlib Figure object
    """
    if time_column not in df.columns:
        raise ValueError(f"Column '{time_column}' not found in dataframe")
    
    # Convert to datetime
    df_temp = df.copy()
    df_temp[time_column] = pd.to_datetime(df_temp[time_column], errors='coerce')
    df_temp = df_temp.dropna(subset=[time_column])
    
    if len(df_temp) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No date data available', ha='center', va='center', transform=ax.transAxes)
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
    ax.set_title(title or f"Incident Frequency by {xlabel}")
    
    # Add value labels
    for i, val in enumerate(counts.values):
        ax.text(i, val, f'{int(val):,}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    return fig


def generate_stacked_bar_chart(df, category_column, stack_column, title=None,
                               figsize=(12, 6), rotation=45):
    """
    Generate a stacked bar chart showing distribution of stack_column within categories.
    
    Args:
        df: DataFrame
        category_column: Column for x-axis categories
        stack_column: Column to stack (e.g., RISK_COLOR, SEVERITY)
        title: Chart title
        figsize: Figure size tuple
        rotation: Rotation angle for x-axis labels
    
    Returns:
        matplotlib Figure object
    """
    if category_column not in df.columns or stack_column not in df.columns:
        raise ValueError(f"One or both columns not found in dataframe")
    
    # Create crosstab
    crosstab = pd.crosstab(df[category_column], df[stack_column])
    
    if crosstab.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot stacked bars
    crosstab.plot(kind='bar', stacked=True, ax=ax, colormap='Set3', alpha=0.8)
    
    ax.set_xlabel(category_column)
    ax.set_ylabel('Count')
    ax.set_title(title or f"Stacked Bar Chart: {stack_column} by {category_column}")
    ax.legend(title=stack_column, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.tick_params(axis='x', rotation=rotation)
    
    plt.tight_layout()
    return fig

