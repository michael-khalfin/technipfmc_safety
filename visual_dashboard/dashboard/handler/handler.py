"""
Handler for processing chart generation requests.
Handles data filtering and chart generation based on user requirements.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add the handler directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph_makers import (
    generate_statistical_summary,
    generate_bar_chart_all_categories,
    generate_bar_chart_top_k,
    generate_pie_chart,
    generate_heatmap,
    generate_wordcloud,
    generate_temporal_distribution,
    generate_stacked_bar_chart,
    get_data_quality_info,
    generate_event_cluster_plot
)

# Path to data file
# please find this file at /projects/dsci435/fmcsafetyevents_fa25/data/merged_incidents_tsne.csv
DATA_FILE = Path(__file__).parent.parent.parent.parent / "data" / "merged_incidents_tsne.csv"

# Cache for loaded data
_data_cache = None


def load_data():
    """
    Load the merged incidents data.
    Uses caching to avoid reloading on every request.
    
    Returns:
        DataFrame with incident data
    """
    global _data_cache
    
    if _data_cache is None:
        if not DATA_FILE.exists():
            raise FileNotFoundError(f"Data file not found: {DATA_FILE}")
        _data_cache = pd.read_csv(DATA_FILE, low_memory=False)
        print(f"Data loaded: {len(_data_cache)} rows, {len(_data_cache.columns)} columns")
    
    return _data_cache.copy()


def is_numeric_column(df, column):
    """
    Check if a column is numeric.
    
    Args:
        df: DataFrame
        column: Column name
    
    Returns:
        True if column is numeric, False otherwise
    """
    if column not in df.columns:
        return False
    
    dtype = df[column].dtype
    return pd.api.types.is_numeric_dtype(dtype)


def filter_data_by_requirements(df, filter_requirements):
    """
    Filter data based on filter requirements dictionary.
    
    For numerical columns: expects a dict with 'min' and 'max' keys
    For categorical columns: expects a string value
    
    Args:
        df: DataFrame to filter
        filter_requirements: Dictionary where:
            - key: column name
            - value: 
                - For numerical: dict with 'min' and/or 'max' keys
                - For categorical: string value to match
    
    Returns:
        Filtered DataFrame
    """
    df_filtered = df.copy()
    
    if not filter_requirements or len(filter_requirements) == 0:
        return df_filtered
    
    for column, requirement in filter_requirements.items():
        if column not in df_filtered.columns:
            print(f"Warning: Column '{column}' not found in data. Skipping filter.")
            continue
        
        # Check if column is numeric
        if is_numeric_column(df_filtered, column):
            # Numerical column: expect dict with 'min' and/or 'max'
            if isinstance(requirement, dict):
                min_val = requirement.get('min')
                max_val = requirement.get('max')
                
                if min_val is not None:
                    df_filtered = df_filtered[df_filtered[column] >= min_val]
                
                if max_val is not None:
                    df_filtered = df_filtered[df_filtered[column] <= max_val]
            else:
                print(f"Warning: Numerical column '{column}' requires dict with 'min'/'max' keys. Got {type(requirement)}. Skipping filter.")
        
        else:
            # Categorical column: expect string value
            if isinstance(requirement, str):
                df_filtered = df_filtered[df_filtered[column] == requirement]
            elif isinstance(requirement, list):
                # Allow list of values for categorical columns
                df_filtered = df_filtered[df_filtered[column].isin(requirement)]
            else:
                print(f"Warning: Categorical column '{column}' requires string or list value. Got {type(requirement)}. Skipping filter.")
    
    return df_filtered


def generate_chart(chart_type, filter_requirements=None, return_metadata=False, **kwargs):
    """
    Main handler function to generate charts based on type and filter requirements.
    
    Args:
        chart_type: String specifying the chart type. Options:
            - 'statistical_summary': Big numbers statistics
            - 'temporal_distribution': Temporal distribution bar chart
            - 'top_locations': Top K locations bar chart
            - 'top_units': Top K units bar chart
            - 'incident_type': Incident type bar/pie chart
            - 'risk_color': Risk color pie/stacked bar chart
            - 'top_causes': Top K causes/hazards bar chart
            - 'severity_likelihood_heatmap': Severity vs Likelihood heatmap
            - 'wordcloud': Word cloud
            - 'bar_chart_all': Bar chart for all categories
            - 'bar_chart_top_k': Bar chart for top K
            - 'pie_chart': Pie chart
            - 'heatmap': Generic heatmap
            - 'stacked_bar': Stacked bar chart
        
        filter_requirements: Dictionary with filtering requirements:
            - For numerical columns: {'column_name': {'min': value, 'max': value}}
            - For categorical columns: {'column_name': 'value'}
        
        return_metadata: If True, return dict with 'chart' and 'data_quality' keys
        
        **kwargs: Additional arguments specific to each chart type
    
    Returns:
        For 'statistical_summary': Dictionary with statistics (or dict with 'summary' and 'data_quality' if return_metadata=True)
        For other chart types: matplotlib Figure object (or dict with 'chart' and 'data_quality' if return_metadata=True)
    """
    # Load data
    df = load_data()
    
    # Apply filters if provided
    if filter_requirements:
        df = filter_data_by_requirements(df, filter_requirements)
        # print(f"After filtering: {len(df)} rows")
    
    if len(df) == 0:
        raise ValueError("No data available after filtering. Please adjust filter requirements.")
    
    # Generate chart based on type
    chart_type = chart_type.lower().strip()
    
    if chart_type == 'statistical_summary':
        summary = generate_statistical_summary(df)
        if return_metadata:
            # For statistical summary, data quality is already included in the summary
            # But we can add overall data quality info
            return {
                'summary': summary,
                'data_quality': {
                    'total_rows': len(df),
                    'missing_count': 0,  # Statistical summary uses all rows
                    'valid_count': len(df),
                    'missing_percentage': 0.0
                }
            }
        return summary
    
    elif chart_type == 'temporal_distribution':
        groupby = kwargs.get('groupby', 'year')
        time_column = kwargs.get('time_column', 'DATE_OF_INCIDENT')
        title = kwargs.get('title', None)
        figsize = kwargs.get('figsize', (12, 6))
        result = generate_temporal_distribution(df, time_column=time_column, 
                                            groupby=groupby, title=title, 
                                            return_metadata=return_metadata)
        if return_metadata:
            return {'chart': result['figure'], 'data_quality': result['data_quality']}
        return result
    
    elif chart_type == 'top_locations':
        k = kwargs.get('k', 10)
        location_column = kwargs.get('location_column', 'WORKPLACE_CITY')
        title = kwargs.get('title', f'Top {k} Locations by Incident Count')
        horizontal = kwargs.get('horizontal', False)
        result = generate_bar_chart_top_k(df, location_column, k=k, title=title, 
                                         horizontal=horizontal, return_metadata=return_metadata)
        if return_metadata:
            return {'chart': result['figure'], 'data_quality': result['data_quality']}
        return result
    
    elif chart_type == 'top_units':
        k = kwargs.get('k', 10)
        unit_column = kwargs.get('unit_column', 'BU')
        title = kwargs.get('title', f'Top {k} Units by Incident Count')
        horizontal = kwargs.get('horizontal', False)
        result = generate_bar_chart_top_k(df, unit_column, k=k, title=title, 
                                         horizontal=horizontal, return_metadata=return_metadata)
        if return_metadata:
            return {'chart': result['figure'], 'data_quality': result['data_quality']}
        return result
    
    elif chart_type == 'incident_type':
        chart_subtype = kwargs.get('subtype', 'bar')  # 'bar' or 'pie'
        column = kwargs.get('column', 'INCIDENT_TYPE')
        title = kwargs.get('title', 'Incident Type Distribution')
        
        if chart_subtype == 'pie':
            top_n = kwargs.get('top_n', None)
            result = generate_pie_chart(df, column, title=title, top_n=top_n, return_metadata=return_metadata)
        else:
            result = generate_bar_chart_all_categories(df, column, title=title, return_metadata=return_metadata)
        
        if return_metadata:
            return {'chart': result['figure'], 'data_quality': result['data_quality']}
        return result
    
    elif chart_type == 'risk_color':
        chart_subtype = kwargs.get('subtype', 'pie')  # 'pie' or 'stacked'
        column = kwargs.get('column', 'RISK_COLOR')
        title = kwargs.get('title', 'Risk Color Distribution')
        
        if chart_subtype == 'stacked':
            category_column = kwargs.get('category_column', 'LOSS_POTENTIAL_SEVERITY')
            result = generate_stacked_bar_chart(df, category_column, column, title=title, return_metadata=return_metadata)
        else:
            top_n = kwargs.get('top_n', None)
            result = generate_pie_chart(df, column, title=title, top_n=top_n, return_metadata=return_metadata)
        
        if return_metadata:
            return {'chart': result['figure'], 'data_quality': result['data_quality']}
        return result
    
    elif chart_type == 'top_causes':
        k = kwargs.get('k', 10)
        column = kwargs.get('column', 'CASE_CATEGORIZATION')
        title = kwargs.get('title', f'Top {k} Causes/Hazards')
        horizontal = kwargs.get('horizontal', False)
        result = generate_bar_chart_top_k(df, column, k=k, title=title, 
                                         horizontal=horizontal, return_metadata=return_metadata)
        if return_metadata:
            return {'chart': result['figure'], 'data_quality': result['data_quality']}
        return result
    
    elif chart_type == 'severity_likelihood_heatmap':
        x_column = kwargs.get('x_column', 'LOSS_POTENTIAL_SEVERITY')
        y_column = kwargs.get('y_column', 'LIKELIHOOD_VALUE')
        title = kwargs.get('title', 'Severity vs Likelihood Heatmap')
        result = generate_heatmap(df, x_column, y_column, title=title, return_metadata=return_metadata)
        if return_metadata:
            return {'chart': result['figure'], 'data_quality': result['data_quality']}
        return result
    
    elif chart_type == 'wordcloud':
        text_column = kwargs.get('text_column', 'TITLE')
        title = kwargs.get('title', f'Word Cloud: {text_column}')
        max_words = kwargs.get('max_words', 100)
        result = generate_wordcloud(df, text_column, title=title, max_words=max_words, return_metadata=return_metadata)
        if return_metadata:
            return {'chart': result['figure'], 'data_quality': result['data_quality']}
        return result
    
    elif chart_type == 'bar_chart_all':
        column = kwargs.get('column')
        if not column:
            raise ValueError("'column' parameter is required for 'bar_chart_all'")
        title = kwargs.get('title', None)
        top_n = kwargs.get('top_n', None)
        result = generate_bar_chart_all_categories(df, column, title=title, top_n=top_n, return_metadata=return_metadata)
        if return_metadata:
            return {'chart': result['figure'], 'data_quality': result['data_quality']}
        return result
    
    elif chart_type == 'bar_chart_top_k':
        column = kwargs.get('column')
        if not column:
            raise ValueError("'column' parameter is required for 'bar_chart_top_k'")
        k = kwargs.get('k', 10)
        title = kwargs.get('title', None)
        horizontal = kwargs.get('horizontal', False)
        result = generate_bar_chart_top_k(df, column, k=k, title=title, horizontal=horizontal, return_metadata=return_metadata)
        if return_metadata:
            return {'chart': result['figure'], 'data_quality': result['data_quality']}
        return result
    
    elif chart_type == 'pie_chart':
        column = kwargs.get('column')
        if not column:
            raise ValueError("'column' parameter is required for 'pie_chart'")
        title = kwargs.get('title', None)
        top_n = kwargs.get('top_n', None)
        result = generate_pie_chart(df, column, title=title, top_n=top_n, return_metadata=return_metadata)
        if return_metadata:
            return {'chart': result['figure'], 'data_quality': result['data_quality']}
        return result
    
    elif chart_type == 'heatmap':
        x_column = kwargs.get('x_column')
        y_column = kwargs.get('y_column')
        if not x_column or not y_column:
            raise ValueError("'x_column' and 'y_column' parameters are required for 'heatmap'")
        title = kwargs.get('title', None)
        result = generate_heatmap(df, x_column, y_column, title=title, return_metadata=return_metadata)
        if return_metadata:
            return {'chart': result['figure'], 'data_quality': result['data_quality']}
        return result
    
    elif chart_type == 'stacked_bar':
        category_column = kwargs.get('category_column')
        stack_column = kwargs.get('stack_column')
        if not category_column or not stack_column:
            raise ValueError("'category_column' and 'stack_column' parameters are required for 'stacked_bar'")
        title = kwargs.get('title', None)
        result = generate_stacked_bar_chart(df, category_column, stack_column, title=title, return_metadata=return_metadata)
        if return_metadata:
            return {'chart': result['figure'], 'data_quality': result['data_quality']}
        return result
    
    elif chart_type == 'event_cluster':
        x_col = kwargs.get('x_col', 'tsne_x')
        y_col = kwargs.get('y_col', 'tsne_y')
        sample_ratio = kwargs.get('sample_ratio', 0.1)
        color_column = kwargs.get('color_column', None)
        title = kwargs.get('title', 'Event Cluster Visualization')

        result = generate_event_cluster_plot(
            df,
            x_col=x_col,
            y_col=y_col,
            sample_ratio=sample_ratio,
            color_column=color_column,
            return_metadata=return_metadata
        )

        if return_metadata:
            return {
                "chart": result["figure"],
                "data_quality": result["data_quality"]
            }
        return result


    else:
        raise ValueError(f"Unknown chart type: {chart_type}. "
                        f"Available types: statistical_summary, temporal_distribution, "
                        f"top_locations, top_units, incident_type, risk_color, top_causes, "
                        f"severity_likelihood_heatmap, wordcloud, bar_chart_all, "
                        f"bar_chart_top_k, pie_chart, heatmap, stacked_bar")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example 1: Statistical summary with filters
    print("Example 1: Statistical Summary")
    filter_req = {
        'YEAR_OF_INCIDENT': {'min': 2020, 'max': 2024},
        'GBU': 'Surface'
    }
    summary = generate_chart('statistical_summary', filter_req)
    print(summary)
    print()
    
    # Example 2: Temporal distribution
    print("Example 2: Temporal Distribution")
    filter_req = {
        'YEAR_OF_INCIDENT': {'min': 2020, 'max': 2024}
    }
    fig = generate_chart('temporal_distribution', filter_req, groupby='year')
    print("Chart generated successfully")
    print()
    
    # Example 3: Top locations
    print("Example 3: Top 10 Locations")
    filter_req = {
        'INCIDENT_TYPE': 'Accident'
    }
    fig = generate_chart('top_locations', filter_req, k=10)
    print("Chart generated successfully")

