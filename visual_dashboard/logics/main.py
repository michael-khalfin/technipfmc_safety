"""
Main module for Streamlit dashboard backend.
Handles initialization of all charts and user interaction requests.
"""

import sys
from pathlib import Path
import os

# Add handler directory to path
handler_dir = Path(__file__).parent / "handler"
sys.path.insert(0, str(handler_dir))

from handler import generate_chart


# ============================================================================
# Default Configuration
# ============================================================================

DEFAULT_TOP_K = 10  # Default value for top K charts


# ============================================================================
# Initialize All Charts Function
# ============================================================================

def initialize_all_charts(filter_requirements=None):
    """
    Initialize all charts for the dashboard when the webpage is first opened.
    
    Args:
        filter_requirements: Optional dictionary with initial filter requirements.
                            If None, uses all data.
    
    Returns:
        Dictionary containing all chart objects, statistical summary, and data quality info:
        {
            'big_numbers': dict,  # Statistical summary
            'temporal_distribution': matplotlib.Figure,
            'top_locations': matplotlib.Figure,
            'top_units': matplotlib.Figure,
            'incident_type_pie': matplotlib.Figure,
            'impact_type_pie': matplotlib.Figure,
            'severity_risk_stacked': matplotlib.Figure,
            'top_causes': matplotlib.Figure,
            'severity_likelihood_heatmap': matplotlib.Figure,
            'wordcloud': matplotlib.Figure,
            'data_quality': dict  # Data quality info for each chart
        }
    """
    charts = {}
    data_quality_info = {}
    
    # 1. Big Numbers - Statistical Summary
    try:
        result = generate_chart('statistical_summary', filter_requirements, return_metadata=True)
        charts['big_numbers'] = result['summary']
        data_quality_info['big_numbers'] = result['data_quality']
    except Exception as e:
        print(f"Error generating big numbers: {e}")
        charts['big_numbers'] = {
            'total_incidents': 0,
            'high_risk_count': 0,
            'high_risk_percentage': 0,
            'avg_severity': 0,
            'avg_likelihood': 0,
            'open_actions': 0
        }
        data_quality_info['big_numbers'] = {
            'total_rows': 0,
            'missing_count': 0,
            'valid_count': 0,
            'missing_percentage': 0.0
        }
    
    # 2. Temporal Distribution Bar Chart - by year
    try:
        result = generate_chart(
            'temporal_distribution',
            filter_requirements,
            groupby='year',
            title='Incident Frequency by Year',
            return_metadata=True
        )
        charts['temporal_distribution'] = result['chart']
        data_quality_info['temporal_distribution'] = result['data_quality']
    except Exception as e:
        print(f"Error generating temporal distribution: {e}")
        charts['temporal_distribution'] = None
        data_quality_info['temporal_distribution'] = None
    
    # 2. Temporal Distribution Bar Chart - by year
    try:
        result = generate_chart(
            'temporal_distribution',
            filter_requirements,
            groupby='year',
            title='Incident Frequency by Year',
            return_metadata=True
        )
        charts['temporal_distribution'] = result['chart']
        data_quality_info['temporal_distribution'] = result['data_quality']
    except Exception as e:
        print(f"Error generating temporal distribution: {e}")
        charts['temporal_distribution'] = None
        data_quality_info['temporal_distribution'] = None
    
    # 3. Spatial Distribution - Top K Locations (WORKPLACE_CITY)
    try:
        result = generate_chart(
            'top_locations',
            filter_requirements,
            k=DEFAULT_TOP_K,
            location_column='WORKPLACE_CITY',
            title=f'Top {DEFAULT_TOP_K} Locations by Incident Count',
            return_metadata=True
        )
        charts['top_locations'] = result['chart']
        data_quality_info['top_locations'] = result['data_quality']
    except Exception as e:
        print(f"Error generating top locations: {e}")
        charts['top_locations'] = None
        data_quality_info['top_locations'] = None
    
    # 4. Spatial Distribution - Top K Units (BU)
    try:
        result = generate_chart(
            'top_units',
            filter_requirements,
            k=DEFAULT_TOP_K,
            unit_column='BU',
            title=f'Top {DEFAULT_TOP_K} Units by Incident Count',
            return_metadata=True
        )
        charts['top_units'] = result['chart']
        data_quality_info['top_units'] = result['data_quality']
    except Exception as e:
        print(f"Error generating top units: {e}")
        charts['top_units'] = None
        data_quality_info['top_units'] = None
    
    # 5. Categorical Pie Chart - Incident Type (INCIDENT_TYPE)
    try:
        result = generate_chart(
            'pie_chart',
            filter_requirements,
            column='INCIDENT_TYPE',
            title='Incident Type Distribution',
            return_metadata=True
        )
        charts['incident_type_pie'] = result['chart']
        data_quality_info['incident_type_pie'] = result['data_quality']
    except Exception as e:
        print(f"Error generating incident type pie chart: {e}")
        charts['incident_type_pie'] = None
        data_quality_info['incident_type_pie'] = None
    
    # 6. Categorical Bar Chart - Impact Type (IMPACT_TYPE)
    try:
        result = generate_chart(
            'bar_chart_all',
            filter_requirements,
            column='IMPACT_TYPE',
            title='Impact Type Distribution',
            return_metadata=True
        )
        charts['impact_type_pie'] = result['chart']  # Keep same key name for compatibility
        data_quality_info['impact_type_pie'] = result['data_quality']
    except Exception as e:
        print(f"Error generating impact type bar chart: {e}")
        charts['impact_type_pie'] = None
        data_quality_info['impact_type_pie'] = None
    
    # 7. Categorical Stacked Bar Chart - Severity Level/Risk Color
    try:
        result = generate_chart(
            'stacked_bar',
            filter_requirements,
            category_column='LOSS_POTENTIAL_SEVERITY',
            stack_column='RISK_COLOR',
            title='Risk Color Distribution by Severity Level',
            return_metadata=True
        )
        charts['severity_risk_stacked'] = result['chart']
        data_quality_info['severity_risk_stacked'] = result['data_quality']
    except Exception as e:
        print(f"Error generating severity/risk stacked chart: {e}")
        charts['severity_risk_stacked'] = None
        data_quality_info['severity_risk_stacked'] = None
    
    # 8. Categorical Bar Chart - Top K Repeated Causes/Hazards (CASE_CATEGORIZATION)
    try:
        result = generate_chart(
            'top_causes',
            filter_requirements,
            k=DEFAULT_TOP_K,
            column='CASE_CATEGORIZATION',
            title=f'Top {DEFAULT_TOP_K} Causes/Hazards',
            return_metadata=True
        )
        charts['top_causes'] = result['chart']
        data_quality_info['top_causes'] = result['data_quality']
    except Exception as e:
        print(f"Error generating top causes: {e}")
        charts['top_causes'] = None
        data_quality_info['top_causes'] = None
    
    # 9. Severity vs. Likelihood Heatmap
    try:
        result = generate_chart(
            'heatmap',
            filter_requirements,
            x_column='LOSS_POTENTIAL_SEVERITY',
            y_column='LIKELIHOOD_VALUE',
            title='Severity vs Likelihood Heatmap',
            return_metadata=True
        )
        charts['severity_likelihood_heatmap'] = result['chart']
        data_quality_info['severity_likelihood_heatmap'] = result['data_quality']
    except Exception as e:
        print(f"Error generating severity/likelihood heatmap: {e}")
        charts['severity_likelihood_heatmap'] = None
        data_quality_info['severity_likelihood_heatmap'] = None
    
    # 10. Word Cloud
    try:
        result = generate_chart(
            'wordcloud',
            filter_requirements,
            text_column='TITLE',
            title='Word Cloud: Incident Titles',
            max_words=100,
            return_metadata=True
        )
        charts['wordcloud'] = result['chart']
        data_quality_info['wordcloud'] = result['data_quality']
    except Exception as e:
        print(f"Error generating word cloud: {e}")
        charts['wordcloud'] = None
        data_quality_info['wordcloud'] = None
    
    # Add data quality info to charts dictionary
    charts['data_quality'] = data_quality_info
    
    return charts


# ============================================================================
# User Interaction Handler
# ============================================================================

def generate_chart_with_filters(chart_type, filter_requirements=None, **kwargs):
    """
    Generate a specific chart based on user-selected filters.
    This function is called when the user modifies data selection range.
    
    Args:
        chart_type: String specifying which chart to generate. Options:
            - 'big_numbers': Statistical summary
            - 'temporal_distribution': Temporal distribution chart
            - 'top_locations': Top K locations chart
            - 'top_units': Top K units chart
            - 'incident_type_pie': Incident type pie chart
            - 'impact_type_pie': Impact type pie chart
            - 'severity_risk_stacked': Severity/risk stacked bar chart
            - 'top_causes': Top K causes chart
            - 'severity_likelihood_heatmap': Severity vs likelihood heatmap
            - 'wordcloud': Word cloud
        
        filter_requirements: Dictionary with filter requirements:
            - For numerical columns: {'column_name': {'min': value, 'max': value}}
            - For categorical columns: {'column_name': 'value'}
        
        **kwargs: Additional chart-specific parameters:
            - For temporal_distribution: groupby ('year', 'month', 'date')
            - For top_locations/top_units/top_causes: k (number of top items)
            - For wordcloud: text_column, max_words
    
    Returns:
        Dictionary with:
            - 'chart': matplotlib Figure object (or dict for big_numbers)
            - 'data_quality': Dictionary with data quality information:
                - 'total_rows': Total number of rows in filtered data
                - 'missing_count': Number of missing values in the relevant column(s)
                - 'valid_count': Number of valid (non-missing) values used in chart
                - 'missing_percentage': Percentage of missing values
    """
    chart_type = chart_type.lower().strip()
    
    # Always request metadata to get data quality info
    kwargs['return_metadata'] = True
    
    # Map chart types to handler chart types and parameters
    chart_mapping = {
        'big_numbers': {
            'handler_type': 'statistical_summary',
            'kwargs': {}
        },
        'temporal_distribution': {
            'handler_type': 'temporal_distribution',
            'kwargs': {
                'groupby': kwargs.get('groupby', 'year'),
                'title': kwargs.get('title', 'Incident Frequency by Year')
            }
        },
        'top_locations': {
            'handler_type': 'top_locations',
            'kwargs': {
                'k': kwargs.get('k', DEFAULT_TOP_K),
                'location_column': 'WORKPLACE_CITY',
                'title': kwargs.get('title', f"Top {kwargs.get('k', DEFAULT_TOP_K)} Locations by Incident Count")
            }
        },
        'top_units': {
            'handler_type': 'top_units',
            'kwargs': {
                'k': kwargs.get('k', DEFAULT_TOP_K),
                'unit_column': 'BU',
                'title': kwargs.get('title', f"Top {kwargs.get('k', DEFAULT_TOP_K)} Units by Incident Count")
            }
        },
        'incident_type_pie': {
            'handler_type': 'pie_chart',
            'kwargs': {
                'column': 'INCIDENT_TYPE',
                'title': kwargs.get('title', 'Incident Type Distribution')
            }
        },
        'impact_type_pie': {
            'handler_type': 'bar_chart_all',
            'kwargs': {
                'column': 'IMPACT_TYPE',
                'title': kwargs.get('title', 'Impact Type Distribution')
            }
        },
        'severity_risk_stacked': {
            'handler_type': 'stacked_bar',
            'kwargs': {
                'category_column': 'LOSS_POTENTIAL_SEVERITY',
                'stack_column': 'RISK_COLOR',
                'title': kwargs.get('title', 'Risk Color Distribution by Severity Level')
            }
        },
        'top_causes': {
            'handler_type': 'top_causes',
            'kwargs': {
                'k': kwargs.get('k', DEFAULT_TOP_K),
                'column': 'CASE_CATEGORIZATION',
                'title': kwargs.get('title', f"Top {kwargs.get('k', DEFAULT_TOP_K)} Causes/Hazards")
            }
        },
        'severity_likelihood_heatmap': {
            'handler_type': 'heatmap',
            'kwargs': {
                'x_column': 'LOSS_POTENTIAL_SEVERITY',
                'y_column': 'LIKELIHOOD_VALUE',
                'title': kwargs.get('title', 'Severity vs Likelihood Heatmap')
            }
        },
        'wordcloud': {
            'handler_type': 'wordcloud',
            'kwargs': {
                'text_column': kwargs.get('text_column', 'TITLE'),
                'title': kwargs.get('title', 'Word Cloud: Incident Titles'),
                'max_words': kwargs.get('max_words', 100)
            }
        }
    }
    
    if chart_type not in chart_mapping:
        raise ValueError(f"Unknown chart type: {chart_type}. "
                        f"Available types: {list(chart_mapping.keys())}")
    
    # Get handler chart type and parameters
    handler_config = chart_mapping[chart_type]
    handler_type = handler_config['handler_type']
    handler_kwargs = handler_config['kwargs']
    
    # Generate chart with metadata
    result = generate_chart(handler_type, filter_requirements, return_metadata=True, **handler_kwargs)
    
    # Format return value based on chart type
    if chart_type == 'big_numbers':
        return {
            'chart': result['summary'],
            'data_quality': result['data_quality']
        }
    else:
        return {
            'chart': result['chart'],
            'data_quality': result['data_quality']
        }


# ============================================================================
# Helper Functions for Streamlit Integration
# ============================================================================

def get_chart_info():
    """
    Get information about available charts for the frontend.
    
    Returns:
        Dictionary with chart metadata
    """
    return {
        'big_numbers': {
            'name': 'Big Numbers',
            'type': 'statistics',
            'description': 'Total incidents, high-risk percentage, average scores, open actions'
        },
        'temporal_distribution': {
            'name': 'Temporal Distribution',
            'type': 'bar_chart',
            'description': 'Incident frequency by year',
            'parameters': {
                'groupby': ['year', 'month', 'date']
            }
        },
        'top_locations': {
            'name': 'Top Locations',
            'type': 'bar_chart',
            'description': 'Top K locations by incident count',
            'parameters': {
                'k': 'integer (default: 10)'
            }
        },
        'top_units': {
            'name': 'Top Units',
            'type': 'bar_chart',
            'description': 'Top K units by incident count',
            'parameters': {
                'k': 'integer (default: 10)'
            }
        },
        'incident_type_pie': {
            'name': 'Incident Type',
            'type': 'pie_chart',
            'description': 'Distribution of incident types'
        },
        'impact_type_pie': {
            'name': 'Impact Type',
            'type': 'bar_chart',
            'description': 'Distribution of impact types'
        },
        'severity_risk_stacked': {
            'name': 'Severity/Risk Distribution',
            'type': 'stacked_bar',
            'description': 'Risk color distribution by severity level'
        },
        'top_causes': {
            'name': 'Top Causes/Hazards',
            'type': 'bar_chart',
            'description': 'Top K causes/hazards',
            'parameters': {
                'k': 'integer (default: 10)'
            }
        },
        'severity_likelihood_heatmap': {
            'name': 'Severity vs Likelihood',
            'type': 'heatmap',
            'description': 'Heatmap showing relationship between severity and likelihood'
        },
        'wordcloud': {
            'name': 'Word Cloud',
            'type': 'wordcloud',
            'description': 'Word cloud from incident titles',
            'parameters': {
                'text_column': 'string (default: "TITLE")',
                'max_words': 'integer (default: 100)'
            }
        }
    }


# ============================================================================
# Example Usage (for testing)
# ============================================================================

def save_all_charts(charts, output_dir=None):
    """
    Save all charts to files for inspection.
    
    Args:
        charts: Dictionary with chart objects
        output_dir: Output directory path (default: code/graphs/)
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "graphs"
    else:
        output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving charts to: {output_dir}")
    
    # Save big numbers as text file
    if 'big_numbers' in charts:
        summary = charts['big_numbers']
        summary_file = output_dir / "big_numbers.txt"
        with open(summary_file, 'w') as f:
            f.write("Big Numbers - Statistical Summary\n")
            f.write("=" * 50 + "\n\n")
            for key, value in summary.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.2f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        print(f"  ✓ Saved: {summary_file}")
    
    # Save all matplotlib figures
    figure_charts = {
        'temporal_distribution': 'Temporal Distribution by Year',
        'top_locations': 'Top 10 Locations',
        'top_units': 'Top 10 Units',
        'incident_type_pie': 'Incident Type Distribution',
        'impact_type_pie': 'Impact Type Distribution',
        'severity_risk_stacked': 'Severity Risk Stacked Bar',
        'top_causes': 'Top 10 Causes Hazards',
        'severity_likelihood_heatmap': 'Severity vs Likelihood Heatmap',
        'wordcloud': 'Word Cloud'
    }
    
    for chart_name, chart_obj in charts.items():
        if chart_name in ['big_numbers', 'data_quality']:
            continue
        
        if chart_obj is not None:
            try:
                # Save as PNG
                filename = f"{chart_name}.png"
                filepath = output_dir / filename
                chart_obj.savefig(filepath, dpi=150, bbox_inches='tight')
                print(f"  ✓ Saved: {filepath}")
                
                # Also save as PDF for better quality
                filename_pdf = f"{chart_name}.pdf"
                filepath_pdf = output_dir / filename_pdf
                chart_obj.savefig(filepath_pdf, bbox_inches='tight')
                print(f"  ✓ Saved: {filepath_pdf}")
            except Exception as e:
                print(f"  ✗ Error saving {chart_name}: {e}")
        else:
            print(f"  ✗ Skipped: {chart_name} (None)")
    
    print(f"\nAll charts saved to: {output_dir}")


if __name__ == "__main__":
    # Test initialization
    print("Initializing all charts...")
    charts = initialize_all_charts()
    
    print("\nCharts initialized:")
    for chart_name, chart_obj in charts.items():
        if chart_name == 'data_quality':
            print(f"  {chart_name}: Data quality info for {len(chart_obj)} charts")
        elif chart_name == 'big_numbers':
            print(f"  {chart_name}: {type(chart_obj).__name__} with {len(chart_obj)} metrics")
        else:
            print(f"  {chart_name}: {type(chart_obj).__name__ if chart_obj else 'None'}")
    
    # Save all charts to files
    save_all_charts(charts)
    
    # Test with filters
    print("\n\nTesting with filters...")
    filter_req = {
        'YEAR_OF_INCIDENT': {'min': 2020, 'max': 2024},
        #'GBU': 'Surface'
    }
    
    result = generate_chart_with_filters('temporal_distribution', filter_req)
    chart = result['chart']
    data_quality = result['data_quality']
    
    print(f"Generated chart type: {type(chart).__name__}")
    print(f"Data quality: Valid={data_quality['valid_count']:,}, Missing={data_quality['missing_count']:,} ({data_quality['missing_percentage']:.2f}%)")
    
    # Save filtered chart
    if chart is not None:
        output_dir = Path(__file__).parent / "graphs"
        output_dir.mkdir(parents=True, exist_ok=True)
        filtered_file = output_dir / "temporal_distribution_filtered.png"
        chart.savefig(filtered_file, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved filtered chart: {filtered_file}")
    
    print("\nAll tests completed!")

