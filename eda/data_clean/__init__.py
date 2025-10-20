"""
Data cleaning and integration module for safety incident analysis.

This package provides comprehensive data cleaning, analysis, and integration
capabilities for safety incident datasets, including column analysis,
data merging, and dataset loading functionality.
"""

from .data_loader import DataLoader
from .column_analyzer import ColumnAnalyzer
from .coaleser import Coalescer
__all__ = ['DataLoader', 'ColumnAnalyzer', 'Coalescer']