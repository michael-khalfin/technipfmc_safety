#!/usr/bin/env python3
"""
Data preprocessing and merging script
Tasks:
1. Standardize column names: rename DESCRIPTIN to DESCRIPTION
2. Vertically merge 7 event files, keeping only common features
3. Check for duplicates by RECORD_NO and remove duplicates
4. Horizontally merge event files and loss potential files
5. Handle missing values and report missing statistics
6. Feature engineering: split date/time, workplace, and work process fields
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Define data file paths
DATA_DIR = Path("data")

# 7 event files
EVENT_FILES = [
    "DIM_CONSOLIDATED_ACCIDENTS_translated.csv",
    "DIM_CONSOLIDATED_HAZARD_OBSERVATIONS_translated.csv",
    "DIM_CONSOLIDATED_NEAR_MISSES_translated.csv",
    "DIM_INTELEX_ACCIDENTS.csv",
    "DIM_INTELEX_HAZARD_OBSERVATIONS.csv",
    "DIM_INTELEX_INCIDENTS.csv",
    "DIM_INTELEX_NEAR_MISSES.csv"
]

# 2 loss potential files
LOSS_POTENTIAL_FILES = [
    "DIM_CONSOLIDATED_LOSS_POTENTIAL.csv",
    "DIM_INTELEX_LOSS_POTENTIAL.csv"
]

def find_common_columns(file_paths):
    """Find columns common to all files"""
    all_columns = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, nrows=0)
        all_columns.append(set(df.columns))
    
    # Find columns present in all files
    common_columns = set.intersection(*all_columns)
    return sorted(list(common_columns))

def load_and_standardize_event_file(file_path):
    """Load event file and standardize column names"""
    print(f"Loading: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    
    # Step 1: Standardize column names - rename DESCRIPTIN to DESCRIPTION
    if 'DESCRIPTIN' in df.columns:
        df = df.rename(columns={'DESCRIPTIN': 'DESCRIPTION'})
        print(f"  Renamed DESCRIPTIN to DESCRIPTION")
    
    return df

def merge_event_files():
    """Step 2: Vertically merge 7 event files, keeping only common features"""
    print("\n" + "="*80)
    print("Step 2: Merging event files")
    print("="*80)
    
    # Find columns common to all event files
    file_paths = [DATA_DIR / f for f in EVENT_FILES]
    common_columns = find_common_columns(file_paths)
    
    print(f"\nFound {len(common_columns)} common columns:")
    for col in common_columns:
        print(f"  - {col}")
    
    # Load all files, keeping only common columns
    dataframes = []
    for file_path in file_paths:
        df = load_and_standardize_event_file(file_path)
        # Keep only common columns
        df_common = df[common_columns].copy()
        # Add source identifiers
        df_common['SOURCE_FILE'] = file_path.name
        df_common['SOURCE_SYSTEM'] = 'CONSOLIDATED' if 'CONSOLIDATED' in file_path.name else 'INTELEX'
        dataframes.append(df_common)
        print(f"  Loaded {len(df_common)} rows")
    
    # Vertical merge
    merged_df = pd.concat(dataframes, ignore_index=True)
    print(f"\nMerged total: {len(merged_df)} rows")
    
    return merged_df, common_columns

def remove_duplicates(df):
    """Step 3: Check and remove duplicate data by RECORD_NO"""
    print("\n" + "="*80)
    print("Step 3: Checking and removing duplicates")
    print("="*80)
    
    initial_count = len(df)
    print(f"Total rows before deduplication: {initial_count}")
    
    # Check for duplicate RECORD_NO
    duplicate_records = df[df.duplicated(subset=['RECORD_NO'], keep=False)]
    if len(duplicate_records) > 0:
        print(f"\nFound {len(duplicate_records)} rows with duplicate RECORD_NO")
        print("\nSample duplicate RECORD_NO:")
        duplicate_record_nos = duplicate_records['RECORD_NO'].unique()[:10]
        for record_no in duplicate_record_nos:
            count = len(df[df['RECORD_NO'] == record_no])
            print(f"  RECORD_NO {record_no}: appears {count} times")
    
    # Remove duplicates, keep first occurrence
    df_deduplicated = df.drop_duplicates(subset=['RECORD_NO'], keep='first')
    removed_count = initial_count - len(df_deduplicated)
    
    print(f"\nAfter deduplication: {len(df_deduplicated)} rows")
    print(f"Removed {removed_count} duplicate rows")
    
    return df_deduplicated

def merge_loss_potential(df_events, common_columns):
    """Step 4: Horizontally merge event files and loss potential files"""
    print("\n" + "="*80)
    print("Step 4: Merging loss potential data")
    print("="*80)
    
    # Load loss potential files
    loss_potential_dfs = []
    for file_path in [DATA_DIR / f for f in LOSS_POTENTIAL_FILES]:
        print(f"Loading: {file_path}")
        df_lp = pd.read_csv(file_path, low_memory=False)
        loss_potential_dfs.append(df_lp)
        print(f"  Loaded {len(df_lp)} rows")
    
    # Merge two loss potential files
    df_loss_potential = pd.concat(loss_potential_dfs, ignore_index=True)
    print(f"\nMerged loss potential data: {len(df_loss_potential)} rows")
    
    # Check for duplicate RECORD_NO_LOSS_POTENTIAL in loss potential
    duplicate_lp = df_loss_potential.duplicated(subset=['RECORD_NO_LOSS_POTENTIAL'], keep=False)
    if duplicate_lp.sum() > 0:
        print(f"Warning: Loss Potential data has {duplicate_lp.sum()} rows with duplicate RECORD_NO_LOSS_POTENTIAL")
        print("Keeping first occurrence")
        df_loss_potential = df_loss_potential.drop_duplicates(subset=['RECORD_NO_LOSS_POTENTIAL'], keep='first')
        print(f"After deduplication: {len(df_loss_potential)} rows")
    
    # Get RECORD_NO_LOSS_POTENTIAL values that exist in event files (convert to same data type for matching)
    # Ensure consistent data types
    df_events['RECORD_NO_LOSS_POTENTIAL'] = pd.to_numeric(df_events['RECORD_NO_LOSS_POTENTIAL'], errors='coerce')
    df_loss_potential['RECORD_NO_LOSS_POTENTIAL'] = pd.to_numeric(df_loss_potential['RECORD_NO_LOSS_POTENTIAL'], errors='coerce')
    
    event_lp_records = set(df_events['RECORD_NO_LOSS_POTENTIAL'].dropna().unique())
    print(f"Unique RECORD_NO_LOSS_POTENTIAL in event data: {len(event_lp_records)}")
    print(f"Unique RECORD_NO_LOSS_POTENTIAL in loss potential data: {df_loss_potential['RECORD_NO_LOSS_POTENTIAL'].nunique()}")
    
    # Keep only loss potential records that match event files
    df_loss_potential_filtered = df_loss_potential[
        df_loss_potential['RECORD_NO_LOSS_POTENTIAL'].isin(event_lp_records)
    ].copy()
    print(f"\nFiltered loss potential data: {len(df_loss_potential_filtered)} rows (only keeping RECORD_NO_LOSS_POTENTIAL that exist in event files)")
    
    # Find columns common to both loss potential and event files (excluding RECORD_NO_LOSS_POTENTIAL)
    lp_columns = set(df_loss_potential_filtered.columns)
    event_columns = set(common_columns)
    columns_in_both = lp_columns.intersection(event_columns) - {'RECORD_NO_LOSS_POTENTIAL'}
    lp_unique_columns = lp_columns - event_columns
    
    print(f"\nAll columns in Loss Potential files: {sorted(lp_columns)}")
    print(f"Columns common to both: {sorted(columns_in_both)}")
    print(f"Unique columns in Loss Potential (will be added): {sorted(lp_unique_columns)}")
    
    # Merge: based on RECORD_NO_LOSS_POTENTIAL, keep all loss potential columns
    print(f"\nMerging data...")
    
    # For common columns, keep event file value if present, otherwise use loss potential value
    # For loss potential unique columns, add directly
    
    # First handle common columns
    if columns_in_both:
        print(f"\nProcessing common columns: {sorted(columns_in_both)}")
        # Use merge with suffixes to distinguish common columns
        df_merged = df_events.merge(
            df_loss_potential_filtered[['RECORD_NO_LOSS_POTENTIAL'] + list(columns_in_both)],
            on='RECORD_NO_LOSS_POTENTIAL',
            how='left',
            suffixes=('', '_lp')
        )
        
        # For common columns, keep event file value if present, otherwise use loss potential value
        for col in columns_in_both:
            # If event file value is empty, use loss potential value
            mask = df_merged[col].isna() | (df_merged[col] == '')
            df_merged.loc[mask, col] = df_merged.loc[mask, f'{col}_lp']
            df_merged = df_merged.drop(columns=[f'{col}_lp'])
    else:
        df_merged = df_events.copy()
    
    # Add loss potential unique columns
    if lp_unique_columns:
        print(f"\nAdding Loss Potential unique columns: {sorted(lp_unique_columns)}")
        df_merged = df_merged.merge(
            df_loss_potential_filtered[['RECORD_NO_LOSS_POTENTIAL'] + list(lp_unique_columns)],
            on='RECORD_NO_LOSS_POTENTIAL',
            how='left'
        )
    
    print(f"\nAfter merge: {len(df_merged)} rows, {len(df_merged.columns)} columns")
    
    return df_merged

def parse_datetime_time(time_str):
    """Parse time string and extract hour, minute, second
    Note: DATE_TIME_OF_INCIDENT format appears to be MM:SS.S (minute:second.decimal)
    Based on data inspection, values like "40:00.0" represent minutes:seconds
    So we interpret: hour=None (not available), minute=first_part, second=second_part
    """
    if pd.isna(time_str) or time_str == '':
        return None, None, None
    
    try:
        # Handle formats like "40:00.0" which appears to be MM:SS.S (minute:second.decimal)
        time_str = str(time_str).strip()
        if ':' in time_str:
            parts = time_str.split(':')
            first_part = int(float(parts[0])) if parts[0] else None
            second_part = parts[1] if len(parts) > 1 else '0.0'
            
            # If first part > 23, it's likely minutes (MM:SS format)
            # Otherwise, it could be hours (HH:MM format)
            # Based on data inspection, format appears to be MM:SS.S
            if first_part is not None and first_part > 23:
                # MM:SS.S format: first part is minutes
                minute = first_part
                hour = None
            else:
                # Could be HH:MM format, but data suggests MM:SS
                # Treat as MM:SS for consistency
                minute = first_part
                hour = None
            
            # Parse second part
            if '.' in second_part:
                second = int(float(second_part.split('.')[0])) if second_part.split('.')[0] else None
            else:
                second = int(float(second_part)) if second_part else None
            
            return hour, minute, second
        else:
            return None, None, None
    except:
        return None, None, None

def parse_date(date_str):
    """Parse date string (M/D/YYYY) and extract year, month, day"""
    if pd.isna(date_str) or date_str == '':
        return None, None, None
    
    try:
        date_str = str(date_str).strip()
        # Handle formats like "1/15/2021" or "10/29/2024"
        if '/' in date_str:
            parts = date_str.split('/')
            if len(parts) == 3:
                month = int(parts[0]) if parts[0] else None
                day = int(parts[1]) if parts[1] else None
                year = int(parts[2]) if parts[2] else None
                return year, month, day
        return None, None, None
    except:
        return None, None, None

def split_datetime_incident(df):
    """Split DATE_TIME_OF_INCIDENT and DATE_OF_INCIDENT into separate components"""
    print("\n" + "="*80)
    print("Feature Engineering: Splitting DATE_TIME_OF_INCIDENT and DATE_OF_INCIDENT")
    print("="*80)
    
    # Parse DATE_OF_INCIDENT for year, month, day
    date_parts = df['DATE_OF_INCIDENT'].apply(parse_date)
    df['YEAR_OF_INCIDENT'] = date_parts.apply(lambda x: x[0] if x else None)
    df['MONTH_OF_INCIDENT'] = date_parts.apply(lambda x: x[1] if x else None)
    df['DAY_OF_INCIDENT'] = date_parts.apply(lambda x: x[2] if x else None)
    
    # Parse DATE_TIME_OF_INCIDENT for hour, minute, second
    time_parts = df['DATE_TIME_OF_INCIDENT'].apply(parse_datetime_time)
    df['HOUR_OF_INCIDENT'] = time_parts.apply(lambda x: x[0] if x else None)
    df['MINUTE_OF_INCIDENT'] = time_parts.apply(lambda x: x[1] if x else None)
    df['SECOND_OF_INCIDENT'] = time_parts.apply(lambda x: x[2] if x else None)
    
    print(f"Created 6 new columns: YEAR_OF_INCIDENT, MONTH_OF_INCIDENT, DAY_OF_INCIDENT, HOUR_OF_INCIDENT, MINUTE_OF_INCIDENT, SECOND_OF_INCIDENT")
    
    return df

def split_workplace(df):
    """Split WORKPLACE by comma into base, city, country, area, company"""
    print("\n" + "="*80)
    print("Feature Engineering: Splitting WORKPLACE")
    print("="*80)
    
    def parse_workplace(workplace_str):
        """Parse workplace string: 'base, city, country, area, company'"""
        if pd.isna(workplace_str) or workplace_str == '':
            return [None] * 6
        
        try:
            parts = [p.strip() for p in str(workplace_str).split(',')]
            # Expected format: base, city, country, area, company
            # But company might be missing
            base = parts[0] if len(parts) > 0 else None
            city = parts[1] if len(parts) > 1 else None
            country = parts[2] if len(parts) > 2 else None
            area = parts[3] if len(parts) > 3 else None
            company = parts[4] if len(parts) > 4 else None
            
            return [base, city, country, area, company]
        except:
            return [None] * 6
    
    workplace_parts = df['WORKPLACE'].apply(parse_workplace)
    df['WORKPLACE_BASE'] = workplace_parts.apply(lambda x: x[0])
    df['WORKPLACE_CITY'] = workplace_parts.apply(lambda x: x[1])
    df['WORKPLACE_COUNTRY'] = workplace_parts.apply(lambda x: x[2])
    df['WORKPLACE_AREA'] = workplace_parts.apply(lambda x: x[3])
    df['WORKPLACE_COMPANY'] = workplace_parts.apply(lambda x: x[4])
    
    print(f"Created 5 new columns: WORKPLACE_BASE, WORKPLACE_CITY, WORKPLACE_COUNTRY, WORKPLACE_AREA, WORKPLACE_COMPANY")
    
    return df

def split_work_process(df):
    """Split WORK_PROCESS by '-' into type and phase"""
    print("\n" + "="*80)
    print("Feature Engineering: Splitting WORK_PROCESS")
    print("="*80)
    
    def parse_work_process(work_process_str):
        """Parse work process string: 'type - phase'"""
        if pd.isna(work_process_str) or work_process_str == '':
            return None, None
        
        try:
            # Split by '-' but keep the parts
            parts = [p.strip() for p in str(work_process_str).split('-', 1)]
            work_type = parts[0] if len(parts) > 0 else None
            work_phase = parts[1] if len(parts) > 1 else None
            return work_type, work_phase
        except:
            return None, None
    
    work_process_parts = df['WORK_PROCESS'].apply(parse_work_process)
    df['WORK_PROCESS_TYPE'] = work_process_parts.apply(lambda x: x[0])
    df['WORK_PROCESS_PHASE'] = work_process_parts.apply(lambda x: x[1])
    
    print(f"Created 2 new columns: WORK_PROCESS_TYPE, WORK_PROCESS_PHASE")
    
    return df

def split_date_fields(df):
    """Split date fields (DATE_REPORTED, DATE_OF_APPROVAL, DATE_OF_CLOSURE, DUE_DATE) into year, month, day"""
    print("\n" + "="*80)
    print("Feature Engineering: Splitting date fields")
    print("="*80)
    
    # Define field mappings: (original_field, suffix_for_new_columns)
    date_field_mappings = [
        ('DATE_REPORTED', 'REPORTED'),
        ('DATE_OF_APPROVAL', 'APPROVAL'),
        ('DATE_OF_CLOSURE', 'CLOSURE'),
        ('DUE_DATE', 'DUE')
    ]
    
    for date_field, suffix in date_field_mappings:
        if date_field not in df.columns:
            continue
        
        date_parts = df[date_field].apply(parse_date)
        df[f'YEAR_{suffix}'] = date_parts.apply(lambda x: x[0] if x else None)
        df[f'MONTH_{suffix}'] = date_parts.apply(lambda x: x[1] if x else None)
        # User specified "date_REPORTED" which we interpret as day component
        # Use DAY_ prefix for all date fields to avoid conflict with original DATE_ fields
        df[f'DAY_{suffix}'] = date_parts.apply(lambda x: x[2] if x else None)
        
        print(f"Split {date_field} into YEAR_{suffix}, MONTH_{suffix}, DAY_{suffix}")
    
    return df

def process_features(df):
    """Step 6: Feature engineering - split various fields"""
    print("\n" + "="*80)
    print("Step 6: Feature Engineering")
    print("="*80)
    
    # Split DATE_TIME_OF_INCIDENT and DATE_OF_INCIDENT
    df = split_datetime_incident(df)
    
    # Split WORKPLACE
    df = split_workplace(df)
    
    # Split WORK_PROCESS
    df = split_work_process(df)
    
    # Split date fields
    df = split_date_fields(df)
    
    print(f"\nFeature engineering complete. Total columns: {len(df.columns)}")
    
    return df

def analyze_missing_values(df):
    """Step 5: Analyze missing values"""
    print("\n" + "="*80)
    print("Step 5: Missing Value Analysis")
    print("="*80)
    
    # Overall missing statistics
    missing_stats = df.isnull().sum()
    missing_percentage = (missing_stats / len(df)) * 100
    
    print("\nMissing values by column:")
    print(f"{'Column':<40} {'Missing Count':<15} {'Missing %':<15}")
    print("-" * 70)
    
    missing_info = []
    for col in df.columns:
        missing_count = missing_stats[col]
        missing_pct = missing_percentage[col]
        if missing_count > 0:
            print(f"{col:<40} {missing_count:<15} {missing_pct:>6.2f}%")
            missing_info.append({
                'column': col,
                'missing_count': missing_count,
                'missing_percentage': missing_pct
            })
    
    if not missing_info:
        print("No missing values!")
    else:
        # Sort by missing percentage
        missing_info.sort(key=lambda x: x['missing_percentage'], reverse=True)
        print(f"\nTop 10 columns with most missing values:")
        for i, info in enumerate(missing_info[:10], 1):
            print(f"{i}. {info['column']}: {info['missing_count']} ({info['missing_percentage']:.2f}%)")
    
    # Check missing values per row
    print("\n" + "-" * 80)
    print("Row-level missing value analysis:")
    print("-" * 80)
    
    row_missing = df.isnull().sum(axis=1)
    total_columns = len(df.columns)
    
    print(f"Total columns: {total_columns}")
    print(f"\nMissing columns statistics:")
    print(f"  No missing: {(row_missing == 0).sum()} rows")
    print(f"  Missing 1-5 columns: {((row_missing >= 1) & (row_missing <= 5)).sum()} rows")
    print(f"  Missing 6-10 columns: {((row_missing >= 6) & (row_missing <= 10)).sum()} rows")
    print(f"  Missing 11-20 columns: {((row_missing >= 11) & (row_missing <= 20)).sum()} rows")
    print(f"  Missing 21+ columns: {(row_missing >= 21).sum()} rows")
    
    # Find rows with most missing values
    if row_missing.max() > 0:
        print(f"\nTop 10 rows with most missing values:")
        worst_rows = row_missing.nlargest(10)
        for idx, missing_count in worst_rows.items():
            missing_pct = (missing_count / total_columns) * 100
            record_no = df.loc[idx, 'RECORD_NO'] if 'RECORD_NO' in df.columns else 'N/A'
            print(f"  Row {idx} (RECORD_NO: {record_no}): {missing_count} missing columns ({missing_pct:.2f}%)")
    
    # Fill missing values (empty string for object columns, keep NaN for numeric)
    print("\n" + "-" * 80)
    print("Filling missing values...")
    print("-" * 80)
    
    df_filled = df.copy()
    for col in df_filled.columns:
        if df_filled[col].dtype == 'object':
            # Fill object columns with empty string
            df_filled[col] = df_filled[col].fillna('')
        # Numeric columns keep NaN
    
    print("Missing values filled (object columns with empty string, numeric columns keep NaN)")
    
    return df_filled

def main():
    """Main function"""
    print("="*80)
    print("Data Preprocessing and Merging")
    print("="*80)
    
    # Step 2: Merge event files
    df_events, common_columns = merge_event_files()
    
    # Step 3: Remove duplicates
    df_events = remove_duplicates(df_events)
    
    # Step 4: Merge loss potential
    df_merged = merge_loss_potential(df_events, common_columns)
    
    # Step 6: Feature engineering
    df_merged = process_features(df_merged)
    
    # Step 5: Analyze missing values and fill
    df_final = analyze_missing_values(df_merged)
    
    # Save results
    output_file = DATA_DIR / "merged_incidents.csv"
    print("\n" + "="*80)
    print(f"Saving merged data to: {output_file}")
    print("="*80)
    df_final.to_csv(output_file, index=False)
    print(f"Saved {len(df_final)} rows, {len(df_final.columns)} columns")
    
    # Save missing values report
    report_file = DATA_DIR / "missing_values_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("Missing Values Analysis Report\n")
        f.write("="*80 + "\n\n")
        
        missing_stats = df_final.isnull().sum()
        missing_percentage = (missing_stats / len(df_final)) * 100
        
        f.write("Missing values by column:\n")
        f.write(f"{'Column':<40} {'Missing Count':<15} {'Missing %':<15}\n")
        f.write("-" * 70 + "\n")
        
        for col in df_final.columns:
            missing_count = missing_stats[col]
            missing_pct = missing_percentage[col]
            if missing_count > 0:
                f.write(f"{col:<40} {missing_count:<15} {missing_pct:>6.2f}%\n")
    
    print(f"Missing values report saved to: {report_file}")
    
    return df_final

if __name__ == "__main__":
    df_final = main()

