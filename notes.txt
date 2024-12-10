pip install pandas openpyxl
import pandas as pd
import numpy as np
from itertools import combinations

# Function to load the Excel file
def load_excel(file_path):
    """Load the Excel file into a pandas DataFrame."""
    return pd.read_excel(file_path)

# Function to preprocess the data
def preprocess_data(data):
    """Preprocess the data to clean and format it."""
    # Convert time columns to datetime
    data['START_TIME'] = pd.to_datetime(data['START_TIME'])
    data['END_TIME'] = pd.to_datetime(data['END_TIME'])

    # Fill missing participant IDs with email if available
    data['FIRMWIDE_ID'] = data['FIRMWIDE_ID'].fillna(data['EMAIL'])
    
    # Replace missing functional roles or titles with a default value
    data['FUNCTIONAL_ROLE'] = data['FUNCTIONAL_ROLE'].fillna(data['TITLE'])
    data['FUNCTIONAL_ROLE'] = data['FUNCTIONAL_ROLE'].fillna('Default Role')
    
    return data

# Function to aggregate participants for each interaction
def aggregate_interactions(data):
    """Aggregate rows corresponding to the same ID into a single interaction."""
    def aggregate_func(group):
        return pd.Series({
            'START_TIME': group['START_TIME'].iloc[0],
            'END_TIME': group['END_TIME'].iloc[0],
            'SUBJECT': group['SUBJECT'].iloc[0],
            'TAGS': list(set(tag for tags in group['TAGS'].dropna() for tag in tags.split(','))),
            'PURPOSE': group['PURPOSE'].iloc[0],
            'PARTICIPANTS': group['FIRMWIDE_ID'].dropna().tolist(),
            'IS_EMPLOYEE': group['IS_EMPLOYEE'].tolist(),
            'FUNCTIONAL_ROLES': group['FUNCTIONAL_ROLE'].tolist()
        })
    
    return data.groupby('ID').apply(aggregate_func).reset_index()

# Function to calculate time difference
def calculate_time_difference(start1, start2):
    """Calculate the time difference in minutes."""
    return abs((start1 - start2).total_seconds() / 60)

# Function to calculate duration match percentage
def calculate_duration_match(start1, end1, start2, end2):
    """Calculate the percentage of duration match."""
    duration1 = (end1 - start1).total_seconds() / 60
    duration2 = (end2 - start2).total_seconds() / 60
    return min(duration1, duration2) / max(duration1, duration2) * 100

# Function to calculate overlap percentage
def calculate_overlap_percentage(start1, end1, start2, end2):
    """Calculate the percentage of overlap."""
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    overlap_duration = max(0, (overlap_end - overlap_start).total_seconds() / 60)
    duration1 = (end1 - start1).total_seconds() / 60
    return overlap_duration / duration1 * 100 if duration1 > 0 else 0

# Function to calculate participant matching
def calculate_participant_matching(participants1, participants2):
    """Calculate the percentage of matching participants."""
    set1 = set(participants1)
    set2 = set(participants2)
    if not set1:
        return 0
    return len(set1 & set2) / len(set1) * 100

# Function to calculate subject matching
def calculate_subject_matching(subject1, subject2):
    """Calculate subject matching as binary (1 if matching, 0 otherwise)."""
    return 100 if subject1 == subject2 else 0

# Function to calculate tag matching
def calculate_tag_matching(tags1, tags2):
    """Calculate the percentage of matching tags."""
    set1 = set(tags1)
    set2 = set(tags2)
    if not set1:
        return 0
    return len(set1 & set2) / len(set1) * 100

# Function to extract pairwise features
def extract_pairwise_features(aggregated_data, attribute_value):
    """Extract pairwise features for all interactions within the same ATTRIBUTE_VALUE."""
    features = []
    group = aggregated_data[aggregated_data['ATTRIBUTE_VALUE'] == attribute_value]
    
    for pair in combinations(group.to_dict('records'), 2):
        int1, int2 = pair
        
        time_diff = calculate_time_difference(int1['START_TIME'], int2['START_TIME'])
        duration_match = calculate_duration_match(int1['START_TIME'], int1['END_TIME'], int2['START_TIME'], int2['END_TIME'])
        overlap_percentage = calculate_overlap_percentage(int1['START_TIME'], int1['END_TIME'], int2['START_TIME'], int2['END_TIME'])
        internal_match = calculate_participant_matching(
            [p for p, e in zip(int1['PARTICIPANTS'], int1['IS_EMPLOYEE']) if e],
            [p for p, e in zip(int2['PARTICIPANTS'], int2['IS_EMPLOYEE']) if e]
        )
        external_match = calculate_participant_matching(
            [p for p, e in zip(int1['PARTICIPANTS'], int1['IS_EMPLOYEE']) if not e],
            [p for p, e in zip(int2['PARTICIPANTS'], int2['IS_EMPLOYEE']) if not e]
        )
        subject_match = calculate_subject_matching(int1['SUBJECT'], int2['SUBJECT'])
        tag_match = calculate_tag_matching(int1['TAGS'], int2['TAGS'])
        
        features.append({
            'ID1': int1['ID'],
            'ID2': int2['ID'],
            'Time Difference': time_diff,
            'Duration Match': duration_match,
            'Overlap Percentage': overlap_percentage,
            'Internal Matching': internal_match,
            'External Matching': external_match,
            'Subject Matching': subject_match,
            'Tag Matching': tag_match
        })
    
    return pd.DataFrame(features)

# Main function to process and extract features from Excel
def main(file_path, attribute_value):
    data = load_excel(file_path)
    data = preprocess_data(data)
    aggregated_data = aggregate_interactions(data)
    pairwise_features = extract_pairwise_features(aggregated_data, attribute_value)
    return pairwise_features

# Example Usage
file_path = 'your_excel_file.xlsx'  # Replace with your Excel file path
attribute_value = 'your_attribute_value'  # Replace with a specific ATTRIBUTE_VALUE
features = main(file_path, attribute_value)
print(features)
