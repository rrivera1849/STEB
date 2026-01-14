import os
import csv
from typing import Dict, List, Any


def load_onestop_english_corpus(data_dir: str) -> List[Dict[str, Any]]:
    """
    Load OneStopEnglishCorpus dataset from CSV files.
    
    Reads all CSV files from the Texts-Together-OneCSVperFile directory
    and returns records where all three complexity levels (Elementary, 
    Intermediate, Advanced) have non-null entries.
    
    Args:
        data_dir: Path to the dataset directory
        
    Returns:
        List of records with 'text' (ordered list [advanced, intermediate, elementary]) 
        and 'label' ("complexity") fields.
    """
    csv_dir = os.path.join(data_dir, "Texts-Together-OneCSVperFile")
    
    if not os.path.exists(csv_dir):
        raise ValueError(f"Directory not found: {csv_dir}")
    
    records = []
    
    # Process all CSV files in the directory
    for filename in os.listdir(csv_dir):
        if not filename.endswith('.csv'):
            continue
            
        csv_path = os.path.join(csv_dir, filename)
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Get the three complexity levels
                elementary = row.get('Elementary', '').strip()
                intermediate = row.get('Intermediate', '').strip()
                advanced = row.get('Advanced', '').strip()
                
                # Only include rows where all three levels are non-null
                if elementary and intermediate and advanced:
                    records.append({
                        'text': [advanced, intermediate, elementary],
                        'label': "complexity"
                    })
    
    return records


def onestop_english_corpus_record_handler(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Custom record handler for the OneStopEnglishCorpus dataset.
    Extracts 'text' and 'label' from each record.
    
    Args:
        record: Record dictionary with 'text' and 'label' fields
        
    Returns:
        Dictionary with 'text' and 'label' fields.
    """
    return {
        'text': record.get('text'),
        'label': record.get('label')
    }
