import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from app.schemas import LabelUpdateRequest
from typing import List, Dict, Any, Optional

# ====================== MAIN FUNCTION ======================
def update_label(customer_id: str, churn_label: str) -> Dict[str, Any]:
    """
    Update or add Churn label for a specific customer by ID.
    
    Args:
        customer_id: Customer ID to update
        churn_label: True label ("Yes" = churn, "No" = no churn)
    
    Returns:
        Dictionary with status and details of the operation
    """
    file_path = Path('data/new_data/new_data.csv')
    
    # Validate input
    if churn_label not in ["Yes", "No"]:
        return {
            "status": "error",
            "message": "Churn label must be 'Yes' or 'No'",
            "customerID": customer_id
        }
    
    # Check if file exists
    if not file_path.exists():
        return {
            "status": "error",
            "message": f"Data file not found at {file_path}",
            "customerID": customer_id
        }
    
    # Read existing data
    df = pd.read_csv(file_path)
    
    # Check if customer exists
    customer_id_str = str(customer_id)
    mask = df['customerID'].astype(str) == customer_id_str
    
    if not mask.any():
        return {
            "status": "error",
            "message": f"Customer ID '{customer_id}' not found in the database",
            "customerID": customer_id,
            "available_customers": df['customerID'].astype(str).tolist()[:10]
        }
    
    # Update the label
    old_label = df.loc[mask, 'Churn'].iloc[0]
    df.loc[mask, 'Churn'] = churn_label
    df.loc[mask, 'label_timestamp'] = datetime.utcnow().isoformat()
    
    # Save back to CSV
    df.to_csv(file_path, index=False)
    
    return {
        "status": "success",
        "message": f"Label updated for customer {customer_id}",
        "customerID": customer_id,
        "old_label": old_label if pd.notna(old_label) else None,
        "new_label": churn_label,
        "timestamp": datetime.utcnow().isoformat()
    }


# ====================== BATCH UPDATE FUNCTION ======================
def batch_update_labels(updates: List[LabelUpdateRequest]) -> Dict[str, Any]:
    """
    Update multiple Churn labels at once.
    
    Args:
        updates: List of LabelUpdateRequest objects
    
    Returns:
        Dictionary with summary of updates
    """
    file_path = Path('data/new_data/new_data.csv')
    
    if not file_path.exists():
        return {
            "status": "error",
            "message": f"Data file not found at {file_path}",
            "successful_updates": [],
            "failed_updates": []
        }
    
    df = pd.read_csv(file_path)
    results = []
    
    for update in updates:
        customer_id = update.customerID
        churn_label = update.Churn
        
        # Validate label
        if churn_label not in ["Yes", "No"]:
            results.append({
                "customerID": customer_id,
                "status": "failed",
                "message": "Churn label must be 'Yes' or 'No'"
            })
            continue
        
        mask = df['customerID'].astype(str) == str(customer_id)
        
        if mask.any():
            old_label = df.loc[mask, 'Churn'].iloc[0]
            df.loc[mask, 'Churn'] = churn_label
            df.loc[mask, 'label_timestamp'] = datetime.utcnow().isoformat()
            results.append({
                "customerID": customer_id,
                "status": "success",
                "old_label": old_label if pd.notna(old_label) else None,
                "new_label": churn_label
            })
        else:
            results.append({
                "customerID": customer_id,
                "status": "failed",
                "message": "Customer ID not found"
            })
    
    # Save all changes at once
    df.to_csv(file_path, index=False)
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    
    return {
        "status": "completed",
        "total_updates": len(updates),
        "successful": len(successful),
        "failed": len(failed),
        "results": results
    }


# ====================== GET LABEL FUNCTION ======================
def get_label(customer_id: str) -> Dict[str, Any]:
    """
    Get the current Churn label for a specific customer.
    
    Args:
        customer_id: Customer ID to look up
    
    Returns:
        Dictionary with label information
    """
    file_path = Path('data/new_data/new_data.csv')
    
    if not file_path.exists():
        return {
            "status": "error",
            "message": f"Data file not found at {file_path}",
            "customerID": customer_id
        }
    
    df = pd.read_csv(file_path)
    mask = df['customerID'].astype(str) == str(customer_id)
    
    if not mask.any():
        return {
            "status": "error",
            "message": f"Customer ID '{customer_id}' not found",
            "customerID": customer_id
        }
    
    row = df.loc[mask].iloc[0]
    churn_value = row.get('Churn')
    
    return {
        "status": "success",
        "customerID": customer_id,
        "Churn": churn_value if pd.notna(churn_value) else None,
        "prediction": int(row.get('prediction')) if pd.notna(row.get('prediction')) else None,
        "probability": float(row.get('probability')) if pd.notna(row.get('probability')) else None,
        "timestamp": row.get('timestamp'),
        "label_timestamp": row.get('label_timestamp') if 'label_timestamp' in row else None
    }


# ====================== GET UNLABELED DATA ======================
def get_unlabeled_data() -> pd.DataFrame:
    """
    Get all records that don't have Churn labels yet.
    
    Returns:
        DataFrame with unlabeled records
    """
    file_path = Path('data/new_data/new_data.csv')
    
    if not file_path.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    unlabeled = df[df['Churn'].isna()]
    
    return unlabeled


# ====================== GET LABELED DATA ======================
def get_labeled_data() -> pd.DataFrame:
    """
    Get all records that have Churn labels.
    
    Returns:
        DataFrame with labeled records
    """
    file_path = Path('data/new_data/new_data.csv')
    
    if not file_path.exists():
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    labeled = df[df['Churn'].notna()]
    
    return labeled



# ====================== STATISTICS FUNCTION ======================
def get_label_statistics() -> Dict[str, Any]:
    """
    Get statistics about labeled vs unlabeled data.
    
    Returns:
        Dictionary with statistics
    """
    file_path = Path('data/new_data/new_data.csv')
    
    if not file_path.exists():
        return {
            "status": "error",
            "message": f"Data file not found at {file_path}"
        }
    
    df = pd.read_csv(file_path)
    
    labeled = df[df['Churn'].notna()]
    unlabeled = df[df['Churn'].isna()]
    
    if len(labeled) > 0:
        churn_yes = len(labeled[labeled['Churn'] == 'Yes'])
        churn_no = len(labeled[labeled['Churn'] == 'No'])
    else:
        churn_yes = 0
        churn_no = 0
    
    return {
        "status": "success",
        "total_records": len(df),
        "labeled_records": len(labeled),
        "unlabeled_records": len(unlabeled),
        "churn_yes": churn_yes,
        "churn_no": churn_no,
        "labeling_progress": f"{len(labeled)}/{len(df)} ({len(labeled)/len(df)*100:.1f}%)"
    }

