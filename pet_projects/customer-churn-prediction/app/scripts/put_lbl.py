# app/scripts/put_lbl.py
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from app.core.database import get_db
from app.utils.logger import logger

def update_label(customer_id: str, churn_label: str) -> Dict[str, Any]:
    """Update or add Churn label for a specific customer"""
    if churn_label not in ["Yes", "No"]:
        return {
            "status": "error",
            "message": "Churn label must be 'Yes' or 'No'",
            "customerID": customer_id
        }
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Check if customer exists
        row = cursor.execute(
            "SELECT churn_label FROM new_data WHERE customer_id = ?",
            (customer_id,)
        ).fetchone()
        
        if not row:
            return {
                "status": "error",
                "message": f"Customer ID '{customer_id}' not found in database",
                "customerID": customer_id
            }
        
        old_label = row['churn_label']
        
        # Update label
        cursor.execute('''
            UPDATE new_data 
            SET churn_label = ?, label_timestamp = CURRENT_TIMESTAMP
            WHERE customer_id = ?
        ''', (churn_label, customer_id))
        
        conn.commit()
    
    return {
        "status": "success",
        "message": f"Label updated for customer {customer_id}",
        "customerID": customer_id,
        "old_label": old_label,
        "new_label": churn_label,
        "timestamp": datetime.now().isoformat()
    }

def batch_update_labels(updates: List) -> Dict[str, Any]:
    """Update multiple Churn labels at once"""
    results = []
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        for update in updates:
            customer_id = update.customerID
            churn_label = update.Churn
            
            if churn_label not in ["Yes", "No"]:
                results.append({
                    "customerID": customer_id,
                    "status": "failed",
                    "message": "Churn label must be 'Yes' or 'No'"
                })
                continue
            
            row = cursor.execute(
                "SELECT churn_label FROM new_data WHERE customer_id = ?",
                (customer_id,)
            ).fetchone()
            
            if row:
                old_label = row['churn_label']
                cursor.execute('''
                    UPDATE new_data 
                    SET churn_label = ?, label_timestamp = CURRENT_TIMESTAMP
                    WHERE customer_id = ?
                ''', (churn_label, customer_id))
                
                results.append({
                    "customerID": customer_id,
                    "status": "success",
                    "old_label": old_label,
                    "new_label": churn_label
                })
            else:
                results.append({
                    "customerID": customer_id,
                    "status": "failed",
                    "message": "Customer ID not found"
                })
        
        conn.commit()
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    
    return {
        "status": "completed",
        "total_updates": len(updates),
        "successful": len(successful),
        "failed": len(failed),
        "results": results
    }

def get_label(customer_id: str) -> Dict[str, Any]:
    """Get current Churn label for a customer"""
    with get_db() as conn:
        row = conn.execute('''
            SELECT customer_id, churn_label, prediction, probability, 
                   created_at, label_timestamp
            FROM new_data WHERE customer_id = ?
        ''', (customer_id,)).fetchone()
        
        if not row:
            return {
                "status": "error",
                "message": f"Customer ID '{customer_id}' not found",
                "customerID": customer_id
            }
    
    return {
        "status": "success",
        "customerID": row['customer_id'],
        "Churn": row['churn_label'],
        "prediction": row['prediction'],
        "probability": row['probability'],
        "timestamp": row['created_at'],
        "label_timestamp": row['label_timestamp']
    }

def get_unlabeled_data() -> pd.DataFrame:
    """Get all records without Churn labels"""
    with get_db() as conn:
        df = pd.read_sql_query('''
            SELECT customer_id, prediction, probability, created_at
            FROM new_data WHERE churn_label IS NULL
        ''', conn)
    
    if not df.empty:
        df = df.rename(columns={'customer_id': 'customerID'})
    return df

def get_labeled_data() -> pd.DataFrame:
    """Get all records with Churn labels"""
    with get_db() as conn:
        df = pd.read_sql_query('''
            SELECT customer_id, churn_label as Churn, prediction, probability, label_timestamp
            FROM new_data WHERE churn_label IS NOT NULL
        ''', conn)
    
    if not df.empty:
        df = df.rename(columns={'customer_id': 'customerID'})
    return df

def get_label_statistics() -> Dict[str, Any]:
    """Get statistics about labeled vs unlabeled data"""
    with get_db() as conn:
        stats = conn.execute('''
            SELECT 
                COUNT(*) as total_records,
                SUM(CASE WHEN churn_label IS NOT NULL THEN 1 ELSE 0 END) as labeled_records,
                SUM(CASE WHEN churn_label = 'Yes' THEN 1 ELSE 0 END) as churn_yes,
                SUM(CASE WHEN churn_label = 'No' THEN 1 ELSE 0 END) as churn_no
            FROM new_data
        ''').fetchone()
    
    total = stats['total_records']
    labeled = stats['labeled_records']
    unlabeled = total - labeled
    
    progress = f"{labeled}/{total} ({labeled/total*100:.1f}%)" if total > 0 else "0/0 (0%)"
    
    return {
        "status": "success",
        "total_records": total,
        "labeled_records": labeled,
        "unlabeled_records": unlabeled,
        "churn_yes": stats['churn_yes'] or 0,
        "churn_no": stats['churn_no'] or 0,
        "labeling_progress": progress
    }