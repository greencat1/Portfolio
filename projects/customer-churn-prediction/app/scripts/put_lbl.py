# app/scripts/put_lbl.py
"""
Label Management Module

WHAT THIS MODULE DOES:
Manages customer churn labels (ground truth) in the database.
Labels are used for model retraining and evaluation.

WHY LABELS ARE IMPORTANT:
- Model training needs labeled data (churn = Yes/No)
- Model evaluation compares predictions vs actual labels
- Retraining requires new labeled data to improve model

LABEL WORKFLOW:
1. Model predicts churn probability for unlabeled customers
2. Human experts review predictions and add actual labels
3. Labels are stored in database
4. When enough labels accumulate, model can be retrained

LABEL VALUES:
- "Yes": Customer actually churned (left the company)
- "No": Customer stayed (still with company)
- NULL: Not yet labeled (waiting for review)

DATABASE SCHEMA (new_data table):
- customerID: Unique identifier
- churn_label: "Yes", "No", or NULL
- prediction: Model's prediction (0/1)
- probability: Model's confidence
- label_timestamp: When label was added/updated
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from app.core.database import get_db
from app.utils.logger import logger


# ============================================================================
# SINGLE LABEL UPDATE
# ============================================================================

def update_label(customer_id: str, churn_label: str) -> Dict[str, Any]:
    """
    Update or add Churn label for a specific customer.
    
    USE CASE:
    After reviewing a customer's data, you know whether they actually churned.
    This function records that ground truth in the database.
    
    WORKFLOW:
    1. Validate label is "Yes" or "No"
    2. Check if customer exists in database
    3. Get old label (if any)
    4. Update churn_label and label_timestamp
    5. Return result with old and new labels
    
    WHAT HAPPENS TO OLD LABEL?
    Overwritten completely. No version history (only latest label kept).
    
    ARGS:
        customer_id: Unique customer identifier (e.g., "7590-VHVEG")
        churn_label: "Yes" or "No" - what actually happened
    
    RETURNS:
        Dict with status, message, old_label, new_label, timestamp
    
    EXAMPLE RETURN:
    {
        "status": "success",
        "message": "Label updated for customer 7590-VHVEG",
        "customerID": "7590-VHVEG",
        "old_label": None,
        "new_label": "Yes",
        "timestamp": "2026-04-18T10:30:00"
    }
    """
    # Validate label value
    if churn_label not in ["Yes", "No"]:
        return {
            "status": "error",
            "message": "Churn label must be 'Yes' or 'No'",
            "customerID": customer_id
        }
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Check if customer exists in database
        # Customer should have been added during batch prediction
        row = cursor.execute(
            "SELECT churn_label FROM new_data WHERE customerID = ?",
            (customer_id,)
        ).fetchone()
        
        if not row:
            return {
                "status": "error",
                "message": f"Customer ID '{customer_id}' not found in database",
                "customerID": customer_id
            }
        
        old_label = row['churn_label']
        
        # Update label and set timestamp
        # CURRENT_TIMESTAMP is SQLite's current time
        cursor.execute('''
            UPDATE new_data 
            SET churn_label = ?, label_timestamp = CURRENT_TIMESTAMP
            WHERE customerID = ?
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


# ============================================================================
# BATCH LABEL UPDATE
# ============================================================================

def batch_update_labels(updates: List) -> Dict[str, Any]:
    """
    Update multiple Churn labels at once.
    
    USE CASE:
    After reviewing a batch of customers, update many labels in one API call.
    More efficient than calling update_label() repeatedly.
    
    HOW IT WORKS:
    1. Iterate through each update in the list
    2. Validate each label
    3. Check if customer exists
    4. Update database
    5. Track successes and failures
    6. Commit all changes at once
    
    TRANSACTION SAFETY:
    - All updates happen in a single database transaction
    - If any update fails, others still succeed (no rollback)
    - Each update is independent
    
    ARGS:
        updates: List of LabelUpdateRequest objects
                 Each has customerID and Churn fields
    
    RETURNS:
        Dict with total_updates, successful, failed, and detailed results
    
    EXAMPLE RETURN:
    {
        "status": "completed",
        "total_updates": 2,
        "successful": 1,
        "failed": 1,
        "results": [
            {"customerID": "7590-VHVEG", "status": "success", ...},
            {"customerID": "INVALID", "status": "failed", "message": "Customer ID not found"}
        ]
    }
    """
    results = []
    
    with get_db() as conn:
        cursor = conn.cursor()
        
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
            
            # Check if customer exists
            row = cursor.execute(
                "SELECT churn_label FROM new_data WHERE customerID = ?",
                (customer_id,)
            ).fetchone()
            
            if row:
                old_label = row['churn_label']
                cursor.execute('''
                    UPDATE new_data 
                    SET churn_label = ?, label_timestamp = CURRENT_TIMESTAMP
                    WHERE customerID = ?
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
        
        # Commit all changes at once
        conn.commit()
    
    # Count successes and failures
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    
    return {
        "status": "completed",
        "total_updates": len(updates),
        "successful": len(successful),
        "failed": len(failed),
        "results": results
    }


# ============================================================================
# GET SINGLE LABEL
# ============================================================================

def get_label(customer_id: str) -> Dict[str, Any]:
    """
    Get current Churn label for a customer.
    
    RETURNS:
        - Customer information
        - Actual label (if labeled)
        - Model prediction and probability
        - Timestamps for both
    
    USE CASE:
    Dashboard displays both model prediction and actual label.
    This allows comparison between what model thought vs what actually happened.
    
    ARGS:
        customer_id: Unique customer identifier
    
    RETURNS:
        Dict with customer data or error if not found
    
    EXAMPLE RETURN:
    {
        "status": "success",
        "customerID": "7590-VHVEG",
        "Churn": "Yes",
        "prediction": 1,
        "probability": 0.935,
        "timestamp": "2026-04-16T11:05:53",
        "label_timestamp": "2026-04-18T10:30:00"
    }
    """
    with get_db() as conn:
        row = conn.execute('''
            SELECT customerID, churn_label, prediction, probability, 
                   created_at, label_timestamp
            FROM new_data WHERE customerID = ?
        ''', (customer_id,)).fetchone()
        
        if not row:
            return {
                "status": "error",
                "message": f"Customer ID '{customer_id}' not found",
                "customerID": customer_id
            }
    
    return {
        "status": "success",
        "customerID": row['customerID'],
        "Churn": row['churn_label'],
        "prediction": row['prediction'],
        "probability": row['probability'],
        "timestamp": row['created_at'],
        "label_timestamp": row['label_timestamp']
    }


# ============================================================================
# GET ALL UNLABELED CUSTOMERS
# ============================================================================

def get_unlabeled_data() -> pd.DataFrame:
    """
    Get all records without Churn labels.
    
    USE CASE:
    Dashboard shows which customers need manual labeling.
    Also includes model predictions to guide reviewers.
    
    WHAT'S INCLUDED:
    - All columns from new_data table (customer data + predictions)
    - Only rows where churn_label IS NULL (not labeled yet)
    
    RETURNS:
        DataFrame with unlabeled customers and their predictions
    
    NOTE: 
    Returns ALL columns including:
    - customerID, prediction, probability
    - All features (gender, tenure, MonthlyCharges, etc.)
    - Useful for dashboard to show customer details
    """
    with get_db() as conn:
        df = pd.read_sql_query('''
            SELECT *
            FROM new_data WHERE churn_label IS NULL
        ''', conn)
    
    # Rename column for consistency (database uses customer_id?)
    if not df.empty:
        df = df.rename(columns={'customer_id': 'customerID'})
    return df


# ============================================================================
# GET ALL LABELED CUSTOMERS
# ============================================================================

def get_labeled_data() -> pd.DataFrame:
    """
    Get all records with Churn labels.
    
    USE CASE:
    - Model evaluation (compare predictions vs actual)
    - Retraining data collection
    - Analytics on actual churn patterns
    
    WHAT'S INCLUDED:
    - All columns from new_data table
    - Only rows where churn_label IS NOT NULL (has label)
    
    RETURNS:
        DataFrame with labeled customers including actual churn status
    """
    with get_db() as conn:
        df = pd.read_sql_query('''
            SELECT *
            FROM new_data WHERE churn_label IS NOT NULL
        ''', conn)
    
    if not df.empty:
        df = df.rename(columns={'customer_id': 'customerID'})
    return df


# ============================================================================
# LABEL STATISTICS
# ============================================================================

def get_label_statistics() -> Dict[str, Any]:
    """
    Get statistics about labeled vs unlabeled data.
    
    WHY THESE NUMBERS MATTER:
    - Track labeling progress
    - Know when enough labels for retraining
    - Understand churn rate in labeled data
    
    METRICS RETURNED:
    - total_records: All customers in database
    - labeled_records: Customers with labels
    - unlabeled_records: Customers needing labels
    - churn_yes: Count of "Yes" labels
    - churn_no: Count of "No" labels
    - labeling_progress: Human-readable string
    
    CALCULATIONS:
    unlabeled_records = total_records - labeled_records
    labeling_progress = f"{labeled}/{total} ({percent}%)"
    
    RETURNS:
        Dict with all statistics
    
    EXAMPLE RETURN:
    {
        "status": "success",
        "total_records": 1000,
        "labeled_records": 150,
        "unlabeled_records": 850,
        "churn_yes": 45,
        "churn_no": 105,
        "labeling_progress": "150/1000 (15.0%)"
    }
    """
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
    
    # Format progress as "labeled/total (percent%)"
    progress = f"{labeled}/{total} ({labeled/total*100:.1f}%)" if total > 0 else "0/0 (0%)"
    
    return {
        "status": "success",
        "total_records": total,
        "labeled_records": labeled,
        "unlabeled_records": unlabeled,
        "churn_yes": stats['churn_yes'] or 0,  # Convert None to 0
        "churn_no": stats['churn_no'] or 0,
        "labeling_progress": progress
    }


