import os
import argparse
import numpy as np
from sklearn.metrics import classification_report, f1_score

def load_data(directory):
    """
    Loads test_predictions.npy and test_targets.npy from the given directory.
    """
    pred_path = os.path.join(directory, 'test_predictions.npy')
    target_path = os.path.join(directory, 'test_targets.npy')

    if not os.path.exists(pred_path) or not os.path.exists(target_path):
        raise FileNotFoundError(f"Could not find .npy files in {directory}")

    print(f"Loading files from: {directory}")
    predictions = np.load(pred_path)
    targets = np.load(target_path)
    
    return predictions, targets

def calculate_category_f1(directory):
    # 1. Load Data
    preds_raw, targets = load_data(directory)

    # 2. Process Predictions
    # If predictions are probabilities/logits (2D array), take the argmax to get class indices.
    # If they are already class indices (1D array), use them directly.
    if preds_raw.ndim > 1:
        preds = np.argmax(preds_raw, axis=1)
    else:
        preds = preds_raw

    # 3. Define Class Labels
    # Based on the "STMconformer_balanced.py" and "corpora_categories" context.
    # NOTE: You should verify the exact order of these labels in your STMconformer_balanced.py 
    # if the output below shows generic "Class 0", "Class 1", etc.
    # Common categories in this domain (Music/Speech) or specific Corpora names:
    
    # Attempting to infer labels from unique target values if strict naming isn't available
    unique_labels = np.unique(targets)
    target_names = [f"Class {i}" for i in unique_labels] 
    
    # Placeholder: If you know the specific corpora names from STMconformer_balanced.py, 
    # uncomment and update the line below:
    # target_names = ['Music', 'Speech', 'Environmental'] # Example 

    print("-" * 60)
    print("Category-specific Performance Report")
    print("-" * 60)

    # 4. Calculate and Display Results
    # Using classification_report to mimic the detailed breakdown style
    report = classification_report(targets, preds, target_names=target_names, digits=4)
    print(report)

    # 5. Calculate specific F1 scores vector (for programmatic use if needed)
    f1_per_class = f1_score(targets, preds, average=None)
    
    print("-" * 60)
    print("Raw F1 Scores per Category:")
    for i, score in enumerate(f1_per_class):
        label = target_names[i] if i < len(target_names) else f"Class {i}"
        print(f"{label}: {score:.4f}")
    print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate category-specific F1 scores from model predictions.")
    
    # Defaulting to the path provided in the prompt, but allowing override via CLI
    default_path = "model/STM/Kanformer_enhanced_corpora_categories/standard/ckpt/2026-01-17_23-14/"
    
    parser.add_argument(
        "--dir", 
        type=str, 
        default=default_path,
        help="Path to the directory containing test_predictions.npy and test_targets.npy"
    )

    args = parser.parse_args()
    
    try:
        calculate_category_f1(args.dir)
    except Exception as e:
        print(f"Error: {e}")