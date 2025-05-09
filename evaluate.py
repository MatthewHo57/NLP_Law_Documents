from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np
import json
import matplotlib.pyplot as plt
import os
from datasets import load_dataset
from main import LegalDocumentProcessor

def evaluate_on_cuad_sample(sample_size=50):
    """
    Evaluate the model on a sample from the CUAD dataset
    """
    print(f"Loading CUAD dataset...")
    try:
        dataset = load_dataset("theatticusproject/cuad")
        train_data = dataset["train"]
        
        print(f"Sampling {sample_size} documents from CUAD...")
        # Get a sample of the dataset
        indices = np.random.choice(len(train_data), size=sample_size, replace=False)
        sample = [train_data[i] for i in indices]
        
        processor = LegalDocumentProcessor()
        
        results = {
            "termination": {"true_positives": 0, "false_positives": 0, "false_negatives": 0},
            "confidentiality": {"true_positives": 0, "false_positives": 0, "false_negatives": 0},
            "liability": {"true_positives": 0, "false_positives": 0, "false_negatives": 0},
            "payment": {"true_positives": 0, "false_positives": 0, "false_negatives": 0},
            "governing_law": {"true_positives": 0, "false_positives": 0, "false_negatives": 0}
        }
        
        # Process documents and compare with annotations
        for i, doc in enumerate(sample):
            print(f"Processing document {i+1}/{sample_size}...")
            
            # Get document text
            text = doc["full_text"]
            
            # Get CUAD annotations for key clauses
            annotations = {}
            for clause_type in results.keys():
                # Map our clause types to CUAD's clause types (this is a simplified mapping)
                cuad_clause_mapping = {
                    "termination": "Termination",
                    "confidentiality": "Confidentiality",
                    "liability": "Limitation of Liability",
                    "payment": "Payment Terms",
                    "governing_law": "Governing Law"
                }
                
                # Get relevant answers for this clause type
                cuad_clause = cuad_clause_mapping[clause_type]
                relevant_answers = []
                
                for q_idx, question in enumerate(doc["questions"]):
                    if cuad_clause in question:
                        answer = doc["answers"][q_idx]
                        if answer.get("text", []) and answer["text"][0].strip():
                            relevant_answers.append(answer["text"][0])
                
                annotations[clause_type] = relevant_answers
            
            # Process document with our model
            sentences = processor.segment_sentences(text)
            predicted_clauses = processor.extract_key_clauses(sentences)
            
            # Compare predictions with annotations
            for clause_type in results.keys():
                predicted = [clause[0] for clause in predicted_clauses.get(clause_type, [])]
                actual = annotations.get(clause_type, [])
                
                # For each predicted clause, check if it's in the actual clauses
                for pred in predicted:
                    found = False
                    for act in actual:
                        if pred in act or act in pred:
                            found = True
                            break
                    
                    if found:
                        results[clause_type]["true_positives"] += 1
                    else:
                        results[clause_type]["false_positives"] += 1
                
                # For each actual clause, check if it's not in the predicted clauses
                for act in actual:
                    found = False
                    for pred in predicted:
                        if pred in act or act in pred:
                            found = True
                            break
                    
                    if not found:
                        results[clause_type]["false_negatives"] += 1
        
        # Calculate metrics
        metrics = {}
        for clause_type, counts in results.items():
            tp = counts["true_positives"]
            fp = counts["false_positives"]
            fn = counts["false_negatives"]
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[clause_type] = {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        
        # Calculate overall metrics
        total_tp = sum(counts["true_positives"] for counts in results.values())
        total_fp = sum(counts["false_positives"] for counts in results.values())
        total_fn = sum(counts["false_negatives"] for counts in results.values())
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        metrics["overall"] = {
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1
        }
        
        # Print metrics
        print("\nEvaluation Results:")
        print("------------------")
        for clause_type, clause_metrics in metrics.items():
            print(f"{clause_type.title()}: Precision={clause_metrics['precision']:.2f}, Recall={clause_metrics['recall']:.2f}, F1={clause_metrics['f1']:.2f}")
        
        # Visualize metrics
        plot_metrics(metrics)
        
        return metrics
        
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return None

def plot_metrics(metrics):
    """
    Plot evaluation metrics
    """
    clause_types = [ct for ct in metrics.keys() if ct != "overall"]
    
    metrics_data = {
        "precision": [metrics[ct]["precision"] for ct in clause_types] + [metrics["overall"]["precision"]],
        "recall": [metrics[ct]["recall"] for ct in clause_types] + [metrics["overall"]["recall"]],
        "f1": [metrics[ct]["f1"] for ct in clause_types] + [metrics["overall"]["f1"]]
    }
    
    x = np.arange(len(clause_types) + 1)
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(x - width, metrics_data["precision"], width, label="Precision")
    ax.bar(x, metrics_data["recall"], width, label="Recall")
    ax.bar(x + width, metrics_data["f1"], width, label="F1 Score")
    
    ax.set_ylabel('Score')
    ax.set_title('Evaluation Metrics by Clause Type')
    ax.set_xticks(x)
    ax.set_xticklabels([ct.title() for ct in clause_types] + ["Overall"])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig("evaluation_metrics.png")
    plt.close()
    
    print("Metrics visualization saved to evaluation_metrics.png")

def main():
    print("Starting evaluation...")
    metrics = evaluate_on_cuad_sample()
    
    if metrics:
        # Save metrics to file
        with open("evaluation_results.json", "w") as f:
            json.dump(metrics, f, indent=4)
        print("Evaluation results saved to evaluation_results.json")

if __name__ == "__main__":
    main()