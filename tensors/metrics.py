import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score


######################################################################
######################################################################


def compute_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the Brier score for binary or multi-class classification.

    Use: Measures the mean squared difference between predicted probabilities and true labels.
         Lower Brier score indicates better probability calibration. Useful for assessing
         prediction reliability in betting or classification tasks.

    Measures: Brier Score = (1/N) * sum((p_i - y_i)^2), where p_i is the predicted probability
              for the true class, and y_i is 1 (true) or 0 (false). For multi-class, computes
              the average score across all classes.
    """
    # Validate inputs
    if probs.shape[0] != labels.shape[0]:
        raise ValueError("probs and labels must have the same number of samples.")
    if not np.all(labels.astype(int) == labels):
        raise ValueError("Labels must be integers.")
    
    # Binary classification (1D probs or 2D with one column)
    if probs.ndim == 1 or (probs.ndim == 2 and probs.shape[-1] == 1):
        if probs.ndim == 2:
            probs = probs.squeeze(-1)  # Convert [n_samples, 1] to [n_samples]
        if not np.all((labels == 0) | (labels == 1)):
            raise ValueError("Binary labels must be 0 or 1.")
        # Compute Brier score: mean squared error between probs and labels
        return float(np.mean((probs - labels) ** 2))
    
    # Multi-class classification (2D probs)
    if probs.ndim != 2:
        raise ValueError("probs must be 1D for binary or 2D for multi-class.")
    if not np.allclose(probs.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Multi-class probs must sum to 1 across classes.")
    if np.any(labels < 0) or np.any(labels >= probs.shape[1]):
        raise ValueError("Labels must be in range [0, n_classes-1].")

    # One-hot encode labels for multi-class
    n_samples, n_classes = probs.shape
    one_hot_labels = np.zeros((n_samples, n_classes))
    one_hot_labels[np.arange(n_samples), labels] = 1

    # Compute Brier score: mean squared error between probs and one-hot labels
    return float(np.mean((probs - one_hot_labels) ** 2))



######################################################################
######################################################################


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 20) -> float:
    """
    Compute the Expected Calibration Error (ECE) for binary or multi-class classification.

    Use: Measures how well predicted probabilities align with actual outcomes. Lower ECE
         indicates better calibration, critical for betting where probability reliability matters.

    Measures: ECE = sum((|b_i| / N) * |acc(b_i) - conf(b_i)|), where b_i is a bin of predictions,
              acc(b_i) is the accuracy in the bin, conf(b_i) is the average predicted probability,
              and |b_i| is the number of samples in the bin. For multi-class, computes ECE per class
              and averages (macro).
    """
    # Validate inputs
    if probs.shape[0] != labels.shape[0]:
        raise ValueError("probs and labels must have the same number of samples.")
    if not np.all(labels.astype(int) == labels):
        raise ValueError("Labels must be integers.")

    # Binary classification (1D probs or 2D with one column)
    if probs.ndim == 1 or (probs.ndim == 2 and probs.shape[-1] == 1):
        if probs.ndim == 2:
            probs = probs.squeeze(-1)  # Convert [n_samples, 1] to [n_samples]
        if not np.all((labels == 0) | (labels == 1)):
            raise ValueError("Binary labels must be 0 or 1.")
        # Bin probabilities
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(probs, bins, right=True).clip(1, n_bins) - 1
        ece = 0.0
        n_samples = len(probs)
        
        for i in range(n_bins):
            bin_mask = bin_indices == i
            if bin_mask.sum() == 0:
                continue
            bin_probs = probs[bin_mask]
            bin_labels = labels[bin_mask]
            bin_conf = np.mean(bin_probs)  # Average predicted probability
            bin_acc = np.mean(bin_labels)  # Average accuracy
            ece += (bin_mask.sum() / n_samples) * np.abs(bin_conf - bin_acc)
        
        return float(ece)

    # Multi-class classification (2D probs)
    if probs.ndim != 2:
        raise ValueError("probs must be 1D for binary or 2D for multi-class.")
    if not np.allclose(probs.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Multi-class probs must sum to 1 across classes.")
    if np.any(labels < 0) or np.any(labels >= probs.shape[1]):
        raise ValueError("Labels must be in range [0, n_classes-1].")

    n_classes = probs.shape[1]
    ece_per_class = []
    
    for c in range(n_classes):
        # Treat class c as positive, others as negative
        binary_labels = (labels == c).astype(int)
        class_probs = probs[:, c]
        # Compute ECE for this class
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(class_probs, bins, right=True).clip(1, n_bins) - 1
        ece = 0.0
        n_samples = len(class_probs)
        
        for i in range(n_bins):
            bin_mask = bin_indices == i
            if bin_mask.sum() == 0:
                continue
            bin_probs = class_probs[bin_mask]
            bin_labels = binary_labels[bin_mask]
            bin_conf = np.mean(bin_probs)
            bin_acc = np.mean(bin_labels)
            ece += (bin_mask.sum() / n_samples) * np.abs(bin_conf - bin_acc)
        
        ece_per_class.append(ece)
    
    return float(np.mean(ece_per_class))

######################################################################
######################################################################




def compute_roc_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the Receiver Operating Characteristic Area Under Curve (ROC-AUC) for binary classification.
    
    Use: Measures the model's ability to distinguish between positive (e.g., hit) and negative (e.g., no-hit)
         classes across all classification thresholds. Higher ROC-AUC (closer to 1) indicates better
         discrimination. Useful for comparing models and evaluating performance on imbalanced datasets.
    
    Measures: Area under the ROC curve, which plots True Positive Rate (TPR, recall) against False
              Positive Rate (FPR) at various thresholds. ROC-AUC = 1 is perfect, 0.5 is random guessing.
    
    Loss Function: Typically used with Binary Cross Entropy (BCE) loss for binary classification.
    """
    if probs.ndim == 2:
        probs = probs.squeeze(-1)  # Convert [n_samples, 1] to [n_samples]
    # Ensure inputs are valid
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("Labels must be binary (0 or 1).")
    if probs.shape != labels.shape:
        raise ValueError("Probs and labels must have the same shape.")
    
    # Compute ROC-AUC using sklearn
    try:
        return roc_auc_score(labels, probs)
    except ValueError as e:
        # Handle cases with only one class or invalid inputs
        print(f"ROC-AUC computation failed: {e}")
        return np.nan



def compute_multiclass_roc_auc(probs: np.ndarray, labels: np.ndarray, average: str = "macro") -> float:
    """
    Compute the Receiver Operating Characteristic Area Under Curve (ROC-AUC) for multi-class classification.
    
    Use: Measures the model's ability to distinguish each class (e.g., home run) from all others (one-vs-rest)
         across all thresholds. Useful for task-specific evaluation (e.g., home run vs. not) in multi-class
         settings, such as baseball outcome prediction. Higher ROC-AUC indicates better discrimination.
    
    Measures: Area under the ROC curve for each class (one-vs-rest), averaged (macro or weighted).
              Macro averages treat all classes equally; weighted averages account for class frequency.
              ROC-AUC = 1 is perfect, 0.5 is random guessing.
    
    Loss Function: Typically used with Cross Entropy loss for multi-class classification.
    
    Args:
        probs: np.ndarray of shape (n_samples, n_classes) with predicted probabilities for each class.
        labels: np.ndarray of shape (n_samples,) with true labels (integers from 0 to n_classes-1).
        average: str, either "macro" (equal weight per class) or "weighted" (weight by class frequency).
    
    Returns:
        float: Multi-class ROC-AUC score (macro or weighted average).
    """
    # Ensure inputs are valid
    if probs.shape[0] != labels.shape[0]:
        raise ValueError("Probs and labels must have the same number of samples.")
    if probs.shape[1] < 2:
        raise ValueError("Probs must have at least 2 classes.")
    if not np.allclose(probs.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Probs must sum to 1 across classes for each sample.")
    if not np.all(labels.astype(int) == labels):
        raise ValueError("Labels must be integers.")
    
    # Compute ROC-AUC for each class (one-vs-rest)
    try:
        return roc_auc_score(labels, probs, multi_class="ovr", average=average)
    except ValueError as e:
        # Handle cases with only one class or invalid inputs
        print(f"Multi-class ROC-AUC computation failed: {e}")
        return np.nan



def compute_pr_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the Precision-Recall Area Under Curve (PR-AUC) for binary classification.
    
    Use: Measures the trade-off between precision and recall for the positive class (e.g., hit) across
         all thresholds. Particularly useful for imbalanced datasets where the positive class is rare
         (e.g., home runs). Higher PR-AUC indicates better precision-recall performance.
    
    Measures: Area under the Precision-Recall curve, which plots precision (TP / (TP + FP)) against
              recall (TP / (TP + FN)) at various thresholds. PR-AUC = 1 is perfect; baseline depends
              on class imbalance (e.g., positive class frequency for random guessing).
    
    Loss Function: Typically used with Binary Cross Entropy (BCE) loss for binary classification.
    
    Args:
        probs: np.ndarray of shape (n_samples,) with predicted probabilities for the positive class.
        labels: np.ndarray of shape (n_samples,) with true binary labels (0 or 1).
    
    Returns:
        float: PR-AUC score.
    """
    # Ensure inputs are valid
    if probs.ndim == 2:
        probs = probs.squeeze(-1)  # Convert [n_samples, 1] to [n_samples]
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("Labels must be binary (0 or 1).")
    if probs.shape != labels.shape:
        raise ValueError("Probs and labels must have the same shape.")
    
    # Compute Precision-Recall curve and AUC
    try:
        precision, recall, _ = precision_recall_curve(labels, probs)
        return auc(recall, precision)
    except ValueError as e:
        # Handle cases with only one class or invalid inputs
        print(f"PR-AUC computation failed: {e}")
        return np.nan





def compute_multiclass_pr_auc(probs: np.ndarray, labels: np.ndarray, average: str = "macro") -> float:
    """
    Compute the Precision-Recall Area Under Curve (PR-AUC) for multi-class classification.
    
    Use: Measures the trade-off between precision and recall for each class (e.g., home run vs. not)
         in a multi-class setting. Ideal for imbalanced classes (e.g., triples or home runs in baseball
         outcomes). Higher PR-AUC indicates better precision-recall performance per class.
    
    Measures: Area under the Precision-Recall curve for each class (one-vs-rest), averaged (macro or
              weighted). Macro averages treat all classes equally; weighted averages account for class
              frequency. PR-AUC = 1 is perfect; baseline varies by class frequency.
    
    Loss Function: Typically used with Cross Entropy loss for multi-class classification.
    
    Args:
        probs: np.ndarray of shape (n_samples, n_classes) with predicted probabilities for each class.
        labels: np.ndarray of shape (n_samples,) with true labels (integers from 0 to n_classes-1).
        average: str, either "macro" (equal weight per class) or "weighted" (weight by class frequency).
    
    Returns:
        float: Multi-class PR-AUC score (macro or weighted average).
    """
    # Ensure inputs are valid
    if probs.shape[0] != labels.shape[0]:
        raise ValueError("Probs and labels must have the same number of samples.")
    if probs.shape[1] < 2:
        raise ValueError("Probs must have at least 2 classes.")
    if not np.allclose(probs.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Probs must sum to 1 across classes for each sample.")
    if not np.all(labels.astype(int) == labels):
        raise ValueError("Labels must be integers.")
    
    # Compute PR-AUC for each class (one-vs-rest)
    n_classes = probs.shape[1]
    pr_aucs = []
    class_counts = np.bincount(labels, minlength=n_classes)
    
    for c in range(n_classes):
        if class_counts[c] == 0:
            continue  # Skip classes with no true instances
        binary_labels = (labels == c).astype(int)
        precision, recall, _ = precision_recall_curve(binary_labels, probs[:, c])
        pr_aucs.append(auc(recall, precision))
    
    if not pr_aucs:
        return np.nan
    
    # Average PR-AUCs
    if average == "macro":
        return np.mean(pr_aucs)
    elif average == "weighted":
        weights = class_counts[class_counts > 0] / np.sum(class_counts)
        return np.average(pr_aucs, weights=weights)
    else:
        raise ValueError("Average must be 'macro' or 'weighted'.")




def compute_log_loss(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-7) -> float:
    """
    Compute the Log Loss (Binary Cross Entropy) for binary classification.
    
    Use: Measures the accuracy of predicted probabilities for binary outcomes (e.g., hit vs. no-hit).
         Penalizes confident incorrect predictions heavily. Lower Log Loss indicates better
         probability estimates. Useful for evaluating model performance and calibration in betting.
    
    Measures: Average negative log-likelihood of true labels given predicted probabilities.
              Log Loss = 0 is perfect; higher values indicate worse performance.
    
    Loss Function: This is the Binary Cross Entropy (BCE) loss, used directly as the training
                   objective for binary classification.
    
    Args:
        probs: np.ndarray of shape (n_samples,) with predicted probabilities for the positive class.
        labels: np.ndarray of shape (n_samples,) with true binary labels (0 or 1).
        eps: float, small value to clip probabilities to avoid log(0).
    
    Returns:
        float: Log Loss score.
    """
    # Ensure inputs are valid
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("Labels must be binary (0 or 1).")
    if probs.shape != labels.shape:
        raise ValueError("Probs and labels must have the same shape.")
    
    # Clip probabilities to avoid log(0)
    probs = np.clip(probs, eps, 1 - eps)
    
    # Compute Log Loss: -[y * log(p) + (1-y) * log(1-p)]
    log_loss = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
    
    return float(log_loss)




def compute_multiclass_log_loss(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-7) -> float:
    """
    Compute the Log Loss (Cross Entropy) for multi-class classification.
    
    Use: Measures the accuracy of predicted probabilities for multi-class outcomes (e.g., fly out,
         ground out, ..., home run). Penalizes confident incorrect predictions heavily. Lower Log
         Loss indicates better probability estimates. Useful for evaluating calibration in multi-class
         settings like baseball outcome prediction.
    
    Measures: Average negative log-likelihood of true labels given predicted probabilities.
              Log Loss = 0 is perfect; higher values indicate worse performance.
    
    Loss Function: This is the Cross Entropy loss, used directly as the training objective for
                   multi-class classification.
    
    Args:
        probs: np.ndarray of shape (n_samples, n_classes) with predicted probabilities for each class.
        labels: np.ndarray of shape (n_samples,) with true labels (integers from 0 to n_classes-1).
        eps: float, small value to clip probabilities to avoid log(0).
    
    Returns:
        float: Multi-class Log Loss score.
    """
    # Ensure inputs are valid
    if probs.shape[0] != labels.shape[0]:
        raise ValueError("Probs and labels must have the same number of samples.")
    if probs.shape[1] < 2:
        raise ValueError("Probs must have at least 2 classes.")
    if not np.allclose(probs.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Probs must sum to 1 across classes for each sample.")
    if not np.all(labels.astype(int) == labels):
        raise ValueError("Labels must be integers.")
    if np.any(labels < 0) or np.any(labels >= probs.shape[1]):
        raise ValueError("Labels must be in range [0, n_classes-1].")
    
    # Clip probabilities to avoid log(0)
    probs = np.clip(probs, eps, 1 - eps)
    
    # Compute Log Loss: -sum(y_i * log(p_i)) for true class i
    n_samples = probs.shape[0]
    log_loss = -np.mean(np.log(probs[np.arange(n_samples), labels]))
    
    return float(log_loss)


def compute_mse_loss(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Mean Squared Error for binary classification.
    Args:
        probs: np.ndarray of shape (n_samples,) with predicted probabilities.
        labels: np.ndarray of shape (n_samples,) with true binary labels (0 or 1).
    Returns:
        float: MSE score.
    """
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("Labels must be binary (0 or 1).")
    if probs.shape != labels.shape:
        raise ValueError("Probs and labels must have the same shape.")
    return float(np.mean((labels - probs) ** 2))




def print_confusion_matrix(class_labels: list, all_labels: np.ndarray, all_preds: np.ndarray, epoch: int = None):
        """
        Print confusion matrix.
        """
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_labels))))
        if cm.shape != (len(class_labels), len(class_labels)):
            print(f"Error: Confusion matrix shape {cm.shape} does not match num_classes {len(class_labels)}")
            return
        cm_df = pd.DataFrame(cm, index=list(range(len(class_labels))), columns=list(range(len(class_labels))))
        prefix = f"Epoch {epoch+1}" if epoch is not None else "Test"
        print(f"Confusion Matrix ({prefix}):\n{cm_df}")
        print()

        precision = precision_score(all_labels, all_preds, average=None)
        recall = recall_score(all_labels, all_preds, average=None)
        f1 = f1_score(all_labels, all_preds, average=None)
        for i, name in enumerate(class_labels):
            try:
                print(f"{name}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1={f1[i]:.3f}\n")
            except IndexError:
                pass

