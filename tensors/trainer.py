import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import List, Tuple

from .core import get_data_loader, split_dataset
from .metrics import ( f1_score, compute_brier_score, compute_ece, compute_pr_auc, compute_roc_auc, 
                        compute_multiclass_pr_auc, compute_multiclass_roc_auc, print_confusion_matrix)


######################################################################
######################################################################


class AtomicTrainer:

    _optimizer = torch.optim.Adam


    def _compute_metrics(self, *args, **kwargs):
        raise NotImplementedError
   

    def _handle_loss_computation(self, outputs, labels):
        raise NotImplementedError


    def _pred_fn(self, outputs):
        raise NotImplementedError


    def _prob_fn(self, outputs):
        raise NotImplementedError


    def _print_metrics(self, metrics: dict, stage: str):
        print(f"\n=== {stage} Metrics ===")
        print(f"Loss: {metrics['loss']:.4f}", end='\t')
        print(f"A*: {metrics['a*']*100:.2f}%", end='\t')
        print(f"F1: {metrics['f1']*100:.2f}%", end='\t')
        print(f"Brier: {metrics['brier']*100:.4f}%", end='\t')
        print(f"ECE: {metrics['ece']*100:.4f}%", end='\t')
        print(f"PR AUC: {metrics['pr_auc']*100:.2f}%", end='\t')
        print(f"ROC AUC: {metrics['roc_auc']*100:.2f}%", end='\t')
        print(f"Sharpness: {metrics['sharp']*100:.2f}%\n")


    def _run_machine(self, model, data_loader, dsc="Running", optimizer=None):
        total_loss = 0
        all_preds, all_labels, all_probs = [], [], []
        for features, labels in tqdm(data_loader, desc=f"{dsc} Model"):
        # for features, labels in data_loader:
            
            # Only during Training
            if optimizer is not None:
                optimizer.zero_grad() 
            #####
                        
            outputs = model(features)
            self._validate_outputs(outputs)

            # Handle loss computation based on loss_type
            loss, labels = self._handle_loss_computation(outputs, labels)

            # Only during Training
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            #####

            total_loss += loss.item() * len(labels)
            preds = self._pred_fn(outputs)
            probs = self._prob_fn(outputs)

            all_preds.append(preds.detach().numpy())
            all_labels.append(labels.detach().numpy())
            all_probs.append(probs.detach().numpy())

        return total_loss, np.concatenate(all_labels, axis=0), np.concatenate(all_probs, axis=0), np.concatenate(all_preds, axis=0)


    def _test_model(self, model, test_data):
        model.eval()
        with torch.no_grad():
            return self._run_machine(model, test_data, "Testing")


    def _train_model(self, model, train_data, optimizer):
        model.train()
        return self._run_machine(model, train_data, "Training", optimizer)


    def _validate_model(self, model, val_data):
        model.eval()
        with torch.no_grad():
            return self._run_machine(model, val_data, "Validating")


    def _validate_outputs(self, outputs: torch.Tensor):
        """
        Validate model output shape based on loss type.

        Args:
            outputs: torch.Tensor, model outputs (logits for CE, logits or probabilities for BCE).
        """
        raise NotImplementedError


    def dojo(self, model, dataset, *, epochs: int = 50, patience: int = 3):
        """
        Train, validate, and test with early stopping.

        Args:
            model: torch.nn.Module, the model to train.
            dataset: Dataset, data to be split
            epochs: int, number of training epochs.
            patience: int, number of epochs to wait before early stopping.
        """
        
        optimizer = self._optimizer(model.parameters(), lr=0.001)
        best_val_loss = float("inf")
        patience_counter = 0

        train_data, val_data, test_data = [get_data_loader(ds) for ds in split_dataset(dataset)]

        for epoch in range(epochs):
            train_loss, train_labels, train_probs, train_preds = self._train_model(model, train_data, optimizer)
            val_loss, val_labels, val_probs, val_preds = self._validate_model(model, val_data)

            # Compute metrics
            train_metrics = self._compute_metrics(train_loss, train_labels, train_probs, train_preds, len(dataset))
            val_metrics = self._compute_metrics(val_loss, val_labels, val_probs, val_preds, len(dataset))
            
            # # # Print metrics
            
            self._print_metrics(train_metrics, "Training")
            self._print_metrics(val_metrics, "Validating")

            # Save model on validation improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                print_confusion_matrix(self.class_labels, val_labels, val_preds, epoch)
                model._save(val_metrics)
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter > patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        test_loss, test_labels, test_probs, test_preds = self._test_model(model, test_data)
        test_metrics = self._compute_metrics(test_loss, test_labels, test_probs, test_preds, len(dataset))

        self._print_metrics(test_metrics, "Testing")
        # print_confusion_matrix(self.class_labels, test_labels, test_preds)
    

        
######################################################################
######################################################################
            

class BinaryTrainer(AtomicTrainer):

    _loss_function = torch.nn.BCEWithLogitsLoss 


    def __init__(self, *, class_labels: List[str]=["a", "b"], class_weights: torch.Tensor = None):
        """
        Initialize the trainer with modular loss function support.

        Args:
            class_labels: List[str], (required for CE, optional for BCE to specify binary output format).
            class_weights: Optional torch.Tensor of class weights for imbalanced datasets (shape: (num_classes,) for CE, (2,) for BCE).

        """
        super().__init__()

        self.class_labels = class_labels
        self.criterion = self._loss_function(pos_weight=class_weights)


    def _compute_metrics(self, total_loss: float, all_labels: np.ndarray, all_probs: np.ndarray, 
                            all_preds: np.ndarray, dataset_length: int) -> dict:
        """
        Compute and validate metrics for predictions.
        """
        # Compute existing metrics
        return {
            "loss": total_loss / dataset_length if dataset_length > 0 else float('inf'),
            "a*": np.sum(all_preds == all_labels) / len(all_labels) if len(all_labels) > 0 else 0.0,
            "f1": f1_score(all_labels, all_preds, average='macro'),
            "brier": compute_brier_score(all_probs, all_labels),
            "ece": compute_ece(all_probs, all_labels),
            "pr_auc": compute_pr_auc(all_probs, all_labels),
            "roc_auc": compute_roc_auc(all_probs, all_labels),
            "sharp": float(np.mean(all_probs[(all_preds == all_labels)]))
        }


    def _handle_loss_computation(self, outputs, labels):
        # BCE expects float labels (0.0 or 1.0) and 1D or (batch_size, 1) outputs
        outputs = outputs.squeeze()  # Remove extra dimension if present
        labels = labels.float()  # Convert labels to float
        loss = self.criterion(outputs, labels)
        return loss, labels


    def _pred_fn(self, outputs):
        return (torch.sigmoid(outputs) > 0.5).float()  # Binary: sigmoid + threshold
        

    def _prob_fn(self, outputs):
        return torch.sigmoid(outputs)
        

    def _validate_outputs(self, outputs: torch.Tensor):
        """
        Validate model output shape based on loss type.
        """
        if outputs.ndim == 1 or (outputs.ndim == 2 and outputs.shape[-1] == 1):
            pass
        else:
            raise ValueError(f"BCE expects 1D outputs or 2D with 1 column, got shape {outputs.shape}")


######################################################################
######################################################################



class ClassifyTrainer(AtomicTrainer):
    
    _loss_function = torch.nn.CrossEntropyLoss 


    def __init__(self, *, class_labels: List[str], class_weights: torch.Tensor = None):
        """
        Initialize the trainer with modular loss function support.

        Args:
            class_labels: List[str], (required for CE, optional for BCE to specify binary output format).
            class_weights: Optional torch.Tensor of class weights for imbalanced datasets (shape: (num_classes,) for CE, (2,) for BCE).
        """
        super().__init__()
       
        self.class_labels = class_labels
        self.criterion = self._loss_function(weight=class_weights)



    def _compute_metrics(self, total_loss: float, all_labels: np.ndarray, all_probs: np.ndarray, 
                            all_preds: np.ndarray, dataset_length: int) -> dict:
        """
        Compute and validate metrics for predictions.
        """
        # Compute existing metrics
        return {
            "loss": total_loss / dataset_length if dataset_length > 0 else float('inf'),
            "a*": np.sum(all_preds == all_labels) / len(all_labels) if len(all_labels) > 0 else 0.0,
            "f1": f1_score(all_labels, all_preds, average='macro'),
            "brier": compute_brier_score(all_probs, all_labels),
            "ece": compute_ece(all_probs, all_labels),
            "pr_auc": compute_multiclass_pr_auc(all_probs, all_labels),
            "roc_auc": compute_multiclass_roc_auc(all_probs, all_labels),
            "sharp": float(np.mean(all_probs[(all_preds == all_labels)]))
        }
        

    def _handle_loss_computation(self, outputs, labels):
        loss =  self.criterion(outputs, labels)
        return loss, labels 


    def _pred_fn(self, outputs):
        return torch.argmax(outputs, dim=1)  # Multiclass: argmax
        

    def _prob_fn(self, outputs):
        return torch.softmax(outputs, dim=1)   


    def _validate_outputs(self, outputs: torch.Tensor):
        """
        Validate model output shape based on loss type.
        """
        if outputs.shape[-1] != len(self.class_labels):
            raise ValueError(f"Model output has {outputs.shape[-1]} classes, expected {len(self.class_labels)}")



######################################################################
######################################################################


