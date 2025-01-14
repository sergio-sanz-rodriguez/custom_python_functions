"""
Contains functions for training and testing a PyTorch model.
"""

import os
import logging
import torch
import torchvision
import random
import time
import numpy as np
import pandas as pd
import copy
from datetime import datetime
from typing import Tuple, Dict, Any, List
from tqdm.auto import tqdm 
from torch.utils.tensorboard import SummaryWriter
from IPython.display import clear_output
import matplotlib.pyplot as plt
from pathlib import Path
from timeit import default_timer as timer
from PIL import Image
from torch import GradScaler, autocast
from sklearn.metrics import precision_recall_curve, classification_report
from contextlib import nullcontext


def sec_to_min_sec(seconds):
    
    """
    Converts seconds to a formatted string in minutes and seconds.

    Args:
        seconds (float): The number of seconds to be converted.

    Returns:
        str: A formatted string representing the time in minutes and seconds.
    """

    minutes = seconds // 60
    remaining_seconds = seconds % 60
    return f"{int(minutes)}m{int(remaining_seconds)}s"
    
# Calculate accuracy (a classification metric)
def calculate_accuracy(y_true, y_pred):

    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 0.78
    """

    assert len(y_true) == len(y_pred), "Length of y_true and y_pred must be the same."
    return torch.eq(y_true, y_pred).sum().item() / len(y_true)

def calculate_fpr_at_recall(y_true, y_pred_probs, recall_threshold):

    """Calculates the False Positive Rate (FPR) at a specified recall threshold.

    Args:
        y_true (torch.Tensor): Ground truth labels.
        y_pred_probs (torch.Tensor): Predicted probabilities.
        recall_threshold (float): The recall threshold at which to calculate the FPR.

    Returns:
        float: The calculated FPR at the specified recall threshold.
    """

    # Check if recall_threhold is a valid number
    if not (0 <= recall_threshold <= 1):
        raise ValueError("recall_threshold must be between 0 and 1.")

    # Convert list to tensor if necessary
    if isinstance(y_pred_probs, list):
        y_pred_probs = torch.cat(y_pred_probs)  
    if isinstance(y_true, list):
        y_true = torch.cat(y_true)

    n_classes = y_pred_probs.shape[1]
    fpr_per_class = []

    for class_idx in range(n_classes):
        # Get true binary labels and predicted probabilities for the class
        y_true_bin = (y_true == class_idx).int().detach().numpy()
        y_scores = y_pred_probs[:, class_idx].detach().numpy()

        # Compute precision-recall curve
        _, recall, thresholds = precision_recall_curve(y_true_bin, y_scores)

        # Find the threshold closest to the desired recall
        idx = np.where(recall >= recall_threshold)[0]
        if len(idx) > 0:
            threshold = thresholds[idx[-1]]
            # Calculate false positive rate
            fp = np.sum((y_scores >= threshold) & (y_true_bin == 0))
            tn = np.sum((y_scores < threshold) & (y_true_bin == 0))
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        else:
            fpr = 0  # No threshold meets the recall condition

        fpr_per_class.append(fpr)

    return np.mean(fpr_per_class)  # Average FPR across all classes

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):

    """Saves a PyTorch model to a target directory.

    Args:
        model: A target PyTorch model to save.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.

    Example usage:
        save_model(model=model_0,
                target_dir="models",
                model_name="05_going_modular_tingvgg_model.pth")
    """

    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving best model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

def load_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    
    """Loads a PyTorch model from a target directory.

    Args:
        model: A target PyTorch model to load.
        target_dir: A directory where the model is located.
        model_name: The name of the model to load.
        Should include either ".pth" or ".pt" as the file extension.

    Example usage:
        model = load_model(model=model,
                        target_dir="models",
                        model_name="model.pth")

    Returns:
    The loaded PyTorch model.
    """

    # Create the model directory path
    model_dir_path = Path(target_dir)

    # Create the model path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_path = model_dir_path / model_name

    # Load the model
    print(f"[INFO] Loading model from: {model_path}")
    
    model.load_state_dict(torch.load(model_path, weights_only=True))
    
    return model


def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               recall_threshold: float=None,
               amp: bool=True,
               enable_grad_clipping=True,
               device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"
               ) -> Tuple[float, float, float]:
    
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        recall_threshold: The recall threshold at which to calculate the FPR (between 0 and 1)
        amp: Whether to use mixed precision training (True) or not (False)
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of training loss, training accuracy, and fpr at recall metrics.
        In the form (train_loss, train_accuracy, fpr_at_recall). For example: (0.1112, 0.8743, 0.01123)
    """

    # Put model in train mode
    model.train()
    model.to(device)

    # Initialize the GradScaler for  Automatic Mixed Precision (AMP)
    scaler = GradScaler() if amp else None

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0    
    all_preds = []
    all_labels = []

    # Loop through data loader data batches
    for _, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader)): 
        #, desc=f"Training epoch {epoch_number}..."):

        # Send data to target device
        X, y = X.to(device), y.to(device)

        # Optimize training with amp if available
        if amp:
            with autocast(device_type='cuda', dtype=torch.float16):
                # Forward pass
                y_pred = model(X)
                y_pred = y_pred.contiguous()
                
                # Calculate  and accumulate loss
                loss = loss_fn(y_pred, y)
            
            # Optimizer zero grad
            optimizer.zero_grad()

            # Backward pass and optimizer step with scaled gradients
            scaler.scale(loss).backward()

            # Gradient clipping
            if enable_grad_clipping:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

        else:
            # Forward pass
            y_pred = model(X)
            y_pred = y_pred.contiguous()
            
            # Calculate  and accumulate loss
            loss = loss_fn(y_pred, y)

            # Optimizer zero grad
            optimizer.zero_grad()

            # Loss backward
            loss.backward()

            # Gradient clipping
            if enable_grad_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Optimizer step
            optimizer.step()

        # Calculate and accumulate loss and accuracy across all batches
        train_loss += loss.item()
        #y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        y_pred_class = y_pred.argmax(dim=1)
        train_acc += calculate_accuracy(y, y_pred_class) #(y_pred_class == y).sum().item()/len(y_pred)

        if recall_threshold:
            all_preds.append(torch.softmax(y_pred, dim=1).detach().cpu())
            all_labels.append(y.detach().cpu())

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    # Final FPR calculation
    if recall_threshold and all_preds:
        try:
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            fpr_at_recall = calculate_fpr_at_recall(all_labels, all_preds, recall_threshold)
        except Exception as e:
            logging.error(f"Error calculating final FPR at recall: {e}")
            fpr_at_recall = None
    else:
        fpr_at_recall = None

    return train_loss, train_acc, fpr_at_recall

# This train step function includes gradient accumulation (experimental)
def train_step_v2(model: torch.nn.Module, 
                  dataloader: torch.utils.data.DataLoader, 
                  loss_fn: torch.nn.Module, 
                  optimizer: torch.optim.Optimizer,
                  recall_threshold: float=None,
                  amp: bool=True,
                  enable_grad_clipping=True,
                  accumulation_steps: int = 1,
                  device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"
                  ) -> Tuple[float, float, float]:
    
    """Trains a PyTorch model for a single epoch with gradient accumulation.

    Args:
        model: A PyTorch model to be trained.
        dataloader: A DataLoader instance for the model to be trained on.
        loss_fn: A PyTorch loss function to minimize.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        amp: Whether to use mixed precision training (True) or not (False).
        device: A target device to compute on (e.g. "cuda" or "cpu").
        accumulation_steps: Number of mini-batches to accumulate gradients before an optimizer step.
            If batch size is 64 and accumulation_steps is 4, gradients are accumulated for 256 mini-batches before an optimizer step.

    Returns:
        A tuple of training loss, training accuracy, and fpr at recall metrics.
        In the form (train_loss, train_accuracy, fpr_at_recall). For example: (0.1112, 0.8743, 0.01123)
    """

    # Put model in train mode
    model.train()
    model.to(device)

    # Initialize the GradScaler for Automatic Mixed Precision (AMP)
    scaler = GradScaler() if amp else None

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0    
    all_preds = []
    all_labels = []

    # Loop through data loader data batches
    optimizer.zero_grad()  # Clear gradients before starting
    for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
        
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # Optimize training with amp if available
        if amp:
            with autocast(device_type='cuda', dtype=torch.float16):
                # Forward pass
                y_pred = model(X)
                y_pred = y_pred.contiguous()
                
                # Calculate loss, normalize by accumulation steps
                loss = loss_fn(y_pred, y) / accumulation_steps
            
            # Backward pass with scaled gradients
            scaler.scale(loss).backward()

            # Gradient cliping
            if enable_grad_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        else:
            # Forward pass
            y_pred = model(X)
            y_pred = y_pred.contiguous()
            
            # Calculate loss, normalize by accumulation steps
            loss = loss_fn(y_pred, y) / accumulation_steps

            # Backward pass
            loss.backward()

            # Gradient cliping
            if enable_grad_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Perform optimizer step and clear gradients every accumulation_steps
        if (batch + 1) % accumulation_steps == 0 or (batch + 1) == len(dataloader):
            if amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

        # Accumulate metrics
        train_loss += loss.item() * accumulation_steps  # Scale back to original loss
        y_pred_class = y_pred.argmax(dim=1)
        train_acc += calculate_accuracy(y, y_pred_class) #(y_pred_class == y).sum().item() / len(y_pred)

        if recall_threshold:
            all_preds.append(torch.softmax(y_pred, dim=1).detach().cpu())
            all_labels.append(y.detach().cpu())

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    # Final FPR calculation
    if recall_threshold and all_preds:
        try:
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            fpr_at_recall = calculate_fpr_at_recall(all_labels, all_preds, recall_threshold)
        except Exception as e:
            logging.error(f"Error calculating final FPR at recall: {e}")
            fpr_at_recall = None
    else:
        fpr_at_recall = None

    return train_loss, train_acc, fpr_at_recall

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              recall_threshold: float=None,
              device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"
              ) -> Tuple[float, float, float]:
    
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of training loss, training accuracy, and fpr at recall metrics.
        In the form (train_loss, train_accuracy, fpr_at_recall). For example: (0.1112, 0.8743, 0.01123)
    """
    # Put model in eval mode
    model.eval() 
    model.to(device)

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    all_preds = []
    all_labels = []

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for _, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader), colour='#FF9E2C'):
            #, desc=f"Validating epoch {epoch_number}..."):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred = model(X)
            test_pred = test_pred.contiguous()

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_class = test_pred.argmax(dim=1)
            test_acc += calculate_accuracy(y, test_pred_class) #((test_pred_class == y).sum().item()/len(test_pred))

            if recall_threshold:
                all_preds.append(torch.softmax(test_pred, dim=1).detach().cpu())
                all_labels.append(y.detach().cpu())

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    # Concatenate lists into tensors
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)

    # Calculate FPR at specified recall threshold
    fpr_at_recall = calculate_fpr_at_recall(all_labels, all_preds, recall_threshold) if recall_threshold else None

    return test_loss, test_acc, fpr_at_recall

# Add writer parameter to train()
def train(model: torch.nn.Module,
          target_dir: str,
          model_name: str,
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          recall_threshold: float=0.95,
          scheduler: torch.optim.lr_scheduler=None, #Optional[torch.optim.lr_scheduler.LRScheduler]=None,
          epochs: int=30, 
          plot_curves: bool=True,
          amp: bool=True,
          enable_grad_clipping: bool=True,
          save_best_model: bool=True,
          accumulation_steps: int=1,
          writer: SummaryWriter=False,
          device: torch.device="cuda" if torch.cuda.is_available() else "cpu"
          ) -> pd.DataFrame: #Dict[str, List]:
    
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Stores metrics to specified writer log_dir if present.

    Args:
        model: A PyTorch model to be trained and tested.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include
            either ".pth" or ".pt" as the file extension.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        scheduler: A PyTorch learning rate scheduler to adjust the learning rate during training.
        epochs: An integer indicating how many epochs to train for.        
        plot: A boolean indicating whether to plot the training and testing curves.
        amp: A boolean indicating whether to use Automatic Mixed Precision (AMP) during training.
        save_best_model: A boolean indicating whether to save the best model during training.
        accumulation_steps: An integer indicating how many mini-batches to accumulate gradients before an optimizer step. Default = 1: no accumulation.
        writer: A SummaryWriter() instance to log model results to.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A dataframe of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for 
        each epoch.
        In the form: {
            train_loss: [...],
            train_acc: [...],
            test_loss: [...],
            test_acc: [...],
            train_time: [...],
            test_time: [...],
            lr: [...],
            train_fpr_at_recall: [...],
            test_fpr_at_recall: [...]
            } 
        For example if training for epochs=2: 
            {
            train_loss: [2.0616, 1.0537],
            train_acc: [0.3945, 0.3945],
            test_loss: [1.2641, 1.5706],
            test_acc: [0.3400, 0.2973],
            train_time: [1.1234, 1.5678],
            test_time: [0.4567, 0.7890],
            lr: [0.001, 0.0005],
            train_fpr_at_recall: [0.1234, 0.2345],
            test_fpr_at_recall: [0.3456, 0.4567]
            } 
    """

    # Define colors
    BLACK = '\033[30m'  # Black
    BLUE = '\033[34m'   # Blue
    ORANGE = '\033[38;5;214m'  # Orange (using extended colors)
    GREEN = '\033[32m'  # Green
    RESET = '\033[39m'  # Reset to default color

    # Create empty results dictionary
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "train_time [s]": [],
        "test_time [s]": [],
        "lr": [] 
        }
    if recall_threshold:
        results.update(
            {"train_fpr_at_recall": [],
             "test_fpr_at_recall": []
            }
        )

    # Initialize the best validation loss
    if save_best_model:
        model_name_best = model_name.replace(".", "_best.")
        best_test_loss = float("inf") 

    # Check out accumulation_steps
    assert isinstance(accumulation_steps, int) and accumulation_steps >= 1, "accumulation_steps must be an integer greater than or equal to 1"

    # Enable anomaly detection
    torch.autograd.set_detect_anomaly(True)

    # Loop through training and testing steps for a number of epochs
    for epoch in range(epochs):

        # Perform training step and time it
        print(f"Training epoch {epoch+1}...")
        train_epoch_start_time = time.time()
        if accumulation_steps > 1:
            train_loss, train_acc, train_fpr_at_recall = train_step_v2(
                model=model,
                dataloader=train_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                recall_threshold=recall_threshold,
                amp=amp,
                enable_grad_clipping=enable_grad_clipping,
                accumulation_steps=accumulation_steps,
                device=device)
        else:
            train_loss, train_acc, train_fpr_at_recall = train_step(
                model=model,
                dataloader=train_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                recall_threshold=recall_threshold,
                amp=amp,
                enable_grad_clipping=enable_grad_clipping,
                device=device)
        train_epoch_end_time = time.time()
        train_epoch_time = train_epoch_end_time - train_epoch_start_time

        # Perform test step and time it
        print(f"Validating epoch {epoch+1}...")
        test_epoch_start_time = time.time()
        test_loss, test_acc, test_fpr_at_recall = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            recall_threshold=recall_threshold,                             
            device=device)
        test_epoch_end_time = time.time()
        test_epoch_time = test_epoch_end_time - test_epoch_start_time

        clear_output(wait=True)

        # Print out what's happening
        if scheduler:
            lr = scheduler.get_last_lr()[0]
        else:
            lr = optimizer.param_groups[0]['lr']

        print(
            f"{BLACK}Epoch: {epoch+1} | "
            f"{BLUE}train_loss: {train_loss:.4f} {BLACK}| "
            f"{BLUE}train_acc: {train_acc:.4f} {BLACK}| "
            f"{BLUE}fpr_at_recall: {train_fpr_at_recall if recall_threshold else 0:.4f} {BLACK}| "
            f"{BLUE}train_time: {sec_to_min_sec(train_epoch_time)} {BLACK}| "            
            f"{ORANGE}test_loss: {test_loss:.4f} {BLACK}| "
            f"{ORANGE}test_acc: {test_acc:.4f} {BLACK}| "
            f"{ORANGE}fpr_at_recall: {test_fpr_at_recall if recall_threshold else 0:.4f} {BLACK}| "
            f"{ORANGE}test_time: {sec_to_min_sec(test_epoch_time)} {BLACK}| "
            f"{GREEN}lr: {lr:.10f}"
        )
    
        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        results["train_time [s]"].append(train_epoch_time)
        results["test_time [s]"].append(test_epoch_time)
        results["lr"].append(lr)
        if recall_threshold:
            results["train_fpr_at_recall"].append(train_fpr_at_recall)
            results["test_fpr_at_recall"].append(test_fpr_at_recall)

        # Scheduler step after the optimizer
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # Check whether scheduler is configured for "min" or "max"
                if scheduler.mode == "min":
                    scheduler.step(test_loss)  # Minimize test_loss
                elif scheduler.mode == "max":
                    scheduler.step(test_accuracy)  # Maximize test_accuracy
            else:
                scheduler.step()  # For other schedulers

        # Plot loss and accuracy curves
        n_plots = 3 if recall_threshold else 2
        if plot_curves:
            
            plt.figure(figsize=(20, 6))
            range_epochs = range(1, len(results["train_loss"])+1)

            # Plot loss
            plt.subplot(1, n_plots, 1)
            plt.plot(range_epochs, results["train_loss"], label="train_loss")
            plt.plot(range_epochs, results["test_loss"], label="test_loss")
            plt.title("Loss")
            plt.xlabel("Epochs")
            plt.grid(visible=True, which="both", axis="both")
            plt.legend()

            # Plot accuracy
            plt.subplot(1, n_plots, 2)
            plt.plot(range_epochs, results["train_acc"], label="train_accuracy")
            plt.plot(range_epochs, results["test_acc"], label="test_accuracy")
            plt.title("Accuracy")
            plt.xlabel("Epochs")
            plt.grid(visible=True, which="both", axis="both")
            plt.legend()
            
            # Plot FPR at recall
            if recall_threshold:
                plt.subplot(1, n_plots, 3)
                plt.plot(range_epochs, results["train_fpr_at_recall"], label="train_fpr_at_recall")
                plt.plot(range_epochs, results["test_fpr_at_recall"], label="test_fpr_at_recall")
                plt.title(f"FPR at {recall_threshold * 100}% recall")
                plt.xlabel("Epochs")
                plt.grid(visible=True, which="both", axis="both")
                plt.legend()
            
            plt.show()

        # Update the tqdm progress bar description with the current epoch
        #epoch_bar.set_description(f"Epoch {epoch+1}/{epochs}")
        #epoch_bar.update(1)

        # See if there's a writer, if so, log to it
        if writer:
            # Add results to SummaryWriter
            writer.add_scalars(main_tag="Loss", 
                               tag_scalar_dict={"train_loss": train_loss,
                                                "test_loss": test_loss},
                               global_step=epoch)
            writer.add_scalars(main_tag="Accuracy", 
                               tag_scalar_dict={"train_acc": train_acc,
                                                "test_acc": test_acc}, 
                               global_step=epoch)
            if recall_threshold:
                writer.add_scalars(main_tag="FPR at Recall", 
                                   tag_scalar_dict={"train_fpr_at_recall": train_fpr_at_recall,
                                                    "test_fpr_at_recall": test_fpr_at_recall}, 
                                   global_step=epoch)
        else:
            pass

        # Check if this is the best test loss so far and save the model
        if save_best_model and (test_loss < best_test_loss):
            best_test_loss = test_loss
            save_model(model=model,
                       target_dir=target_dir,
                       model_name=model_name_best)

    # Close the writer
    writer.close() if writer else None

    # Save the model (last epoch)
    save_model(model=model,
               target_dir=target_dir,
               model_name=model_name)

    # Return and save the filled results at the end of the epochs
    name , _ = model_name.rsplit('.', 1)
    csv_file_name = f"{name}.csv"
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(target_dir, csv_file_name), index=False)

    return df_results

def eval_model(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               device: torch.device = "cuda" if torch.cuda.is_available() else "cpu"
               ) -> Dict[str, float]:
    
    """Evaluates a given model on a given dataset.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        dataloader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        device (str, optional): Target device to compute on. Defaults to device.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """

    loss, acc = 0, 0

    model.to(device)  #may not be necessary
    
    model.eval()
    with torch.inference_mode():
        for X, y in dataloader:
            # Send data to the target device
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred, y)
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            acc += calculate_accuracy(y, y_pred_class) #((y_pred_class == y).sum().item()/len(y_pred_class))            
        
        # Scale loss and acc
        loss /= len(dataloader)
        acc /= len(dataloader)

    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}


def predict(model: torch.nn.Module,
            dataloader: torch.utils.data.DataLoader,
            output_type: str="softmax",
            device: torch.device="cuda" if torch.cuda.is_available() else "cpu"
            ) -> torch.Tensor:
    
    """
    Predicts classes for a given dataset using a trained model.

    Args:
        model (torch.nn.Module): A trained PyTorch model.
        dataloader (torch.utils.data.DataLoader): The dataset to predict on.
        output_type (str): The type of output to return. Either "softmax", "logits", or "argmax".
        device (str, optional): Target device to compute on. Defaults to device.

    Returns:
        (list): All of the predicted class labels represented by prediction probabilities (softmax)
    """

    # Check output_max
    valid_output_types = {"softmax", "argmax", "logits"}
    assert output_type in valid_output_types, f"Invalid output_max value: {output_type}. Must be one of {valid_output_types}"

    y_preds = []
    model.eval()
    model.to(device)
    with torch.inference_mode():
        for X, y in tqdm(dataloader, desc="Making predictions"):

            # Send data and targets to target device
            X, y = X.to(device), y.to(device)
            
            # Do the forward pass
            y_logit = model(X)

            if output_type == "softmax":
                y_pred = torch.softmax(y_logit, dim=1)
            elif output_type == "argmax":
                y_pred = torch.argmax(y_logit, dim=1).argmax(dim=1)
            else:
                y_pred = y_logit

            # Put predictions on CPU for evaluation
            y_preds.append(y_pred.cpu())

    # Concatenate list of predictions into a tensor
    return torch.cat(y_preds)


def predict_and_store(model: torch.nn.Module,
                      test_dir: str, 
                      transform: torchvision.transforms, 
                      class_names: List[str], 
                      percent_samples: float = 1.0,
                      seed=42,
                      device: str = "cuda" if torch.cuda.is_available() else "cpu"
                      ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    
    """
    Predicts classes for a given dataset using a trained model and stores the per-sample results in dictionaries.

    Args:
        model (torch.nn.Module): A trained PyTorch model.
        test_dir (str): The directory containing the test images.
        transform (torchvision.transforms): The transformation to apply to the test images.
        class_names (list): A list of class names.
        percent_samples (float, optional): The percentage of samples to predict. Defaults to 1.0.
        seed (int, optional): The random seed for reproducibility. Defaults to 42.
        device (str, optional): Target device to compute on. Defaults to device.

    Returns:
        A list of dictionaries with sample predictions, sample names, prediction probabilities, prediction times, actual labels and prediction classes
        A classification report as a dictionary from sckit-learng.metrics
    """
    # Create a list of test images and checkout existence
    print(f"[INFO] Finding all filepaths ending with '.jpg' in directory: {test_dir}")
    paths = list(Path(test_dir).glob("*/*.jpg"))
    assert len(list(paths)) > 0, f"No files ending with '.jpg' found in this directory: {test_dir}"

    # Number of random images to extract
    num_samples = len(paths)
    num_random_images = int(percent_samples * num_samples)

    # Ensure the number of images to extract is less than or equal to the total number of images
    assert num_random_images <= len(paths), f"Number of images to extract exceeds total images in directory: {len(paths)}"

    # Randomly select a subset of file paths
    torch.manual_seed(seed)
    paths = random.sample(paths, num_random_images)

    # Store predictions and ground-truth labels
    y_true = []
    y_pred = []

    # Create an empty list to store prediction dictionaires
    pred_list = []
    
    # Loop through target paths
    for path in tqdm(paths, total=num_samples):
        
        # Create empty dictionary to store prediction information for each sample
        pred_dict = {}

        # Get the sample path and ground truth class name
        pred_dict["image_path"] = path
        class_name = path.parent.stem
        pred_dict["class_name"] = class_name
        
        # Start the prediction timer
        start_time = timer()
        
        # Open image path
        img = Image.open(path)
        
        # Transform the image, add batch dimension and put image on target device
        transformed_image = transform(img).unsqueeze(0).to(device) 
        
        # Prepare model for inference by sending it to target device and turning on eval() mode
        model.to(device)
        model.eval()
        
        # Get prediction probability, predicition label and prediction class
        with torch.inference_mode():
            pred_logit = model(transformed_image) # perform inference on target sample 
            #pred_logit = pred_logit.contiguous()
            pred_prob = torch.softmax(pred_logit, dim=1) # turn logits into prediction probabilities
            pred_label = torch.argmax(pred_prob, dim=1) # turn prediction probabilities into prediction label
            pred_class = class_names[pred_label.cpu()] # hardcode prediction class to be on CPU

            # Make sure things in the dictionary are on CPU (required for inspecting predictions later on) 
            pred_dict["pred_prob"] = round(pred_prob.unsqueeze(0).max().cpu().item(), 4)
            pred_dict["pred_class"] = pred_class
            
            # End the timer and calculate time per pred
            end_time = timer()
            pred_dict["time_for_pred"] = round(end_time-start_time, 4)

        # Does the pred match the true label?
        pred_dict["correct"] = class_name == pred_class

        # Add the dictionary to the list of preds
        pred_list.append(pred_dict)

        # Append true and predicted label indexes
        y_true.append(class_names.index(class_name))
        y_pred.append(pred_label.cpu().item())

    # Generate the classification report
    classification_report_dict = classification_report(
        y_true=y_true,
        y_pred=y_pred,
        target_names=class_names,
        output_dict=True
    )

    # Return list of prediction dictionaries
    return pred_list, classification_report_dict


def create_writer(experiment_name: str, 
                  model_name: str, 
                  extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter():
    
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)


# Grouping all the functions into a single class
class Trainer:

    """A class to handle training, evaluation, and predictions for a PyTorch model."""

    def __init__(
        self,
        model: torch.nn.Module,
        save_best_model: bool=False,
        mode: str="min",
        device: str="cuda" if torch.cuda.is_available() else "cpu"
        ):
        super().__init__()

        """
        Initialize the class.
        
        Args:
            model (torch.nn.Module): The PyTorch model to handle. Must be instatiated
            save_best_model (bool): Save the best model based on a criterion mode
            mode: criterion mode: "min" (validation loss), "max" (validation accuracy), "all" (all epochs to be saved)
            device (str, optional): Device to use ('cuda' or 'cpu'). If None, it defaults to 'cuda' if available.
        """

        # Check mode
        valid_mode = {"min", "max", "all"}
        assert mode in valid_mode, f"Invalid mode value: {mode}. Must be one of {valid_mode}"

        # Initialize self variables
        self.device = device
        self.model = model
        self.model.to(self.device)
        self.save_best_model = save_best_model
        self.mode = mode
        self.model_best = None
        self.model_epoch = None

        # Create empty results dictionary
        self.results = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "train_fpr_at_recall": [],
            "train_time [s]": [],
            "test_loss": [],
            "test_acc": [],
            "test_fpr_at_recall": [],
            "test_time [s]": [],
            "lr": [],
            } 

    @staticmethod
    def sec_to_min_sec(seconds):
    
        """
        Converts seconds to a formatted string in minutes and seconds.

        Args:
            seconds (float): The number of seconds to be converted.

        Returns:
            str: A formatted string representing the time in minutes and seconds.
        """

        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{int(minutes)}m{int(remaining_seconds)}s"
    
    # Calculate accuracy (a classification metric)
    @staticmethod
    def calculate_accuracy(
        y_true,
        y_pred):

        """Calculates accuracy between truth labels and predictions.

        Args:
            y_true (torch.Tensor): Truth labels for predictions.
            y_pred (torch.Tensor): Predictions to be compared to predictions.

        Returns:
            [torch.float]: Accuracy value between y_true and y_pred, e.g. 0.78
        """

        assert len(y_true) == len(y_pred), "Length of y_true and y_pred must be the same."
        return torch.eq(y_true, y_pred).sum().item() / len(y_true)

    @staticmethod
    def calculate_fpr_at_recall(
        y_true,
        y_pred_probs,
        recall_threshold):

        """Calculates the False Positive Rate (FPR) at a specified recall threshold.

        Args:
            y_true (torch.Tensor): Ground truth labels.
            y_pred_probs (torch.Tensor): Predicted probabilities.
            recall_threshold (float): The recall threshold at which to calculate the FPR.

        Returns:
            float: The calculated FPR at the specified recall threshold.
        """

        # Check if recall_threhold is a valid number
        if not (0 <= recall_threshold <= 1):
            raise ValueError("recall_threshold must be between 0 and 1.")

        # Convert list to tensor if necessary
        if isinstance(y_pred_probs, list):
            y_pred_probs = torch.cat(y_pred_probs)  
        if isinstance(y_true, list):
            y_true = torch.cat(y_true)

        n_classes = y_pred_probs.shape[1]
        fpr_per_class = []

        for class_idx in range(n_classes):
            # Get true binary labels and predicted probabilities for the class
            y_true_bin = (y_true == class_idx).int().detach().numpy()
            y_scores = y_pred_probs[:, class_idx].detach().numpy()

            # Compute precision-recall curve
            _, recall, thresholds = precision_recall_curve(y_true_bin, y_scores)

            # Find the threshold closest to the desired recall
            idx = np.where(recall >= recall_threshold)[0]
            if len(idx) > 0:
                threshold = thresholds[idx[-1]]
                # Calculate false positive rate
                fp = np.sum((y_scores >= threshold) & (y_true_bin == 0))
                tn = np.sum((y_scores < threshold) & (y_true_bin == 0))
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            else:
                fpr = 0  # No threshold meets the recall condition

            fpr_per_class.append(fpr)

        return np.mean(fpr_per_class)  # Average FPR across all classes

    @staticmethod
    def save(
        model: torch.nn.Module,
        target_dir: str,
        model_name: str):

        """Saves a PyTorch model to a target directory.

        Args:
            model: A target PyTorch model to save.
            target_dir: A directory for saving the model to.
            model_name: A filename for the saved model. Should include
            either ".pth" or ".pt" as the file extension.

        Example usage:
            save_model(model=model_0,
                    target_dir="models",
                    model_name="05_going_modular_tingvgg_model.pth")
        """

        # Create target directory
        target_dir_path = Path(target_dir)
        target_dir_path.mkdir(parents=True,
                            exist_ok=True)

        # Create model save path
        assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
        model_save_path = target_dir_path / model_name

        # Save the model state_dict()
        print(f"[INFO] Saving model to: {model_save_path}")
        torch.save(obj=model.state_dict(), f=model_save_path)

    @staticmethod
    def load(
        model: torch.nn.Module,
        target_dir: str,
        model_name: str):
        
        """Loads a PyTorch model from a target directory.

        Args:
            model: A target PyTorch model to load.
            target_dir: A directory where the model is located.
            model_name: The name of the model to load.
            Should include either ".pth" or ".pt" as the file extension.

        Example usage:
            model = load_model(model=model,
                            target_dir="models",
                            model_name="model.pth")

        Returns:
        The loaded PyTorch model.
        """

        # Create the model directory path
        model_dir_path = Path(target_dir)

        # Create the model path
        assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
        model_path = model_dir_path / model_name

        # Load the model
        print(f"[INFO] Loading model from: {model_path}")
        
        model.load_state_dict(torch.load(model_path, weights_only=True))
        
        return model
    
    @staticmethod
    def create_writer(
        experiment_name: str, 
        model_name: str, 
        extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter():

        """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

        log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

        Where timestamp is the current date in YYYY-MM-DD format.

        Args:
            experiment_name (str): Name of experiment.
            model_name (str): Name of model.
            extra (str, optional): Anything extra to add to the directory. Defaults to None.

        Returns:
            torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

        Example usage:
            # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
            writer = create_writer(experiment_name="data_10_percent",
                                model_name="effnetb2",
                                extra="5_epochs")
            # The above is the same as:
            writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
        """

        # Get timestamp of current date (all experiments on certain day live in same folder)
        timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

        if extra:
            # Create log directory path
            log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
        else:
            log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
            
        print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
        return SummaryWriter(log_dir=log_dir)

    def train_step(
        self,
        dataloader: torch.utils.data.DataLoader, 
        loss_fn: torch.nn.Module, 
        optimizer: torch.optim.Optimizer,
        recall_threshold: float=0.95,
        amp: bool=True,
        enable_clipping=True,
        debug_mode: bool=False
        ) -> Tuple[float, float, float]:
        
        """Trains a PyTorch model for a single epoch.

        Turns a target PyTorch model to training mode and then
        runs through all of the required training steps (forward
        pass, loss calculation, optimizer step).

        Args:
            dataloader: A DataLoader instance for the model to be trained on.
            loss_fn: A PyTorch loss function to minimize.
            optimizer: A PyTorch optimizer to help minimize the loss function.
            recall_threshold: The recall threshold at which to calculate the FPR (between 0 and 1).
            amp: Whether to use mixed precision training (True) or not (False).
            enable_clipping: enables clipping on gradients and model outputs.
            debug_mode: A boolean indicating whether the debug model is enabled or not. It may slow down the training process.

        Returns:
            A tuple of training loss, training accuracy, and fpr at recall metrics.
            In the form (train_loss, train_accuracy, fpr_at_recall). For example: (0.1112, 0.8743, 0.01123).
        """

        # Put model in train mode
        self.model.train()
        self.model.to(self.device)

        # Initialize the GradScaler for  Automatic Mixed Precision (AMP)
        scaler = GradScaler() if amp else None

        # Setup train loss and train accuracy values
        train_loss, train_acc = 0, 0    
        all_preds = []
        all_labels = []

        # Loop through data loader data batches
        for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader)): 
            #, desc=f"Training epoch {epoch_number}..."):

            # Send data to target device
            X, y = X.to(self.device), y.to(self.device)

            # Optimize training with amp if available
            if amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    # Forward pass
                    y_pred = self.model(X)
                    y_pred = y_pred.contiguous()
                    
                    # Check if the output has NaN or Inf values
                    if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                        if enable_clipping:
                            print(f"[WARNING] y_pred is NaN or Inf at batch {batch}. Replacing Nans/Infs...")
                            #y_pred = torch.clamp(y_pred, min=-1e5, max=1e5)
                            y_pred = torch.nan_to_num(
                                y_pred,
                                nan=torch.mean(y_pred).item(), 
                                posinf=torch.max(y_pred).item(), 
                                neginf=torch.min(y_pred).item()
                                )
                        else:
                            print(f"[WARNING] y_pred is NaN or Inf at batch {batch}. Skipping batch...")
                            continue

                    # Calculate  and accumulate loss
                    loss = loss_fn(y_pred, y)
                
                    # Check for NaN or Inf in loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"[WARNING] Loss is NaN or Inf at batch {batch}. Skipping batch...")
                        continue

                # Backward pass with scaled gradients
                if debug_mode:
                    # Use anomaly detection
                    with torch.autograd.detect_anomaly():
                        scaler.scale(loss).backward()
                else:
                    scaler.scale(loss).backward()

                # Gradient clipping
                if enable_clipping:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Check gradients for NaN or Inf values
                if debug_mode:
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            if torch.any(torch.isnan(param.grad)) or torch.any(torch.isinf(param.grad)):
                                print(f"[WARNING] NaN or Inf gradient detected in {name} at batch {batch}.")
                                break
                
                # scaler.step() first unscales the gradients of the optimizer's assigned parameters.
                # If these gradients do not contain ``inf``s or ``NaN``s, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)
                scaler.update()

                # Optimizer zero grad
                optimizer.zero_grad()

            else:
                # Forward pass
                y_pred = self.model(X)
                y_pred = y_pred.contiguous()
                
                # Calculate  and accumulate loss
                loss = loss_fn(y_pred, y)

                # Optimizer zero grad
                optimizer.zero_grad()

                # Loss backward
                loss.backward()

                # Gradient clipping
                if enable_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Optimizer step
                optimizer.step()

            # Calculate and accumulate loss and accuracy across all batches
            train_loss += loss.item()
            y_pred_class = y_pred.argmax(dim=1)
            train_acc += self.calculate_accuracy(y, y_pred_class) #(y_pred_class == y).sum().item()/len(y_pred)
            
            # Collect outputs for fpr-at-recall calculation
            all_preds.append(torch.softmax(y_pred, dim=1).detach().cpu())
            all_labels.append(y.detach().cpu())

        # Adjust metrics to get average loss and accuracy per batch 
        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)

        # Final FPR calculation
        try:
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            fpr_at_recall = self.calculate_fpr_at_recall(all_labels, all_preds, recall_threshold)
        except Exception as e:
            logging.error(f"[ERROR] Error calculating final FPR at recall: {e}")
            fpr_at_recall = None


        return train_loss, train_acc, fpr_at_recall

    # This train step function includes gradient accumulation (experimental)
    def train_step_v2(
        self,
        dataloader: torch.utils.data.DataLoader, 
        loss_fn: torch.nn.Module, 
        optimizer: torch.optim.Optimizer,
        recall_threshold: float=0.95,
        amp: bool=True,
        enable_clipping=False,
        accumulation_steps: int = 1,
        debug_mode: bool=False
        ) -> Tuple[float, float, float]:
    
        """Trains a PyTorch model for a single epoch with gradient accumulation.

        Args:
            dataloader: A DataLoader instance for the model to be trained on.
            loss_fn: A PyTorch loss function to minimize.
            optimizer: A PyTorch optimizer to help minimize the loss function.
            recall_threshold: The recall threshold at which to calculate the FPR (between 0 and 1)
            amp: Whether to use mixed precision training (True) or not (False).
            enable_clipping: enables clipping on gradients and model outputs.
            accumulation_steps: Number of mini-batches to accumulate gradients before an optimizer step.
                If batch size is 64 and accumulation_steps is 4, gradients are accumulated for 256 mini-batches before an optimizer step.
            debug_mode: A boolean indicating whether the debug model is enabled or not. It may slow down the training process.

        Returns:
            A tuple of training loss, training accuracy, and fpr at recall metrics.
            In the form (train_loss, train_accuracy, fpr_at_recall). For example: (0.1112, 0.8743, 0.01123)
        """

        # Put model in train mode
        self.model.train()
        self.model.to(self.device)

        # Initialize the GradScaler for Automatic Mixed Precision (AMP)
        scaler = GradScaler() if amp else None

        # Setup train loss and train accuracy values
        train_loss, train_acc = 0, 0    
        all_preds = []
        all_labels = []

        # Loop through data loader data batches
        optimizer.zero_grad()  # Clear gradients before starting
        for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader)):
            
            # Send data to target device
            X, y = X.to(self.device), y.to(self.device)

            # Optimize training with amp if available
            if amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    # Forward pass
                    y_pred = self.model(X)
                    y_pred = y_pred.contiguous()

                    # Check if the output has NaN or Inf values
                    if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                        if enable_clipping:
                            print(f"[WARNING] y_pred is NaN or Inf at batch {batch}. Replacing Nans/Infs...")
                            #y_pred = torch.clamp(y_pred, min=-1e5, max=1e5)
                            y_pred = torch.nan_to_num(
                                y_pred,
                                nan=torch.mean(y_pred).item(), 
                                posinf=torch.max(y_pred).item(), 
                                neginf=torch.min(y_pred).item()
                                )
                        else:
                            print(f"[WARNING] y_pred is NaN or Inf at batch {batch}. Skipping batch...")
                            continue
                    
                    # Calculate loss, normalize by accumulation steps
                    loss = loss_fn(y_pred, y) / accumulation_steps
                
                    # Check for NaN or Inf in loss
                    if debug_mode and (torch.isnan(loss) or torch.isinf(loss)):
                        print(f"[WARNING] Loss is NaN or Inf at batch {batch}. Skipping...")
                        continue

                # Backward pass with scaled gradients
                if debug_mode:
                    # Use anomaly detection
                    with torch.autograd.detect_anomaly():
                        scaler.scale(loss).backward()
                else:
                    scaler.scale(loss).backward()

            else:
                # Forward pass
                y_pred = self.model(X)
                y_pred = y_pred.contiguous()
                
                # Calculate loss, normalize by accumulation steps
                loss = loss_fn(y_pred, y) / accumulation_steps

                # Backward pass
                loss.backward()

                # Gradient cliping
                if enable_clipping:
                    # Apply clipping if needed
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Perform optimizer step and clear gradients every accumulation_steps
            if (batch + 1) % accumulation_steps == 0 or (batch + 1) == len(dataloader):

                if amp:         
                    # Gradient cliping
                    if enable_clipping:
                        # Unscale the gradients before performing any operations on them
                        scaler.unscale_(optimizer)
                        # Apply clipping if needed
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    # Check gradients for NaN or Inf values
                    if debug_mode:
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                if torch.any(torch.isnan(param.grad)) or torch.any(torch.isinf(param.grad)):
                                    print(f"[WARNING] NaN or Inf gradient detected in {name} at batch {batch}")
                                    break

                    # scaler.step() first unscales the gradients of the optimizer's assigned parameters.
                    # If these gradients do not contain ``inf``s or ``NaN``s, optimizer.step() is then called,
                    # otherwise, optimizer.step() is skipped.
                    scaler.step(optimizer)
                    scaler.update()

                else:
                    # Optimizer step
                    optimizer.step()

                # Optimizer zero grad
                optimizer.zero_grad()

            # Accumulate metrics
            train_loss += loss.item() * accumulation_steps  # Scale back to original loss
            y_pred_class = y_pred.argmax(dim=1)
            train_acc += self.calculate_accuracy(y, y_pred_class) #(y_pred_class == y).sum().item() / len(y_pred)

            # Collect outputs for fpr-at-recall calculation
            all_preds.append(torch.softmax(y_pred, dim=1).detach().cpu())
            all_labels.append(y.detach().cpu())

        # Adjust metrics to get average loss and accuracy per batch
        train_loss = train_loss / len(dataloader)
        train_acc = train_acc / len(dataloader)

        # Final FPR calculation
        try:
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            fpr_at_recall = self.calculate_fpr_at_recall(all_labels, all_preds, recall_threshold)
        except Exception as e:
            logging.error(f"[ERROR] Error calculating final FPR at recall: {e}")
            fpr_at_recall = None

        return train_loss, train_acc, fpr_at_recall

    def test_step(
        self,
        dataloader: torch.utils.data.DataLoader, 
        loss_fn: torch.nn.Module,
        recall_threshold: float = 0.95,
        amp: bool = True,
        debug_mode: bool = False,
        enable_clipping: bool = False
    ) -> Tuple[float, float, float]:
        
        """Tests a PyTorch model for a single epoch.

        Args:
            dataloader: A DataLoader instance for the model to be tested on.
            loss_fn: A PyTorch loss function to calculate loss on the test data.
            recall_threshold: The recall threshold at which to calculate the FPR (between 0 and 1).
            amp: Whether to use Automatic Mixed Precision for inference.
            debug_mode: Enables logging for debugging purposes.
            enable_clipping: Enables NaN/Inf value clipping for test predictions.

        Returns:
            A tuple of test loss, test accuracy, and FPR at recall metrics.
        """

        # Put model in eval mode
        self.model.eval() 
        self.model.to(self.device)

        # Setup test loss and test accuracy values
        test_loss, test_acc = 0, 0
        all_preds = []
        all_labels = []

        # Turn on inference context manager
        with torch.inference_mode():
            # Loop through DataLoader batches
            for batch, (X, y) in tqdm(enumerate(dataloader), total=len(dataloader), colour='#FF9E2C'):
                #, desc=f"Validating epoch {epoch_number}..."):
                # Send data to target device
                X, y = X.to(self.device), y.to(self.device)

                # Enable AMP if specified
                with torch.autocast(device_type='cuda', dtype=torch.float16) if amp else nullcontext():
                    test_pred = self.model(X)
                    test_pred = test_pred.contiguous()

                    # Check for NaN/Inf in predictions
                    if torch.isnan(test_pred).any() or torch.isinf(test_pred).any():
                        if enable_clipping:
                            print(f"[WARNING] Predictions contain NaN/Inf at batch {batch}. Applying clipping...")
                            test_pred = torch.nan_to_num(
                                test_pred,
                                nan=torch.mean(test_pred).item(),
                                posinf=torch.max(test_pred).item(),
                                neginf=torch.min(test_pred).item()
                            )
                        else:
                            print(f"[WARNING] Predictions contain NaN/Inf at batch {batch}. Skipping batch...")
                            continue

                    # Calculate and accumulate loss
                    loss = loss_fn(test_pred, y)
                    test_loss += loss.item()

                    # Debug NaN/Inf loss
                    if debug_mode and (torch.isnan(loss) or torch.isinf(loss)):
                        print(f"[WARNING] Loss is NaN/Inf at batch {batch}. Skipping...")
                        continue

                # Calculate and accumulate accuracy
                test_pred_class = test_pred.argmax(dim=1)
                test_acc += self.calculate_accuracy(y, test_pred_class) #((test_pred_class == y).sum().item()/len(test_pred))

                # Collect outputs for fpr-at-recall calculation
                all_preds.append(torch.softmax(test_pred, dim=1).detach().cpu())
                all_labels.append(y.detach().cpu())

        # Adjust metrics to get average loss and accuracy per batch 
        test_loss = test_loss / len(dataloader)
        test_acc = test_acc / len(dataloader)

        # Final FPR calculation
        try:
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            fpr_at_recall = self.calculate_fpr_at_recall(all_labels, all_preds, recall_threshold)
        except Exception as e:
            logging.error(f"[ERROR] Error calculating final FPR at recall: {e}")
            fpr_at_recall = None

        return test_loss, test_acc, fpr_at_recall
    
    def print_config(
            self,
            dataloader,
            recall_threshold,
            epochs,
            plot_curves,
            amp,
            enable_clipping,
            accumulation_steps,
            writer):
        
        """
        Prints the configuration of the training process.
        """

        print(f"[INFO] Device: {self.device}")
        print(f"[INFO] Epochs: {epochs}")
        print(f"[INFO] Batch size: {dataloader.batch_size}")
        print(f"[INFO] Accumulation steps: {accumulation_steps}")
        print(f"[INFO] Recall threshold: {recall_threshold}")
        print(f"[INFO] Plot curvers: {plot_curves}")
        print(f"[INFO] Automatic Mixed Precision (AMP): {amp}")
        print(f"[INFO] Enable clipping: {enable_clipping}")
        print(f"[INFO] Enable writer: {writer}")
        print("")
    
    def update_results(
            self,
            epoch,
            train_loss,
            train_acc,
            recall_threshold,
            train_fpr_at_recall,
            train_epoch_time,
            test_loss,
            test_acc,
            test_fpr_at_recall,
            test_epoch_time,
            lr,
            writer
            ):

        # Define colors
        BLACK = '\033[30m'  # Black
        BLUE = '\033[34m'   # Blue
        ORANGE = '\033[38;5;214m'  # Orange (using extended colors)
        GREEN = '\033[32m'  # Green
        RESET = '\033[39m'  # Reset to default color

        print(
            f"{BLACK}Epoch: {epoch+1} | "
            f"{BLUE}train_loss: {train_loss:.4f} {BLACK}| "
            f"{BLUE}train_acc: {train_acc:.4f} {BLACK}| "
            f"{BLUE}fpr_at_recall: {train_fpr_at_recall:.4f} {BLACK}| "
            f"{BLUE}train_time: {self.sec_to_min_sec(train_epoch_time)} {BLACK}| "            
            f"{ORANGE}test_loss: {test_loss:.4f} {BLACK}| "
            f"{ORANGE}test_acc: {test_acc:.4f} {BLACK}| "
            f"{ORANGE}fpr_at_recall: {test_fpr_at_recall:.4f} {BLACK}| "
            f"{ORANGE}test_time: {self.sec_to_min_sec(test_epoch_time)} {BLACK}| "
            f"{GREEN}lr: {lr:.10f}"
            )
        
        # Update results dictionary
        self.results["epoch"].append(epoch+1)
        self.results["train_loss"].append(train_loss)
        self.results["train_acc"].append(train_acc)
        self.results["test_loss"].append(test_loss)
        self.results["test_acc"].append(test_acc)
        self.results["train_time [s]"].append(train_epoch_time)
        self.results["test_time [s]"].append(test_epoch_time)
        self.results["lr"].append(lr)
        self.results["train_fpr_at_recall"].append(train_fpr_at_recall)
        self.results["test_fpr_at_recall"].append(test_fpr_at_recall)
        
        # See if there's a writer, if so, log to it
        if writer:
            # Add results to SummaryWriter
            writer.add_scalars(
                main_tag="Loss", 
                tag_scalar_dict={"train_loss": train_loss,
                                 "test_loss": test_loss},
                global_step=epoch)
            writer.add_scalars(
                main_tag="Accuracy", 
                tag_scalar_dict={"train_acc": train_acc,
                                 "test_acc": test_acc},
                global_step=epoch)
            writer.add_scalars(
                main_tag=f"FPR at {recall_threshold * 100}% recall", 
                tag_scalar_dict={"train_fpr_at_recall": train_fpr_at_recall,
                                    "test_fpr_at_recall": test_fpr_at_recall}, 
                global_step=epoch)
        else:
            pass
        
    
    def plot_curves(
            self,
            recall_threshold):
        
        """
        Plots training and test loss, accuracy, and fpr-at-recall curves.
        """

        n_plots = 3
        plt.figure(figsize=(20, 6))
        range_epochs = range(1, len(self.results["train_loss"])+1)

        # Plot loss
        plt.subplot(1, n_plots, 1)
        plt.plot(range_epochs, self.results["train_loss"], label="train_loss")
        plt.plot(range_epochs, self.results["test_loss"], label="test_loss")
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.grid(visible=True, which="both", axis="both")
        plt.legend()

        # Plot accuracy
        plt.subplot(1, n_plots, 2)
        plt.plot(range_epochs, self.results["train_acc"], label="train_accuracy")
        plt.plot(range_epochs, self.results["test_acc"], label="test_accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.grid(visible=True, which="both", axis="both")
        plt.legend()
                
        # Plot FPR at recall
        plt.subplot(1, n_plots, 3)
        plt.plot(range_epochs, self.results["train_fpr_at_recall"], label="train_fpr_at_recall")
        plt.plot(range_epochs, self.results["test_fpr_at_recall"], label="test_fpr_at_recall")
        plt.title(f"FPR at {recall_threshold * 100}% recall")
        plt.xlabel("Epochs")
        plt.grid(visible=True, which="both", axis="both")
        plt.legend()
                
        plt.show()


    def fit(
        self,
        target_dir: str,
        model_name: str,
        train_dataloader: torch.utils.data.DataLoader, 
        test_dataloader: torch.utils.data.DataLoader, 
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        recall_threshold: float=0.95,
        scheduler: torch.optim.lr_scheduler=None,
        epochs: int=30, 
        plot_curves: bool=True,
        amp: bool=True,
        enable_clipping: bool=True,
        accumulation_steps: int=1,
        debug_mode: bool=False,
        writer: SummaryWriter=False,
        ) -> pd.DataFrame:

        """Trains and tests a PyTorch model.

        Passes a target PyTorch models through train_step() and test_step()
        functions for a number of epochs, training and testing the model
        in the same epoch loop.

        Calculates, prints and stores evaluation metrics throughout.

        Stores metrics to specified writer log_dir if present.

        Args:
            target_dir: A directory for saving the model to.
            model_name: A filename for the saved model. Should include
                either ".pth" or ".pt" as the file extension.
            train_dataloader: A DataLoader instance for the model to be trained on.
            test_dataloader: A DataLoader instance for the model to be tested on.
            optimizer: A PyTorch optimizer to help minimize the loss function.
            loss_fn: A PyTorch loss function to calculate loss on both datasets.
            recall_threshold: The recall threshold at which to calculate the FPR (between 0 and 1). 
            scheduler: A PyTorch learning rate scheduler to adjust the learning rate during training.
            epochs: An integer indicating how many epochs to train for.        
            plot: A boolean indicating whether to plot the training and testing curves.
            amp: A boolean indicating whether to use Automatic Mixed Precision (AMP) during training.
            enable_clipping: enables clipping on gradients and model outputs.
            accumulation_steps: An integer indicating how many mini-batches to accumulate gradients before an optimizer step. Default = 1: no accumulation.
            debug_mode: A boolean indicating whether the debug model is enabled or not. It may slow down the training process.
            writer: A SummaryWriter() instance to log model results to.

        Returns:
            A dataframe of training and testing loss as well as training and
            testing accuracy metrics. Each metric has a value in a list for 
            each epoch.
            In the form: {
                epoch: [...],
                train_loss: [...],
                train_acc: [...],
                test_loss: [...],
                test_acc: [...],
                train_time: [...],
                test_time: [...],
                lr: [...],
                train_fpr_at_recall: [...],
                test_fpr_at_recall: [...]
                } 
            For example if training for epochs=2: 
                {
                epoch: [ 1, 2].
                train_loss: [2.0616, 1.0537],
                train_acc: [0.3945, 0.3945],
                test_loss: [1.2641, 1.5706],
                test_acc: [0.3400, 0.2973],
                train_time: [1.1234, 1.5678],
                test_time: [0.4567, 0.7890],
                lr: [0.001, 0.0005],
                train_fpr_at_recall: [0.1234, 0.2345],
                test_fpr_at_recall: [0.3456, 0.4567]
                } 
        """

        # Check out recall_threshold
        assert isinstance(recall_threshold, float) and 0.0 <= recall_threshold <= 1.0, "recall_threshold must be a float between 0.0 and 1.0"
        
        # Check out accumulation_steps
        assert isinstance(accumulation_steps, int) and accumulation_steps >= 1, "accumulation_steps must be an integer greater than or equal to 1"

        # Check out epochs
        assert isinstance(epochs, int) and epochs >= 1, "epochs must be an integer greater than or equal to 1"

        # Print configuration parameters
        self.print_config(
            dataloader=train_dataloader,
            recall_threshold=recall_threshold,
            epochs=epochs,
            plot_curves=plot_curves,
            amp=amp,
            enable_clipping=enable_clipping,
            accumulation_steps=accumulation_steps,
            writer=writer
            )
        
        # Initialize the best model and model_epoch list based on the specified mode.
        if self.save_best_model:
            if (self.mode == "min") or (self.mode == "max"):
                self.model_best = copy.deepcopy(self.model)                            
                self.model_best.to(self.device)
                model_name_best = model_name.replace(".", f"_best.")            
                best_test_loss = float("inf") 
                best_test_acc = 0.0
            else:
                self.model_epoch = []
                for k in range(epochs):
                    self.model_epoch.append(copy.deepcopy(self.model))
                    self.model_epoch[k].to(self.device)

        # Loop through training and testing steps for a number of epochs
        for epoch in range(epochs):

            # Perform training step and time it
            print(f"Training epoch {epoch+1}...")
            train_epoch_start_time = time.time()
            if accumulation_steps > 1:
                train_loss, train_acc, train_fpr_at_recall = self.train_step_v2(
                    dataloader=train_dataloader,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    recall_threshold=recall_threshold,
                    amp=amp,
                    enable_clipping=enable_clipping,
                    accumulation_steps=accumulation_steps,
                    debug_mode=debug_mode
                    )
            else:
                train_loss, train_acc, train_fpr_at_recall = self.train_step(
                    dataloader=train_dataloader,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    recall_threshold=recall_threshold,
                    amp=amp,
                    enable_clipping=enable_clipping,
                    debug_mode=debug_mode
                    )
            train_epoch_end_time = time.time()
            train_epoch_time = train_epoch_end_time - train_epoch_start_time

            # Perform test step and time it
            print(f"Validating epoch {epoch+1}...")
            test_epoch_start_time = time.time()
            test_loss, test_acc, test_fpr_at_recall = self.test_step(
                dataloader=test_dataloader,
                loss_fn=loss_fn,
                recall_threshold=recall_threshold,
                amp=amp,
                enable_clipping=enable_clipping,
                debug_mode=debug_mode
                )
            test_epoch_end_time = time.time()
            test_epoch_time = test_epoch_end_time - test_epoch_start_time

            clear_output(wait=True)

            # Print results
            self.update_results(
                epoch=epoch,
                train_loss=train_loss,
                train_acc=train_acc,
                recall_threshold=recall_threshold,
                train_fpr_at_recall=train_fpr_at_recall,
                train_epoch_time=train_epoch_time,
                test_loss=test_loss,
                test_acc=test_acc,
                test_fpr_at_recall=test_fpr_at_recall,
                test_epoch_time=test_epoch_time,
                lr=scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]['lr'],
                writer=writer
            )

            # Scheduler step after the optimizer
            if scheduler:
                scheduler.step()

            # Plot loss and accuracy curves
            if plot_curves:
                self.plot_curves(recall_threshold=recall_threshold)

            # Update the tqdm progress bar description with the current epoch
            #epoch_bar.set_description(f"Epoch {epoch+1}/{epochs}")
            #epoch_bar.update(1)
            
            # Update the best model and model_epoch list based on the specified mode.
            if self.save_best_model:                
                if (self.mode == "min") and (test_loss < best_test_loss):
                    best_test_loss = test_loss
                    self.save(
                        model=self.model,
                        target_dir=target_dir,
                        model_name=model_name_best)
                    self.model_best.load_state_dict(self.model.state_dict())
                elif (self.mode == "max") and (test_acc > best_test_acc):
                    best_test_acc = test_acc
                    self.save(
                        model=self.model,
                        target_dir=target_dir,
                        model_name=model_name_best)
                    self.model_best.load_state_dict(self.model.state_dict())
                elif self.mode == "all":
                    self.save(
                        model=self.model,
                        target_dir=target_dir,
                        model_name=model_name.replace(".", f"_epoch{epoch+1}."))
                    self.model_epoch[k].load_state_dict(self.model.state_dict())

        # Close the writer
        writer.close() if writer else None

        # Save the model (last epoch)
        self.save(
            model=self.model,
            target_dir=target_dir,
            model_name=model_name)

        # Return and save the filled results at the end of the epochs
        name , _ = model_name.rsplit('.', 1)
        csv_file_name = f"{name}.csv"
        df_results = pd.DataFrame(self.results)
        df_results.to_csv(os.path.join(target_dir, csv_file_name), index=False)

        return df_results

    def predict(
        self,
        dataloader: torch.utils.data.DataLoader,
        model_state: str="last",
        output_type: str="softmax",        
        ) -> torch.Tensor:

        """
        Predicts classes for a given dataset using a trained model.

        Args:
            model_state: specifies the model to use for making predictions
            dataloader (torch.utils.data.DataLoader): The dataset to predict on.
            output_type (str): The type of output to return. Either "softmax", "logits", or "argmax".            

        Returns:
            (list): All of the predicted class labels represented by prediction probabilities (softmax)
        """
 
        # Check model to use
        valid_models = {"last", "best"}
        assert model_state in valid_models or isinstance(model_state, int), f"Invalid model value: {model_state}. Must be one of {valid_models} or an integer."

        if model_state == "last":
            model = self.model
        elif model_state == "best":
            if self.model_best is None:
                print(f"[INFO] Model best not found, using last model for prediction.")
                model = self.model
            else:
                model = self.model_best
        elif isinstance(model_state, int):
            if self.model_epoch is None:
                print(f"[INFO] Model epoch {model_state} not found, using last model for prediction.")
                model = self.model
            else:
                if model_state > len(self.model_epoch):
                    print(f"[INFO] Model epoch {model_state} not found, using last model for prediction.")
                    model = self.model
                else:
                    model = self.model_epoch[model_state-1]            


        # Check output_max
        valid_output_types = {"softmax", "argmax", "logits"}
        assert output_type in valid_output_types, f"Invalid output_max value: {output_type}. Must be one of {valid_output_types}"

        y_preds = []
        model.eval()
        model.to(self.device)
        with torch.inference_mode():
            for X, y in tqdm(dataloader, desc="Making predictions"):

                # Send data and targets to target device
                X, y = X.to(self.device), y.to(self.device)
                
                # Do the forward pass
                y_logit = model(X)

                if output_type == "softmax":
                    y_pred = torch.softmax(y_logit, dim=1)
                elif output_type == "argmax":
                    y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
                else:
                    y_pred = y_logit

                # Put predictions on CPU for evaluation
                y_preds.append(y_pred.cpu())

        # Concatenate list of predictions into a tensor
        return torch.cat(y_preds)

    def predict_and_store(
        self,
        test_dir: str, 
        transform: torchvision.transforms, 
        class_names: List[str], 
        model_state: str="last",
        percent_samples: float=1.0,
        seed=42,        
        ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        
        """
        Predicts classes for a given dataset using a trained model and stores the per-sample results in dictionaries.

        Args:
            model_state: specifies the model to use for making predictions. "last" (default), "best", an integer
            test_dir (str): The directory containing the test images.
            transform (torchvision.transforms): The transformation to apply to the test images.
            class_names (list): A list of class names.
            percent_samples (float, optional): The percentage of samples to predict. Defaults to 1.0.
            seed (int, optional): The random seed for reproducibility. Defaults to 42.

        Returns:
            A list of dictionaries with sample predictions, sample names, prediction probabilities, prediction times, actual labels and prediction classes
            A classification report as a dictionary from sckit-learng.metrics
        """

        # Check model to use
        valid_models = {"last", "best"}
        assert model_state in valid_models or isinstance(model_state, int), f"Invalid model value: {model_state}. Must be one of {valid_models} or an integer."


        if model_state == "last":
            model = self.model
        elif model_state == "best":
            if self.model_best is None:
                print(f"[INFO] Model best not found, using last model for prediction.")
                model = self.model
            else:
                model = self.model_best
        elif isinstance(model_state, int):
            if self.model_epoch is None:
                print(f"[INFO] Model epoch {model_state} not found, using last model for prediction.")
                model = self.model
            else:
                if model_state > len(self.model_epoch):
                    print(f"[INFO] Model epoch {model_state} not found, using last model for prediction.")
                    model = self.model
                else:
                    model = self.model_epoch[model_state-1]

        # Create a list of test images and checkout existence
        print(f"[INFO] Finding all filepaths ending with '.jpg' in directory: {test_dir}")
        paths = list(Path(test_dir).glob("*/*.jpg"))
        assert len(list(paths)) > 0, f"No files ending with '.jpg' found in this directory: {test_dir}"

        # Number of random images to extract
        num_samples = len(paths)
        num_random_images = int(percent_samples * num_samples)

        # Ensure the number of images to extract is less than or equal to the total number of images
        assert num_random_images <= len(paths), f"Number of images to extract exceeds total images in directory: {len(paths)}"

        # Randomly select a subset of file paths
        torch.manual_seed(seed)
        paths = random.sample(paths, num_random_images)

        # Store predictions and ground-truth labels
        y_true = []
        y_pred = []

        # Create an empty list to store prediction dictionaires
        pred_list = []
        
        # Loop through target paths
        for path in tqdm(paths, total=num_samples):
            
            # Create empty dictionary to store prediction information for each sample
            pred_dict = {}

            # Get the sample path and ground truth class name
            pred_dict["image_path"] = path
            class_name = path.parent.stem
            pred_dict["class_name"] = class_name
            
            # Start the prediction timer
            start_time = timer()
            
            # Open image path
            img = Image.open(path)
            
            # Transform the image, add batch dimension and put image on target device
            transformed_image = transform(img).unsqueeze(0).to(self.device) 
            
            # Prepare model for inference by sending it to target device and turning on eval() mode
            model.to(self.device)
            model.eval()
            
            # Get prediction probability, predicition label and prediction class
            with torch.inference_mode():
                pred_logit = model(transformed_image) # perform inference on target sample 
                #pred_logit = pred_logit.contiguous()
                pred_prob = torch.softmax(pred_logit, dim=1) # turn logits into prediction probabilities
                pred_label = torch.argmax(pred_prob, dim=1) # turn prediction probabilities into prediction label
                pred_class = class_names[pred_label.cpu()] # hardcode prediction class to be on CPU

                # Make sure things in the dictionary are on CPU (required for inspecting predictions later on) 
                pred_dict["pred_prob"] = round(pred_prob.unsqueeze(0).max().cpu().item(), 4)
                pred_dict["pred_class"] = pred_class
                
                # End the timer and calculate time per pred
                end_time = timer()
                pred_dict["time_for_pred"] = round(end_time-start_time, 4)

            # Does the pred match the true label?
            pred_dict["correct"] = class_name == pred_class

            # Add the dictionary to the list of preds
            pred_list.append(pred_dict)

            # Append true and predicted label indexes
            y_true.append(class_names.index(class_name))
            y_pred.append(pred_label.cpu().item())

        # Generate the classification report
        classification_report_dict = classification_report(
            y_true=y_true,
            y_pred=y_pred,
            target_names=class_names,
            output_dict=True
            )

        # Return list of prediction dictionaries
        return pred_list, classification_report_dict