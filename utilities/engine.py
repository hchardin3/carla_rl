import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import os
from tqdm.auto import tqdm
from typing import Dict, List, Tuple



def training_step(model: nn.Module, 
                  train_dataloader: DataLoader, 
                  loss_function: nn.Module, 
                  optimizer: optim.Optimizer,
                  regression_problem: bool = False) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    model.train()
    train_loss, train_acc = 0., 0.
    
    for X, y in train_dataloader:
        X, y = X.to(model.device), y.to(model.device)

        optimizer.zero_grad()
        y_pred = model(X)
        loss = loss_function(y_pred, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if regression_problem:
            train_acc += T.mean(T.abs(y_pred - y)).item()
        else:
            class_pred = T.argmax(y_pred, dim=1)
            train_acc += (class_pred == y).sum().item()

    num_samples = len(train_dataloader.dataset)
    train_loss /= len(train_dataloader)
    train_acc /= num_samples

    return train_loss, train_acc



def testing_step(model: T.nn.Module, 
              test_dataloader: DataLoader, 
              loss_function: nn.Module,
              regression_problem: bool = False) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    model.eval()

    test_loss, test_acc = 0., 0.

    with T.inference_mode():
        for batch, (X, y) in enumerate(test_dataloader):
            X, y = X.to(model.device), y.to(model.device)

            y_pred = model(X)
            loss = loss_function(y_pred, y)

            test_loss += loss.item() 
            if regression_problem:
                test_acc += T.mean(T.abs(y_pred - y)).item()
            else:
                class_pred = T.argmax(y_pred, dim=1)
                test_acc += (class_pred == y).sum().item()
    
    test_loss = test_loss/len(test_dataloader)
    test_acc = test_acc/len(test_dataloader)

    return test_loss, test_acc


def train_model(model: nn.Module, 
          train_dataloader: DataLoader, 
          test_dataloader: DataLoader, 
          optimizer: optim.Optimizer,
          loss_function: nn.Module,
          epochs: int = 10,
          writer: SummaryWriter = None,
          printing: bool = True,
          regression_problem: bool = False) -> Dict[str, List[float]]:
    """
    Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    writer: A SummaryWriter() instance to log model results to.
    printing: A boolean to specify wether the metrics should be printed during each epoch or not.

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]} 
    For example if training for epochs=2: 
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]} 
    """

    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = training_step(model, train_dataloader, loss_function, optimizer, regression_problem=regression_problem)
        test_loss, test_acc = testing_step(model, test_dataloader, loss_function, regression_problem=regression_problem)

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if printing:
            print(f"Epoch: {epoch+1} | train_loss: {train_loss} | train_acc: {train_acc} | test_loss: {test_loss} | test_acc: {test_acc} |")
    
        if writer is not None:
            # Add results to SummaryWriter
            writer.add_scalars(main_tag="Loss", 
                               tag_scalar_dict={"train_loss": train_loss,
                                                "test_loss": test_loss},
                               global_step=epoch)
            writer.add_scalars(main_tag="Accuracy", 
                               tag_scalar_dict={"train_acc": train_acc,
                                                "test_acc": test_acc}, 
                               global_step=epoch)

    # Close the writer
    if writer is not None:
        writer.close()

    return results
