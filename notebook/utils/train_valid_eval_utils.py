"""
This code is generated by Ridvan Salih KUZU @DLR
LAST EDITED:  14.09.2021
ABOUT SCRIPT:
It defines some utility functions required in training and evaluation of model
"""

import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import pdb
import sklearn


def confusion_matrix_figure(conf_matrix, labels):
    """
    THIS FUNCTION GENERATES A FIGURE FROM THE CONFUSION MATRIX

    :param conf_matrix: confusion matrix
    :param labels: labels in confusion matrixs

    :return: matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    cm = conf_matrix / (conf_matrix.sum(1) + 1e-12)
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va="center", ha="center", size="xx-large")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predictions", fontsize=18)
    ax.set_ylabel("Actuals", fontsize=18)
    ax.set_title("Confusion Matrix", fontsize=18)
    return fig


def metrics(y_true, y_pred):
    """
    THIS FUNCTION DETERMINES THE EVALUATION METRICS OF THE MODEL

    :param y_true: ground-truth labels
    :param y_pred: predicted labels

    :return: dictionary of Accuracy, Kappa, F1, Recall, and Precision
    """
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    kappa = sklearn.metrics.cohen_kappa_score(y_true, y_pred)
    f1_micro = sklearn.metrics.f1_score(y_true, y_pred, average="micro", zero_division=0)
    f1_macro = sklearn.metrics.f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = sklearn.metrics.f1_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_micro = sklearn.metrics.recall_score(y_true, y_pred, average="micro")
    recall_macro = sklearn.metrics.recall_score(y_true, y_pred, average="macro")
    recall_weighted = sklearn.metrics.recall_score(y_true, y_pred, average="weighted")
    precision_micro = sklearn.metrics.precision_score(
        y_true, y_pred, average="micro", zero_division=0
    )
    precision_macro = sklearn.metrics.precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted = sklearn.metrics.precision_score(y_true, y_pred, average="weighted")

    return dict(
        accuracy=accuracy,
        kappa=kappa,
        f1_micro=f1_micro,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        recall_micro=recall_micro,
        recall_macro=recall_macro,
        recall_weighted=recall_weighted,
        precision_micro=precision_micro,
        precision_macro=precision_macro,
        precision_weighted=precision_weighted,
    )


def train_epoch(model, optimizer, criterion, dataloader, device="cpu"):
    """
    THIS FUNCTION ITERATES A SINGLE EPOCH FOR TRAINING

    :param model: torch model for training
    :param optimizer: torch training optimizer
    :param criterion: torch objective for loss calculation
    :param dataloader: training data loader
    :param device: where to run the epoch

    :return: loss
    """
    model.train()
    losses = list()
    with tqdm(enumerate(dataloader), total=len(dataloader), position=0, leave=True) as iterator:
        for idx, batch in iterator:
            optimizer.zero_grad()
            x, y_true, _, _ = batch
            loss = criterion(model.forward(x.to(device)), y_true.to(device))
            loss.backward()
            optimizer.step()
            iterator.set_description(f"train loss={loss:.2f}")
            losses.append(loss)
    return torch.stack(losses)


def validation_epoch(model, criterion, dataloader, device="cpu"):
    """
    THIS FUNCTION ITERATES A SINGLE EPOCH FOR VALIDATION

    :param model: torch model for validation
    :param criterion: torch objective for loss calculation
    :param dataloader: validation data loader
    :param device: where to run the epoch

    :return: loss, y_true, y_pred, y_score, field_id
    """
    model.eval()
    with torch.no_grad():
        losses = list()
        y_true_list = list()
        y_pred_list = list()
        y_score_list = list()
        field_ids_list = list()
        with tqdm(enumerate(dataloader), total=len(dataloader), position=0, leave=True) as iterator:
            for idx, batch in iterator:
                x, y_true, _, field_id = batch
                logprobabilities = model.forward(x.to(device))
                loss = criterion(logprobabilities, y_true.to(device))
                iterator.set_description(f"valid loss={loss:.2f}")
                losses.append(loss)
                y_true_list.append(y_true)
                y_pred_list.append(logprobabilities.argmax(-1))
                y_score_list.append(logprobabilities.exp())
                field_ids_list.append(field_id)
        return (
            torch.stack(losses),
            torch.cat(y_true_list),
            torch.cat(y_pred_list),
            torch.cat(y_score_list),
            torch.cat(field_ids_list),
        )
