import pdb
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD

import wandb
from helper import load_reader
from src.utils import train_valid_eval_utils as tveu
from src.utils.baseline_models import SpatiotemporalModel
from src.utils.data_loader import DataLoader

import time

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)

# ----------------------------------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

arg_parser = ArgumentParser()
arg_parser.add_argument(
    "--competition", 
    type=str, 
    default="south_africa",
    help="germany, south_africa"
)
arg_parser.add_argument("--model_type", type=str, default="spatiotemporal")
arg_parser.add_argument("--batch_size", type=int, default=64)
arg_parser.add_argument("--num_epochs", type=int, default=100)
arg_parser.add_argument(
    "--satellite",
    type=str,
    default="sentinel_2",
    help="sentinel_1, sentinel_2, or planet_5day, s1_s2, planet_daily, s1_s2_planet_daily",
)
arg_parser.add_argument(
    "--pos", type=str, default="both_34", help="both_34, 34S_19E_258N, 34S_19E_259N, 33N_18E_242N"
)
arg_parser.add_argument("--lr", type=float, default=0.001)
arg_parser.add_argument("--optimizer", type=str, default="Adam")
arg_parser.add_argument("--loss", type=str, default="CrossEntropyLoss")
arg_parser.add_argument("--spatial_backbone", type=str, default="mean_pixel")
arg_parser.add_argument("--temporal_backbone", type=str, default="tempcnn")
arg_parser.add_argument("--image_size", type=int, default=32)
arg_parser.add_argument("--save_model_threshold", type=float, default=0.9)
arg_parser.add_argument("--pse_sample_size", type=int, default=32)
arg_parser.add_argument("--validation_split", type=float, default=0.2)
arg_parser.add_argument("--split_by", type=str, default="longitude", help="latitude or longitude")
arg_parser.add_argument("--include_bands", type=bool, default=True)
arg_parser.add_argument("--include_cloud", type=bool, default=True)
arg_parser.add_argument("--include_ndvi", type=bool, default=False)
arg_parser.add_argument("--include_rvi", type=bool, default=False)
arg_parser.add_argument("--alignment", type=str, default="1to2", help="Can be: 1to2 or 2to1 (76 vs. 41 for SA, 144 vs. 122)")

# WandB params
arg_parser.add_argument("--s1_temporal_dropout", type=float, default=0.0)
arg_parser.add_argument("--s2_temporal_dropout", type=float, default=0.0)
arg_parser.add_argument("--planet_temporal_dropout", type=float, default=0.0)
arg_parser.add_argument("--lr_scheduler", type=str, default="none")
arg_parser.add_argument("--ta_model_path", type=str, default="")

arg_parser.add_argument("--disable_wandb", dest="enable_wandb", action="store_false")
arg_parser.set_defaults(enable_wandb=True)
arg_parser.add_argument("--name", type=str, default=None, help="Manually the run name (e.g., snowy-owl-10); None for automatic naming.")
arg_parser.add_argument("--unique", dest="unique", action="store_true", help="Make the name unique by appending random digits after the name")
arg_parser.set_defaults(unique=False)
arg_parser.add_argument("--project", type=str, default="original", help="original (Ivan), kevin")

config = arg_parser.parse_args().__dict__

assert config["satellite"] in [
    "sentinel_1",
    "sentinel_2",
    "planet_5day",
    "s1_s2",
    "planet_daily",
    "s1_s2_planet_daily",
]
assert config["pos"] in ["both_34", "34S_19E_258N", "34S_19E_259N", "33N_18E_242N"]
assert config["competition"] in ["germany", "south_africa"]
assert config["split_by"] in [None, "latitude", "longitude"]

if config['competition'] == 'germany':
    assert config['pos'] == "33N_18E_242N"
elif config['competition'] == 'south_africa':
    assert config['pos'] in ['both_34', '34S_19E_258N', '34S_19E_259N']

if config['project'] == 'original':
    config['project'] = "ai4food-challenge"
else:
    config['project'] = "ai4food-challenge-germany"

if str(config['name']) == 'None':
    config['name'] = None
elif config['unique']:
    config['name'] += "_" + str(int(time.time()))[-4:]

# ---------------------------------------------------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------------------------------------------------
# Initialize data loaders
kwargs = dict(
    competition=config["competition"],
    satellite=config["satellite"],
    include_bands=config["include_bands"],
    include_cloud=config["include_cloud"],
    include_ndvi=config["include_ndvi"],
    include_rvi=config["include_rvi"],
    image_size=config["image_size"],
    spatial_backbone=config["spatial_backbone"],
    pse_sample_size=config["pse_sample_size"],
    min_area_to_ignore=1000,
    train_or_test="train",
    alignment=config["alignment"],
    s1_temporal_dropout=config["s1_temporal_dropout"],
    s2_temporal_dropout=config["s2_temporal_dropout"],
    planet_temporal_dropout=config["planet_temporal_dropout"],
)

if config["pos"] == "both_34":
    label_names_258, reader_258 = load_reader(pos="34S_19E_258N", **kwargs)
    print("\u2713 Loaded 258")
    label_names_259, reader_259 = load_reader(pos="34S_19E_259N", **kwargs)
    print("\u2713 Loaded 259")
    reader_258.labels.reset_index(inplace=True)
    reader_259.labels.reset_index(inplace=True)
    assert (
        label_names_258 == label_names_259
    ), f"{label_names_258} and {label_names_259} are not equal"
    label_names = label_names_258
    reader = torch.utils.data.ConcatDataset([reader_258, reader_259])
    reader.labels = pd.concat([reader_258.labels, reader_259.labels], ignore_index=True)
else:
    label_names, reader = load_reader(pos=config["pos"], **kwargs)
    reader.labels.reset_index(inplace=True)

config["num_classes"] = len(label_names)
config["classes"] = label_names
config["input_dim"] = reader[0][0].shape[1]
config["sequence_length"] = reader[0][0].shape[0]
config["X_shape"] = reader[0][0].shape

print("\u2713 Datasets initialized")

data_loader = DataLoader(
    train_val_reader=reader,
    validation_split=config["validation_split"],
    split_by=config["split_by"],
)
train_loader = data_loader.get_train_loader(batch_size=config["batch_size"], num_workers=0)
valid_loader = data_loader.get_validation_loader(batch_size=config["batch_size"], num_workers=0)

config["train_dataset_size"] = len(train_loader.dataset)
config["train_minibatch_size"] = len(train_loader)
config["val_dataset_size"] = len(valid_loader.dataset)
config["val_minibatch_size"] = len(valid_loader)

print("\u2713 Data loaders initialized")
# ----------------------------------------------------------------------------------------------------------------------
# Initialize model
# ----------------------------------------------------------------------------------------------------------------------
model = SpatiotemporalModel(
    spatial_backbone=config["spatial_backbone"],
    temporal_backbone=config["temporal_backbone"],
    input_dim=config["input_dim"],
    num_classes=len(label_names),
    sequencelength=config["sequence_length"],
    ta_model_path=config["ta_model_path"],
    device=DEVICE,
)

# OPTIONAL: trying gradient clipping to avoid loss being NaN.
clip_value = 1e-2
config["clip_value"] = clip_value
for p in model.parameters():
    if p.requires_grad:
        p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

print("\u2713 Model initialized")
# ----------------------------------------------------------------------------------------------------------------------
# Optimizer and loss function
# ----------------------------------------------------------------------------------------------------------------------
if config["optimizer"] == "Adam":
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
elif config["optimizer"] == "SGD":
    optimizer = SGD(model.parameters(), lr=config["lr"], momentum=0.9)
scheduler = None
if config["lr_scheduler"] == "onecyclelr":
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["lr"],
        steps_per_epoch=config["train_dataset_size"],
        epochs=config["num_epochs"],
    )

# config["weight"] = [1.0, 1.0, 1.0, 1.0, 1.0]
# weight = torch.tensor(config["weight"]).to(DEVICE)
loss_criterion = CrossEntropyLoss(reduction="mean")

print("\u2713 Optimizer and loss set")
# ----------------------------------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------------------------------
print("Beginning training")
print(config)
if config["enable_wandb"]:
    run = wandb.init(
        entity="nasa-harvest",
        project=config['project'],
        name=config['name'],
        config=config,
        settings=wandb.Settings(start_method="fork"),
    )

# Don't turn on until needed
# wandb.watch(model, log_freq=100)
valid_losses = []
accuracies = []
model_path = None
for epoch in range(config["num_epochs"] + 1):
    train_loss = tveu.train_epoch(
        model, optimizer, loss_criterion, train_loader, device=DEVICE, scheduler=scheduler
    )
    valid_loss, y_true, y_pred, *_ = tveu.validation_epoch(
        model, loss_criterion, valid_loader, device=DEVICE
    )

    scores = tveu.metrics(y_true.cpu(), y_pred.cpu())

    scores_msg = ", ".join([f"{k}={v:.2f}" for (k, v) in scores.items()])

    valid_loss = valid_loss.cpu().detach().numpy().mean()
    train_loss = train_loss.cpu().detach().numpy().mean()
    valid_losses.append(valid_loss)
    accuracies.append(scores["accuracy"])

    scores["epoch"] = epoch
    scores["train_loss"] = train_loss
    scores["valid_loss"] = valid_loss

    cm = confusion_matrix(
        y_true=y_true, y_pred=y_pred.cpu().detach().numpy(), labels=np.arange(len(label_names))
    )

    print(
        f"INFO: epoch {epoch}: train_loss {train_loss:.2f}, valid_loss {valid_loss:.2f} "
        + scores_msg
    )
    if config["save_model_threshold"] > valid_loss:
        model_path = f"model_dump/{run.id}/{epoch}.pth"
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                model_state=model.state_dict(),
                # optimizer_state=optimizer.state_dict(),
                epoch=epoch,
                config=config,
            ),
            model_path,
        )

    if not config["enable_wandb"]:
        continue

    metrics = {
        key: scores[key]
        for key in [
            "accuracy",
            "kappa",
            "f1_micro",
            "f1_macro",
            "f1_weighted",
            "recall_micro",
            "recall_macro",
            "recall_weighted",
            "precision_micro",
            "precision_macro",
            "precision_weighted",
        ]
    }
    metrics["max_accuracy"] = max(accuracies)
    closeness = np.abs(valid_loss - train_loss) + valid_loss
    wandb.log(
        {
            "losses": dict(
                train=train_loss,
                valid=valid_loss,
                valid_min=min(valid_losses),
                train_val_closeness=closeness,
            ),
            "epoch": epoch,
            "metrics": metrics,
            "confusion_matrix": tveu.confusion_matrix_figure(cm, labels=label_names),
        }
    )


if config["enable_wandb"]:
    if model_path:
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(model_path)
        run.log_artifact(artifact)
    run.finish()
