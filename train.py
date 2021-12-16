import pdb
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

import wandb
from helper import load_reader
from notebook.utils import train_valid_eval_utils as tveu
from notebook.utils.baseline_models import SpatiotemporalModel
from notebook.utils.data_loader import DataLoader

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)
# torch.use_deterministic_algorithms(True)


# ----------------------------------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
competition = "ref_fusion_competition_south_africa"
arg_parser = ArgumentParser()
arg_parser.add_argument("--competition", type=str, default=competition)
arg_parser.add_argument("--model_type", type=str, default="spatiotemporal")
arg_parser.add_argument("--batch_size", type=int, default=64)
arg_parser.add_argument("--num_epochs", type=int, default=100)
arg_parser.add_argument(
    "--satellite",
    type=str,
    default="sentinel_2",
    help="sentinel_1, sentinel_2, or planet_5day",
)
arg_parser.add_argument(
    "--pos", type=str, default="both", help="Can be: both, 34S_19E_258N, 34S_19E_259N"
)
arg_parser.add_argument("--lr", type=float, default=0.001)
arg_parser.add_argument("--optimizer", type=str, default="Adam")
arg_parser.add_argument("--loss", type=str, default="CrossEntropyLoss")
arg_parser.add_argument("--spatial_backbone", type=str, default="mean_pixel")
arg_parser.add_argument("--temporal_backbone", type=str, default="LSTM")
arg_parser.add_argument("--image_size", type=int, default=32)
arg_parser.add_argument("--save_model_validation_threshold", type=float, default=0.6)
arg_parser.add_argument("--pse_sample_size", type=int, default=64)
arg_parser.add_argument("--validation_split", type=float, default=0.1)
arg_parser.add_argument("--split_by", type=str, default="longitude", help="latitude or longitude")
arg_parser.add_argument("--skip_bands", dest="include_bands", action="store_false")
arg_parser.set_defaults(include_bands=True)
arg_parser.add_argument("--skip_cloud", dest="include_cloud", action="store_false")
arg_parser.set_defaults(include_cloud=True)
arg_parser.add_argument("--skip_ndvi", dest="include_ndvi", action="store_false")
arg_parser.set_defaults(include_ndvi=True)
arg_parser.add_argument("--disable_wandb", dest="enable_wandb", action="store_false")
arg_parser.set_defaults(enable_wandb=True)
config = arg_parser.parse_args().__dict__

assert config["satellite"] in ["sentinel_1", "sentinel_2", "planet_5day", "s1_s2", "planet_daily"]
assert config["pos"] in ["both", "34S_19E_258N", "34S_19E_259N"]
assert config["split_by"] in [None, "latitude", "longitude"]
# ---------------------------------------------------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------------------------------------------------
# Initialize data loaders
kwargs = dict(
    satellite=config["satellite"],
    include_bands=config["include_bands"],
    include_cloud=config["include_cloud"],
    include_ndvi=config["include_ndvi"],
    image_size=config["image_size"],
    spatial_backbone=config["spatial_backbone"],
    pse_sample_size=config["pse_sample_size"],
    min_area_to_ignore=1000,
    train_or_test="train",
)

if config["pos"] == "both":
    label_names_258, reader_258 = load_reader(pos="34S_19E_258N", **kwargs)
    print("\u2713 Loaded 258")
    label_names_259, reader_259 = load_reader(pos="34S_19E_259N", **kwargs)
    print("\u2713 Loaded 259")
    assert (
        label_names_258 == label_names_259
    ), f"{label_names_258} and {label_names_259} are not equal"
    label_names = label_names_258
    reader = torch.utils.data.ConcatDataset([reader_258, reader_259])
    reader.labels = pd.concat([reader_258.labels, reader_259.labels], ignore_index=True)
else:
    label_names, reader = load_reader(pos=config["pos"], **kwargs)

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
    device=DEVICE,
)

# OPTIONAL: trying gradient clipping to avoid loss being NaN.
clip_value = 1e-2
config["clip_value"] = clip_value
for p in model.parameters():
    p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

print("\u2713 Model initialized")
# ----------------------------------------------------------------------------------------------------------------------
# Optimizer and loss function
# ----------------------------------------------------------------------------------------------------------------------
optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
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
        project="ai4food-challenge",
        config=config,
    )

# Don't turn on until needed
# wandb.watch(model, log_freq=100)
valid_losses = []
model_path = None
for epoch in range(config["num_epochs"] + 1):
    train_loss = tveu.train_epoch(model, optimizer, loss_criterion, train_loader, device=DEVICE)
    valid_loss, y_true, y_pred, *_ = tveu.validation_epoch(
        model, loss_criterion, valid_loader, device=DEVICE
    )

    scores = tveu.metrics(y_true.cpu(), y_pred.cpu())

    scores_msg = ", ".join([f"{k}={v:.2f}" for (k, v) in scores.items()])

    valid_loss = valid_loss.cpu().detach().numpy()[0]
    train_loss = train_loss.cpu().detach().numpy()[0]
    valid_losses.append(valid_loss)

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
    if config["save_model_validation_threshold"] > valid_loss:
        model_path = f"model_dump/{run.id}/{epoch}.pth"
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                epoch=epoch,
                config=config,
            ),
            model_path,
        )

    if not config["enable_wandb"]:
        continue

    wandb.log(
        {
            "losses": dict(train=train_loss, valid=valid_loss, valid_min=min(valid_losses)),
            "epoch": epoch,
            "metrics": {
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
            },
            "confusion_matrix": tveu.confusion_matrix_figure(cm, labels=label_names),
        }
    )


if config["enable_wandb"]:
    if model_path:
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(model_path)
        run.log_artifact(artifact)
    run.finish()
