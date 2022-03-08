from pathlib import Path
from torch.optim import Adam, SGD
from typing import List

import argparse
import gpytorch
import numpy as np
import pandas as pd
import pdb
import random
import torch
import wandb

from visualize_ta import plot_preds
from src.temporal_augmentor import TemporalAugmentor
from src.utils import DataLoader, train_valid_eval_utils as tveu
from helper import load_reader


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
competition = "ref_fusion_competition_south_africa"

# Argument parser for command line arguments

arg_parser = argparse.ArgumentParser(description="Train a model for temporal augmentation")
arg_parser.add_argument("--competition", type=str, default=competition)
arg_parser.add_argument("--model_type", type=str, default="spatiotemporal")
arg_parser.add_argument("--batch_size", type=int, default=64)
arg_parser.add_argument("--num_epochs", type=int, default=100)
arg_parser.add_argument(
    "--satellite", type=str, default="planet_daily", help="sentinel_2, planet_daily"
)
arg_parser.add_argument(
    "--pos", type=str, default="both", help="Can be: both, 34S_19E_258N, 34S_19E_259N"
)
arg_parser.add_argument("--lstm_lr", type=float, default=0.001)
arg_parser.add_argument("--lstm_weight_decay", type=float, default=1e-6)
arg_parser.add_argument("--gp_lr", type=float, default=0.01)
arg_parser.add_argument("--optimizer", type=str, default="Adam")
arg_parser.add_argument("--loss", type=str, default="SmoothL1")
arg_parser.add_argument("--validation_split", type=float, default=0.2)
arg_parser.add_argument("--split_by", type=str, default="longitude", help="latitude or longitude")
arg_parser.add_argument("--lstm_hidden_size", type=int, default=128)
arg_parser.add_argument("--lstm_dropout", type=float, default=0.1)
arg_parser.add_argument("--lstm_layers", type=int, default=1)
arg_parser.add_argument("--input_timesteps", type=int, default=100)
arg_parser.add_argument("--save_model_threshold", type=float, default=0.2)
arg_parser.add_argument("--gp_loss_weight", type=float, default=0.01)
arg_parser.add_argument("--gp_inference_index", type=int, default=10)
arg_parser.add_argument("--gp_interval", type=int, default=1)
arg_parser.add_argument("--spatial_backbone", type=str, default="mean_pixel")
arg_parser.add_argument("--disable_wandb", dest="enable_wandb", action="store_false")
arg_parser.add_argument("--debug", dest="debug", action="store_true")
arg_parser.add_argument("--planet_temporal_dropout", type=float, default=0.0)
arg_parser.add_argument("--plot_amount", type=int, default=3)
arg_parser.add_argument("--disable_gp", dest="gp_enabled", action="store_false")
arg_parser.add_argument("--use_teacher_forcing", dest="teacher_forcing", action="store_true")
arg_parser.add_argument("--lstm_type", type=str, default="unrolled")
arg_parser.set_defaults(gp_enabled=True)
arg_parser.set_defaults(enable_wandb=True)
arg_parser.set_defaults(teacher_forcing=False)
arg_parser.set_defaults(debug=False)

config = arg_parser.parse_args().__dict__

# Random seeds
assert config["satellite"] in ["sentinel_2", "planet_daily", "planet_5day"]
assert config["pos"] in ["both", "34S_19E_258N", "34S_19E_259N"]
assert config["split_by"] in [None, "latitude", "longitude"]
assert config["loss"] in ["SmoothL1", "MSE", "L1"]

config["normalize"] = True
config["gp_enabled"] = False

# ----------------------------------------------------------------------------------------------------------------------
# Data loaders
# ----------------------------------------------------------------------------------------------------------------------
kwargs = dict(
    satellite=config["satellite"],
    include_bands=True,
    include_cloud=True,
    include_ndvi=False,
    include_rvi=False,
    image_size=None,
    spatial_backbone=config["spatial_backbone"],
    min_area_to_ignore=1000,
    train_or_test="train",
    planet_temporal_dropout=config["planet_temporal_dropout"],
    normalize=config["normalize"],
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

config["input_dim"] = reader[0][0].shape[1]
config["sequence_length"] = reader[0][0].shape[0]
config["X_shape"] = reader[0][0].shape
if config["input_timesteps"] is None:
    config["output_timesteps"] = None
else:
    config["output_timesteps"] = config["sequence_length"] - config["input_timesteps"]

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
gp_enabled = config["gp_enabled"]
gp_interval = config["gp_interval"]
model = TemporalAugmentor(
    num_bands=config["input_dim"],
    hidden_size=config["lstm_hidden_size"],
    dropout=config["lstm_dropout"],
    input_timesteps=config["input_timesteps"],
    output_timesteps=config["output_timesteps"],
    gp_inference_indexes=[config["gp_inference_index"]],
    device=DEVICE,
    gp_enabled=gp_enabled,
    teacher_forcing=config["teacher_forcing"],
    gp_interval=gp_interval,
    lstm_layers=config["lstm_layers"],
    lstm_type=config["lstm_type"],
    perturb_h_indexes=[10, 20],
)

# OPTIONAL: trying gradient clipping to avoid loss being NaN.
# clip_value = 1e-2
# config["clip_value"] = clip_value
# for p in model.parameters():
#     if p.requires_grad:
#         p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

# ----------------------------------------------------------------------------------------------------------------------
# Optimizer and loss function
# ----------------------------------------------------------------------------------------------------------------------
if config["optimizer"] == "Adam":
    optimizer = Adam
elif config["optimizer"] == "SGD":
    optimizer = SGD
else:
    raise ValueError(f"{config['optimizer']} is not a valid optimizer")

lstm_optimizer = optimizer(
    model.parameters(), lr=config["lstm_lr"], weight_decay=config["lstm_weight_decay"]
)
if config["loss"] == "SmoothL1":
    lstm_loss_func = nn.SmoothL1Loss()
elif config["loss"] == "L1":
    lstm_loss_func = nn.L1Loss()
elif config["loss"] == "MSE":
    lstm_loss_func = nn.MSELoss()
else:
    raise ValueError(f"{config['loss']} is not a valid loss function")

if gp_enabled:
    gp_optimizer = optimizer(
        [
            {"params": model.gp_layer.parameters()},
            {"params": model.likelihood.parameters()},
        ],
        lr=config["gp_lr"],
    )

    # Our loss for GP object. We're using the VariationalELBO, which essentially just computes the ELBO
    variational_elbo = gpytorch.mlls.VariationalELBO(
        model.likelihood, model.gp_layer, num_data=config["batch_size"], combine_terms=True
    )
else:
    gp_optimizer = None
    variational_elbo = None


def gp_loss_func(gp_y_pred: List[torch.Tensor], y_true: torch.Tensor):
    gp_loss_sum = 0
    predicted_seq_len = len(gp_y_pred)
    for i in range(0, predicted_seq_len, gp_interval):
        gp_y_true = y_true[:, i : i + gp_interval].view(y_true.shape[0], -1).transpose(0, 1)
        gp_loss_sum -= variational_elbo(gp_y_pred[i], gp_y_true).sum()
    return gp_loss_sum / predicted_seq_len


print("\u2713 Optimizer and loss set")
# ----------------------------------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------------------------------
print("Beginning training")
print(config)
if config["enable_wandb"]:
    run = wandb.init(
        entity="nasa-harvest",
        project="temporal-augmentation",
        config=config,
        settings=wandb.Settings(start_method="fork"),
    )

all_valid_losses = []
accuracies = []
model_path = None
for epoch in range(config["num_epochs"] + 1):

    train_losses = tveu.train_epoch_ta(
        model=model,
        lstm_optimizer=lstm_optimizer,
        gp_optimizer=gp_optimizer,
        lstm_loss_func=lstm_loss_func,
        gp_loss_func=gp_loss_func,
        dataloader=train_loader,
        device=DEVICE,
        gp_loss_weight=config["gp_loss_weight"],
        gp_enabled=gp_enabled,
        debug=config["debug"],
    )
    valid_losses = tveu.validation_epoch_ta(
        model=model,
        lstm_loss_func=lstm_loss_func,
        gp_loss_func=gp_loss_func,
        dataloader=valid_loader,
        device=DEVICE,
        gp_loss_weight=config["gp_loss_weight"],
        gp_enabled=gp_enabled,
        debug=config["debug"],
    )

    if gp_enabled:
        train_loss, train_lstm, train_gp = train_losses
        valid_loss, valid_lstm, valid_gp = valid_losses
        losses = {
            "train": train_loss,
            "train_lstm": train_lstm,
            "train_gp": train_gp,
            "valid": valid_loss,
            "valid_lstm": valid_lstm,
            "valid_gp": valid_gp,
        }
    else:
        losses = {
            "train_lstm": train_losses,
            "valid_lstm": valid_losses,
        }
        train_loss = train_losses
        valid_loss = valid_losses

    losses = {k: v.item() for k, v in losses.items()}

    all_valid_losses.append(valid_losses)

    print(f"INFO: epoch {epoch}: train_loss {train_loss:.2f}, valid_loss {valid_loss:.2f} ")

    if not config["enable_wandb"]:
        continue

    if config["save_model_threshold"] > valid_loss:
        model_path = f"temporal_augment_model_dump/{run.id}/{epoch}.pth"
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

    # Make predictions on validation set
    model.eval()
    with torch.no_grad():
        if config["debug"]:
            x = torch.stack([train_loader.dataset[i][0] for i in range(config["plot_amount"])]).to(
                DEVICE
            )
        else:
            x = torch.stack([valid_loader.dataset[i][0] for i in range(config["plot_amount"])]).to(
                DEVICE
            )

    if gp_enabled:
        losses["closeness"] = np.abs(losses["valid"] - losses["train"]) + losses["valid"]

    # Model timesteps must be set before plotting predictions
    timesteps_manually_set = False
    if model.input_timesteps is None:
        model.input_timesteps = np.random.randint(10, x.shape[1] - 10)
        timesteps_manually_set = True

    to_log = {
        "losses": losses,
        "epoch": epoch,
        "lstm_predictions_plot": plot_preds(title="LSTM only", model=model, x=x),
        "lstm_predictions_dropout_plot": plot_preds(
            title="LSTM with dropout", model=model, x=x, preds_with_dropout=3
        ),
        "lstm_predictions_perturb_plot": plot_preds(
            title="LSTM with perturbation (10 random)",
            model=model,
            x=x,
            perturb_h_indexes=random.sample(range(0, 144), 10),
            perturb_amount=0.5,
            predict_amount=3,
        ),
        "lstm_predictions_perturb_and_dropout_plot": plot_preds(
            title="LSTM with dropout and perturbation (10 random)",
            model=model,
            x=x,
            preds_with_dropout=3,
            perturb_h_indexes=random.sample(range(0, 144), 10),
            perturb_amount=0.5,
        ),
    }
    if gp_enabled:
        to_log["gp_10th_plot"] = plot_preds(title="GP at 10th", gp_indexes=[10], model=model, x=x)
        to_log["gp_5_random_plot"] = plot_preds(
            title="GP random trigger 5 times",
            gp_indexes=random.sample(range(0, 144), 5),
            model=model,
            x=x,
        )

    if timesteps_manually_set:
        model.input_timesteps = None

    wandb.log(to_log)


if config["enable_wandb"]:
    if model_path:
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(model_path)
        run.log_artifact(artifact)
    run.finish()
