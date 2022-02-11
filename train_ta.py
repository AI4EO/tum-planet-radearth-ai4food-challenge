from pathlib import Path
from torch import nn
from torch.optim import Adam
from typing import List, Optional, Tuple

import argparse
import gpytorch
import numpy as np
import pandas as pd
import pdb
import torch
import wandb

from src.utils import DataLoader, train_valid_eval_utils as tveu
from src.utils.unrolled_lstm import UnrolledLSTM
from src.utils.gp_models import GPRegressionLayer1
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
arg_parser.add_argument("--gp_lr", type=float, default=0.01)
arg_parser.add_argument("--optimizer", type=str, default="Adam")
arg_parser.add_argument("--loss", type=str, default="SmoothL1")
arg_parser.add_argument("--validation_split", type=float, default=0.2)
arg_parser.add_argument("--split_by", type=str, default="longitude", help="latitude or longitude")
arg_parser.add_argument("--lstm_hidden_size", type=int, default=128)
arg_parser.add_argument("--lstm_dropout", type=float, default=0.2)
arg_parser.add_argument("--input_timesteps", type=int, default=36)
arg_parser.add_argument("--save_model_threshold", type=float, default=1.0)
arg_parser.add_argument("--gp_loss_weight", type=float, default=0.01)
arg_parser.add_argument("--gp_inference_index", type=int, default=10)
arg_parser.add_argument("--disable_wandb", dest="enable_wandb", action="store_false")
arg_parser.set_defaults(enable_wandb=True)

config = arg_parser.parse_args().__dict__

# Random seeds
assert config["satellite"] in ["sentinel_2", "planet_daily"]
assert config["pos"] in ["both", "34S_19E_258N", "34S_19E_259N"]
assert config["split_by"] in [None, "latitude", "longitude"]

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
    spatial_backbone="random_pixel",
    min_area_to_ignore=1000,
    train_or_test="train",
    temporal_augmentation=False,
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
class TemporalAugmentor(nn.Module):
    def __init__(
        self,
        num_bands,
        hidden_size,
        dropout,
        input_timesteps,
        output_timesteps,
        gp_inference_indexes,
        device,
    ):
        super(TemporalAugmentor, self).__init__()

        # LSTM
        self.lstm = UnrolledLSTM(
            input_size=num_bands,
            hidden_size=hidden_size,
            dropout=dropout,
            batch_first=True,
        )
        self.num_bands = num_bands
        self.input_timesteps = input_timesteps
        self.output_timesteps = output_timesteps
        self.to_bands = nn.Linear(in_features=hidden_size, out_features=num_bands)

        # Gaussian Process
        self.gp_layer = GPRegressionLayer1(batch_size=num_bands)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_size=num_bands)
        self.gp_inference_indexes = gp_inference_indexes

        self.to(device)

    def forward(self, x, training=True):

        hidden_tuple: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        seq_length = x.shape[1]

        gp_output_list: List[torch.Tensor] = []  # List to store gp outputs (training)
        lstm_output_list: List[torch.Tensor] = []  # List to store lstm outputs (training)
        inference_output_list: List[torch.Tensor] = []  # List to store lstm + gp output (inference)

        # Loop through input sequence
        for i in range(seq_length - 1):

            input = x[:, i : i + 1, :]

            if training:
                # Make prediction of next time step with gp (for training)
                gp_pred = self.gp_layer(input.permute(2, 0, 1))
                gp_output_list.append(gp_pred)

            if i < self.input_timesteps:
                # Use the input to make the next lstm prediction
                lstm_pred, hidden_tuple = self.lstm(input, hidden_tuple)
                lstm_pred = self.to_bands(torch.transpose(lstm_pred[0, :, :, :], 0, 1))
            else:
                if i == self.input_timesteps:
                    # Use the last lstm prediction from the real sequence as the first output step
                    if training:
                        lstm_output_list.append(lstm_pred)
                    else:
                        inference_output_list.append(lstm_pred)
                        next_step_pred = lstm_pred

                if training:
                    # Make a prediction of the next time step with lstm
                    lstm_pred, hidden_tuple = self.lstm(lstm_pred, hidden_tuple)
                    lstm_pred = self.to_bands(torch.transpose(lstm_pred[0, :, :, :], 0, 1))
                    lstm_output_list.append(lstm_pred)
                else:
                    if (i - self.input_timesteps) in self.gp_inference_indexes:
                        # Generate next hidden tuple
                        _, hidden_tuple = self.lstm(next_step_pred, hidden_tuple)
                        gp_pred = self.likelihood(self.gp_layer(next_step_pred.permute(2, 0, 1)))
                        next_step_pred = gp_pred.rsample().transpose(0, 1).unsqueeze(1)
                    else:
                        next_step_pred, hidden_tuple = self.lstm(next_step_pred, hidden_tuple)
                        next_step_pred = self.to_bands(
                            torch.transpose(next_step_pred[0, :, :, :], 0, 1)
                        )

                    inference_output_list.append(next_step_pred)

        if training:
            assert len(lstm_output_list) == self.output_timesteps
            return torch.cat(lstm_output_list, dim=1), gp_output_list
        else:
            assert len(inference_output_list) == self.output_timesteps
            return torch.cat(inference_output_list, dim=1)


model = TemporalAugmentor(
    num_bands=config["input_dim"],
    hidden_size=config["lstm_hidden_size"],
    dropout=config["lstm_dropout"],
    input_timesteps=config["input_timesteps"],
    output_timesteps=config["output_timesteps"],
    gp_inference_indexes=[config["gp_inference_index"]],
    device=DEVICE,
)

# OPTIONAL: trying gradient clipping to avoid loss being NaN.
clip_value = 1e-2
config["clip_value"] = clip_value
for p in model.parameters():
    if p.requires_grad:
        p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

# ----------------------------------------------------------------------------------------------------------------------
# Optimizer and loss function
# ----------------------------------------------------------------------------------------------------------------------
lstm_optimizer = Adam(model.parameters(), lr=config["lstm_lr"], weight_decay=1e-6)
gp_optimizer = Adam(
    [
        {"params": model.gp_layer.parameters()},
        {"params": model.likelihood.parameters()},
    ],
    lr=config["gp_lr"],
)
lstm_loss_func = nn.SmoothL1Loss()

# Our loss for GP object. We're using the VariationalELBO, which essentially just computes the ELBO
variational_elbo = gpytorch.mlls.VariationalELBO(
    model.likelihood, model.gp_layer, num_data=config["batch_size"], combine_terms=True
)


def gp_loss_func(gp_y_pred: List[torch.Tensor], y_true: torch.Tensor):
    gp_loss_sum = 0
    predicted_seq_len = y_true.shape[1] - 1
    for i in range(predicted_seq_len):
        gp_loss_sum -= variational_elbo(gp_y_pred[i], y_true[:, i + 1].transpose(0, 1)).sum()
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

valid_losses = []
accuracies = []
model_path = None
for epoch in range(config["num_epochs"] + 1):
    train_loss, train_lstm_loss, train_gp_loss = tveu.train_epoch_ta(
        model=model,
        lstm_optimizer=lstm_optimizer,
        gp_optimizer=gp_optimizer,
        lstm_loss_func=lstm_loss_func,
        gp_loss_func=gp_loss_func,
        dataloader=train_loader,
        device=DEVICE,
        gp_loss_weight=config["gp_loss_weight"],
    )
    valid_loss, valid_lstm_loss, valid_gp_loss = tveu.validation_epoch_ta(
        model=model,
        lstm_loss_func=lstm_loss_func,
        gp_loss_func=gp_loss_func,
        dataloader=valid_loader,
        device=DEVICE,
        gp_loss_weight=config["gp_loss_weight"],
    )

    losses = {
        "train": train_loss,
        "train_lstm": train_lstm_loss,
        "train_gp": train_gp_loss,
        "valid": valid_loss,
        "valid_lstm": valid_lstm_loss,
        "valid_gp": valid_gp_loss,
    }
    losses = {k: v.item() for k, v in losses.items()}

    valid_losses.append(losses["valid"])

    print(f"INFO: epoch {epoch}: train_loss {train_loss:.2f}, valid_loss {valid_loss:.2f} ")
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

    if not config["enable_wandb"]:
        continue

    closeness = np.abs(losses["valid"] - losses["train"]) + losses["valid"]
    losses["closeness"] = closeness
    wandb.log(
        {
            "losses": losses,
            "epoch": epoch,
        }
    )


if config["enable_wandb"]:
    if model_path:
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(model_path)
        run.log_artifact(artifact)
    run.finish()
