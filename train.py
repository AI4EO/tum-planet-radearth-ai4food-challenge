import geopandas as gpd
import numpy as np
import random
import torch
import wandb
from sklearn.metrics import confusion_matrix
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from notebook.utils import train_valid_eval_utils as tveu
from notebook.utils.baseline_models import SpatiotemporalModel
from notebook.utils.data_loader import DataLoader
from notebook.utils.data_transform import (
    Sentinel2Transform,
)  # , PlanetTransform, Sentinel1Transform,
from notebook.utils.sentinel_1_reader import S1Reader
from notebook.utils.sentinel_2_reader import S2Reader

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# ----------------------------------------------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
competition = "ref_fusion_competition_south_africa"
INPUT_DIM = 12  # number of channels in S2 data
SEQUENCE_LENGTH = 50  # Sequence size of Temporal Data
START_EPOCH = 0
TOTAL_EPOCH = 10
KEY = "s2_258"
BATCH_SIZE = 8
LR = 1e-3

s2_input_dir = f"data/{competition}_train_source_sentinel_2/{competition}_train_source_sentinel_2"
s1_input_dir = f"data/{competition}_train_source_sentinel_1/{competition}_train_source_sentinel_1"

# ----------------------------------------------------------------------------------------------------------------------
# Data loaders
# ----------------------------------------------------------------------------------------------------------------------


def load_reader(key: str):

    if key.endswith("258"):
        pos = "34S_19E_258N"
    else:
        pos = "34S_19E_259N"

    label_file = f"data/{competition}_train_labels/{competition}_train_labels_{pos}/labels.geojson"
    labels = gpd.read_file(label_file)
    label_ids = labels["crop_id"].unique()
    label_names = labels["crop_name"].unique()

    if key.startswith("s1"):
        reader = S1Reader(
            input_dir=f"{s1_input_dir}_{pos}_asc_{pos}_2017",
            label_ids=label_ids,
            label_dir=label_file,
            min_area_to_ignore=1000,
        )
    elif key.startswith("s2"):
        reader = S2Reader(
            input_dir=f"{s2_input_dir}_{pos}_{pos}_2017",
            label_ids=label_ids,
            label_dir=label_file,
            transform=Sentinel2Transform().transform,
            min_area_to_ignore=1000,
        )

    return label_names, reader


# Initialize data loaders
label_names, reader = load_reader(KEY)

data_loader = DataLoader(train_val_reader=reader, validation_split=0.25)
train_loader = data_loader.get_train_loader(batch_size=BATCH_SIZE, num_workers=1)
valid_loader = data_loader.get_validation_loader(batch_size=BATCH_SIZE, num_workers=1)

print("\u2713 Data loaders initialized")
# ----------------------------------------------------------------------------------------------------------------------
# Initialize model
# ----------------------------------------------------------------------------------------------------------------------
model = SpatiotemporalModel(
    input_dim=INPUT_DIM, num_classes=len(label_names), sequencelength=SEQUENCE_LENGTH, device=DEVICE
)
model = model.to(DEVICE)

# OPTIONAL: trying gradient clipping to avoid loss being NaN.
clip_value = 1e-2
for p in model.parameters():
    p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

print("\u2713 Model initialized")
# ----------------------------------------------------------------------------------------------------------------------
# Optimizer and loss function
# ----------------------------------------------------------------------------------------------------------------------
optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-6)
loss_criterion = CrossEntropyLoss(reduction="mean")

print("\u2713 Optimizer and loss set")
# ----------------------------------------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------------------------------------
print("Beginning training")
run = wandb.init(
    entity="nasa-harvest",
    project="ai4food-challenge",
    config={
        "model_type": "baseline",
        "key": KEY,
        "competition": competition,
        "sequence_length": SEQUENCE_LENGTH,
        "input_dim": INPUT_DIM,
        "batch_size": BATCH_SIZE,
        "num_classes": len(label_names),
        "classes": label_names,
        "optimizer": "Adam",
        "lr": LR,
        "clip_value": clip_value,
        "device": DEVICE,
        "start_epoch": START_EPOCH,
        "total_epoch": TOTAL_EPOCH,
    },
)
wandb.watch(model, log_freq=100)

for epoch in range(START_EPOCH, TOTAL_EPOCH):
    train_loss = tveu.train_epoch(model, optimizer, loss_criterion, train_loader, device=DEVICE)
    valid_loss, y_true, y_pred, *_ = tveu.validation_epoch(
        model, loss_criterion, valid_loader, device=DEVICE
    )

    scores = tveu.metrics(y_true.cpu(), y_pred.cpu())

    scores_msg = ", ".join([f"{k}={v:.2f}" for (k, v) in scores.items()])

    valid_loss = valid_loss.cpu().detach().numpy()[0]
    train_loss = train_loss.cpu().detach().numpy()[0]

    scores["epoch"] = epoch
    scores["train_loss"] = train_loss
    scores["valid_loss"] = valid_loss

    cm = confusion_matrix(
        y_true=y_true, y_pred=y_pred.cpu().detach().numpy(), labels=np.arange(len(label_names))
    )

    wandb.log(
        {
            "losses": dict(train=train_loss, valid=valid_loss),
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

    model_path = f"model_dump/{run.id}_{epoch}.pth"
    torch.save(
        dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(), epoch=epoch),
        model_path,
    )
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file(model_path)
    run.log_artifact(artifact)

    print(
        f"INFO: epoch {epoch}: train_loss {train_loss:.2f}, valid_loss {valid_loss:.2f} "
        + scores_msg
    )

run.finish()
