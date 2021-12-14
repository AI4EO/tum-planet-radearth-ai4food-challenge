import argparse
import numpy as np
import pandas as pd
import torch
import os

from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser

from helper import load_reader
from notebook.utils.baseline_models import SpatiotemporalModel

parser = ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--name", type=str)
args = parser.parse_args()

assert "model_path" in args
assert "name" in args

# Load model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

saved = torch.load(args.model_path)
config = saved["config"]

print(config)

_, reader = load_reader(
    satellite=config["satellite"],
    pos="34S_20E_259N",
    include_bands=config["include_bands"],
    include_cloud=config["include_cloud"],
    include_ndvi=config["include_ndvi"],
    image_size=config["image_size"],
    spatial_backbone=config["spatial_backbone"],
    min_area_to_ignore=0,
    train_or_test="test",
)

print("\u2713 Data loaded")

model = SpatiotemporalModel(
    spatial_backbone=config["spatial_backbone"],
    input_dim=config["input_dim"],
    num_classes=config["num_classes"],
    sequencelength=config["sequence_length"],
    device=DEVICE,
)

model.load_state_dict(saved["model_state"])
model.eval()
print("\u2713 Model loaded")

output_list = []
softmax = torch.nn.Softmax()

for X, _, mask, fid in tqdm(
    reader,
    total=len(reader),
    position=0,
    leave=True,
    desc="INFO: Saving predictions:",
):
    if config["spatial_backbone"] == "pixelsetencoder":
        logits = model((X.unsqueeze(0).to(DEVICE), mask.unsqueeze(0).to(DEVICE)))
    else:
        logits = model(X.unsqueeze(0).to(DEVICE))
    predicted_probabilities = softmax(logits).cpu().detach().numpy()[0]
    predicted_class = np.argmax(predicted_probabilities)

    output_list.append(
        {
            "fid": fid,
            "crop_id": predicted_class + 1,  # save label list
            "crop_name": config["classes"][predicted_class],
            "crop_probs": predicted_probabilities,
        }
    )

output_frame = pd.DataFrame.from_dict(output_list)

submission_path = Path(f"submissions/{args.name}/34S-20E-259N-2017-submission-{args.name}.json")
submission_path.parent.mkdir(parents=True, exist_ok=True)
output_frame.to_json(submission_path)

os.system(f"cd submissions && tar czf {args.name}.tar.gz {args.name}")
