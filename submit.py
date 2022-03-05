import numpy as np
import pandas as pd
import pdb
import torch
import os
import geopandas as gpd
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
from sklearn.metrics import classification_report, accuracy_score

from helper import load_reader
from src.utils.baseline_models import SpatiotemporalModel
from src.temporal_augmentor import TemporalAugmentor

parser = ArgumentParser()
parser.add_argument("--model_path", type=str)
parser.add_argument("--model_type", type=str, help="(t)emporal augmentor, (o)riginal", default="o")
parser.add_argument("--competition", type=str, help="(g)ermany, (s)outh africa", default="g")
args = parser.parse_args()
 
assert "model_path" in args and "model_type" in args

# Load model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

saved = torch.load(args.model_path)
name = Path(args.model_path).parent.name + "-" + Path(args.model_path).stem
config = saved["config"]

model_type = "temporal_augmentor" if args.model_type.lower() == "t" else "original"
competition = "germany" if args.competition.lower() == "g" else "south_africa"

if model_type == "temporal_augmentor":
    assert all([f not in config for f in ['include_bands', "include_cloud", "include_ndvi", "image_size"]])

print(f"Creating: {name}.tar.gz ")

print(config)

if 'germany' in config['competition']:
    pos = '33N_17E_243N'
    config['competition'] = "germany"
    assert args.competition.lower() == "g"
    groundtruth = "/cmlscratch/hkjoo/repo/ai4eo/data/dlr_fusion_competition_germany_test_labels/dlr_fusion_competition_germany_test_labels_33N_17E_243N/crops_test_2019.geojson"
elif 'south_africa' in config['competition']:
    pos = "34S_20E_259N"
    config['competition'] = "south_africa"
    assert args.competition.lower() == "s"
    groundtruth = None # "Put the path to South Africa's groundtruth test geojson file"
else:
    raise NameError

groundtruth = gpd.read_file(groundtruth)

_, reader = load_reader(
    competition=config['competition'],
    satellite=config["satellite"],
    pos=pos,
    include_bands=config["include_bands"] if "include_bands" in config else True,
    include_cloud=config["include_cloud"] if "include_cloud" in config else True,
    include_ndvi=config["include_ndvi"] if "include_ndvi" in config else False,
    image_size=config["image_size"] if "image_size" in config else None,
    spatial_backbone=config["spatial_backbone"],
    min_area_to_ignore=0,
    train_or_test="test",
    s1_temporal_dropout=0.0,
    s2_temporal_dropout=0.0,
    planet_temporal_dropout=0.0,
)

print("\u2713 Data loaded")

if model_type == "original":
    model = SpatiotemporalModel(
        spatial_backbone=config["spatial_backbone"],
        temporal_backbone=config["temporal_backbone"],
        input_dim=config["input_dim"],
        num_classes=config["num_classes"],
        sequencelength=config["sequence_length"],
        device=DEVICE,
    )
elif model_type == "temporal_augmentor":
    model = TemporalAugmentor(
        num_bands=config["input_dim"],
        hidden_size=config["lstm_hidden_size"],
        dropout=config["lstm_dropout"],
        input_timesteps=config["input_timesteps"],
        output_timesteps=config["output_timesteps"],
        gp_inference_indexes=[config["gp_inference_index"]],
        device=DEVICE,
    )
else:
    raise NameError

model.load_state_dict(saved["model_state"])
model.eval()
print("\u2713 Model loaded")

softmax = torch.nn.Softmax()
output_list = []

predictions = []
groundtruths = []

if model_type == "original":
    with torch.no_grad():
        for X, _, mask, fid in tqdm(
            reader,
            total=len(reader),
            position=0,
            leave=True,
            desc="INFO: Saving predictions:",
        ):
            if config["spatial_backbone"] == "pixelsetencoder":
                logits = model((X.unsqueeze(0).to(DEVICE), mask.unsqueeze(0).to(DEVICE)))
                predicted_probabilities = softmax(logits).cpu().detach().numpy()[0]
            elif config["spatial_backbone"] == "random_pixel":
                # Ensemble the pixel predictions
                model_input = torch.permute(X, (2, 0, 1)).to(DEVICE)
                # Model input can get too large for GPU memory, so we need to split it up
                # into chunks of size batch_size
                model_input_chunks = torch.split(model_input, config["batch_size"], dim=0)
                probabilities_list = []
                for model_input_chunk in model_input_chunks:
                    logits = model(model_input_chunk)
                    predicted_probabilities = softmax(logits).cpu().detach().numpy()
                    probabilities_list.append(predicted_probabilities)
                predicted_probabilities = np.concatenate(probabilities_list).mean(axis=0)
            else:
                logits = model(X.unsqueeze(0).to(DEVICE))
                predicted_probabilities = softmax(logits).cpu().detach().numpy()[0]
            predicted_class = np.argmax(predicted_probabilities)

            groundtruths.append(fid)
            predictions.append(config["classes"][predicted_class])

            output_list.append(
                {
                    "fid": fid,
                    "crop_id": predicted_class + 1,  # save label list
                    "crop_name": config["classes"][predicted_class],
                    "crop_probs": predicted_probabilities,
                }
            )
elif model_type == "temporal_augmentor":
    pass

assert len(predictions) == len(groundtruths)
groundtruths = groundtruth.iloc[pd.Index(groundtruth['fid']).get_indexer(groundtruths)]['crop_name'].to_numpy().tolist()

names = sorted(set(groundtruths + predictions))
mapper = {k: v for k, v in zip(names, range(len(names)))}

y_true = [mapper[f] for f in groundtruths]
y_pred = [mapper[f] for f in predictions]

accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=names, output_dict=True)


# output_frame = pd.DataFrame.from_dict(output_list)

# submission_path = Path(f"submissions/{name}/{pos}-submission-{name}.json")
# submission_path.parent.mkdir(parents=True, exist_ok=True)
# output_frame.to_json(submission_path)

# print(f"Saving: {name}.tar.gz ")

# os.system(f"cd submissions && tar czf {name}.tar.gz {name}")
