from torch import nn
from typing import List, Optional, Tuple

import gpytorch
import numpy as np
import pdb
import torch

from src.utils.unrolled_lstm import UnrolledLSTM
from src.utils.gp_models import GPRegressionLayer1


class TemporalAugmentor(nn.Module):
    def __init__(
        self,
        num_bands,
        hidden_size,
        dropout,
        input_timesteps,
        output_timesteps,
        device,
        gp_inference_indexes=[],
        perturb_h_indexes=[],
        gp_enabled=True,
        teacher_forcing=False,  # Whether to use teacher forcing for LSTM training
        gp_interval=1,  # Number of timesteps to use for GP
        lstm_layers=1,
        lstm_type="pytorch",
    ):
        super(TemporalAugmentor, self).__init__()

        # LSTM
        self.lstm_type = lstm_type
        if lstm_type == "pytorch":
            # Dropout between layers
            self.lstm = nn.LSTM(
                input_size=num_bands,
                hidden_size=hidden_size,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif lstm_type == "unrolled":
            # Dropout between hidden states
            self.lstm = UnrolledLSTM(
                input_size=num_bands,
                hidden_size=hidden_size,
                dropout=dropout,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unknown LSTM type {lstm_type}")
        self.num_bands = num_bands
        self.input_timesteps: Optional[int] = input_timesteps
        self.output_timesteps: Optional[int] = output_timesteps
        self.to_bands = nn.Linear(in_features=hidden_size, out_features=num_bands)

        # Gaussian Process
        self.gp_enabled = gp_enabled
        if self.gp_enabled:
            self.gp_layer = GPRegressionLayer1(batch_size=num_bands * gp_interval)
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
                batch_size=num_bands * gp_interval
            )
            self.gp_inference_indexes = gp_inference_indexes
        self.perturb_h_indexes = perturb_h_indexes
        self.perturb_amount = 1e-4
        self.teacher_forcing = teacher_forcing
        self.gp_interval = gp_interval
        self.device = device
        self.to(device)

    def gp_output_using_x(self, x):
        gp_output_list: List[torch.Tensor] = []
        for i in range(0, x.shape[1] - self.gp_interval, self.gp_interval):
            # Take a slice of timesteps and put them into gp shape
            gp_input = x[:, i : i + self.gp_interval, :]
            gp_input_reshaped = gp_input.view(x.shape[0], -1).transpose(0, 1).unsqueeze(2)
            gp_pred = self.gp_layer(gp_input_reshaped)
            gp_output_list.append(gp_pred)
        return gp_output_list

    def lstm_output_using_x(self, x, hidden_tuple=None):
        lstm_output_list = []
        for i in range(x.shape[1]):
            lstm_input = x[:, i : i + 1, :]
            lstm_pred, hidden_tuple = self.lstm(lstm_input, hidden_tuple)
            if self.lstm_type == "pytorch":
                lstm_pred = self.to_bands(lstm_pred)
            else:
                lstm_pred = self.to_bands(torch.transpose(lstm_pred[0, :, :, :], 0, 1))
            assert lstm_input.shape == lstm_pred.shape
            lstm_output_list.append(lstm_pred)
        return lstm_output_list, hidden_tuple

    def lstm_output_using_auto_regression(self, timesteps, input, hidden_tuple=None):
        lstm_output_list = [input]
        for i in range(timesteps):
            lstm_pred, hidden_tuple = self.lstm(lstm_output_list[-1], hidden_tuple)
            if self.lstm_type == "pytorch":
                lstm_pred = self.to_bands(lstm_pred)
            else:
                lstm_pred = self.to_bands(torch.transpose(lstm_pred[0, :, :, :], 0, 1))
            lstm_output_list.append(lstm_pred)
            with torch.no_grad():
                if not self.training and i in self.perturb_h_indexes:
                    new_h0 = hidden_tuple[0] + (
                        self.perturb_amount * torch.randn_like(hidden_tuple[0]).to(self.device)
                    )
                    new_h1 = hidden_tuple[1] + (
                        self.perturb_amount * torch.randn_like(hidden_tuple[1]).to(self.device)
                    )
                    hidden_tuple = (new_h0, new_h1)

        return lstm_output_list[1:], hidden_tuple

    def forward_train(self, x, input_timesteps):
        """Forward pass for training"""
        seq_length = x.shape[1]
        if self.teacher_forcing:
            lstm_output_list, _ = self.lstm_output_using_x(x[:, :-1])
        else:
            lstm_output_list_using_x, last_hidden_tuple = self.lstm_output_using_x(
                x=x[:, :input_timesteps]
            )
            remaining_timesteps = seq_length - input_timesteps - 1
            lstm_output_list_using_self, _ = self.lstm_output_using_auto_regression(
                timesteps=remaining_timesteps,
                input=x[:, input_timesteps : input_timesteps + 1],
                hidden_tuple=last_hidden_tuple,
            )

            lstm_output_list = [x[:, 0:1]] + lstm_output_list_using_x + lstm_output_list_using_self
        assert len(lstm_output_list) == (seq_length), f"{len(lstm_output_list)} != {(seq_length)}"
        if self.gp_enabled:
            gp_output_list = self.gp_output_using_x(x)
            return torch.cat(lstm_output_list, dim=1), gp_output_list

        return torch.cat(lstm_output_list, dim=1)

    def forward_inference_lstm_only(self, x, input_timesteps):
        """Forward pass using LSTM only"""
        seq_length = x.shape[1]
        remaining_timesteps = seq_length - input_timesteps - 1
        # inference_output_list = [x[:, t : t + 1, :] for t in range(input_timesteps)]
        lstm_output_list, last_hidden_tuple = self.lstm_output_using_x(
            x[:, :input_timesteps], hidden_tuple=None
        )
        inference_output_list = [x[:, 0:1]]
        inference_output_list += lstm_output_list
        lstm_output_list, _ = self.lstm_output_using_auto_regression(
            timesteps=remaining_timesteps,
            input=x[:, input_timesteps : input_timesteps + 1, :],
            hidden_tuple=last_hidden_tuple,
        )

        inference_output_list += lstm_output_list
        assert (
            len(inference_output_list) == seq_length
        ), f"{len(inference_output_list)} != {seq_length}"
        return torch.cat(inference_output_list, dim=1)

    def forward_inference_lstm_gp(self, x, input_timesteps):
        """Forward pass using LSTM and GP"""
        seq_length = x.shape[1]
        lstm_output_list, last_hidden_tuple = self.lstm_output_using_x(
            x[:, :input_timesteps], hidden_tuple=None
        )

        inference_output_list = [x[:, 0:1]] + lstm_output_list
        # inference_output_list = [x[:, t : t + 1, :] for t in range(input_timesteps)]

        # Loop through remaining timesteps and use LSTM and GP to predict the next timestep
        gp_timesteps = [t + input_timesteps for t in self.gp_inference_indexes]
        for i in range(input_timesteps, seq_length, self.gp_interval):

            current_timesteps = list(range(i, i + self.gp_interval))
            if any([t for t in current_timesteps if t in gp_timesteps]):
                # Use last n timesteps as input to GP
                last_n = torch.cat(inference_output_list[i - self.gp_interval : i], dim=1)
                gp_input_reshaped = last_n.view(last_n.shape[0], -1).transpose(0, 1).unsqueeze(2)
                gp_pred = self.likelihood(self.gp_layer(gp_input_reshaped))
                gp_pred_reshaped = (
                    gp_pred.rsample().transpose(0, 1).view(x.shape[0], self.gp_interval, -1)
                )
                assert gp_pred_reshaped.shape == last_n.shape
                gp_output_list = torch.tensor_split(gp_pred_reshaped, self.gp_interval, dim=1)
                # Generate next hidden tuple using GP outputs
                lstm_input = torch.cat(gp_output_list, dim=1)
                _, last_hidden_tuple = self.lstm_output_using_x(lstm_input, last_hidden_tuple)

                inference_output_list += gp_output_list
            else:
                # Use last timestep as input to LSTM
                lstm_output_list, last_hidden_tuple = self.lstm_output_using_auto_regression(
                    timesteps=self.gp_interval,
                    input=inference_output_list[-1],
                    hidden_tuple=last_hidden_tuple,
                )

                inference_output_list += lstm_output_list

        if len(inference_output_list) < seq_length:
            remaining_timesteps = seq_length - len(inference_output_list)
            inference_output_list += self.lstm_output_using_auto_regression(
                remaining_timesteps, inference_output_list[-1], last_hidden_tuple
            )
        elif len(inference_output_list) > seq_length:
            inference_output_list = inference_output_list[:seq_length]

        assert (
            len(inference_output_list) == seq_length
        ), f"{len(inference_output_list)} != {seq_length}"
        return torch.cat(inference_output_list, dim=1)

    def forward(self, x, training=True, perturb_hidden_state=False):

        seq_length = x.shape[1]

        # Randomize input timesteps if a specific index is not provided
        if self.input_timesteps is None:
            input_timesteps = np.random.randint(10, seq_length - 10)
        else:
            input_timesteps = self.input_timesteps

        if training:
            return self.forward_train(x, input_timesteps)

        if self.gp_enabled:
            return self.forward_inference_lstm_gp(x, input_timesteps)
        else:
            return self.forward_inference_lstm_only(x, input_timesteps)
