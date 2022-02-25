from torch import nn
from typing import List, Optional, Tuple

import gpytorch
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
