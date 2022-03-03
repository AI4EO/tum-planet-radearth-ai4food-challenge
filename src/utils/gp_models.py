import gpytorch
import torch
from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

import pdb

# GP Layer
class GPRegressionLayer1(AbstractVariationalGP):
    def __init__(self, batch_size=90):
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=40, batch_size=batch_size
        )
        inducing_points = torch.rand(batch_size, 40, 1)
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(GPRegressionLayer1, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_size=batch_size)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_size=batch_size), batch_size=batch_size
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
