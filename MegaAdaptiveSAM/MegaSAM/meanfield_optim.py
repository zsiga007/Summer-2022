import torch
from torch.optim import Optimizer
from abc import ABC, abstractmethod

class MeanFieldOptimizer(Optimizer, ABC):
"""Abstract base class for Mean-Field Type Optimizers, including
Mean-Field Variational Inference or Variational SAM.

Subclasses must implement the get_perturbation method that returns
a D-dimensional parameter vector of norm sqrt(D), shaped in the same
format as in param_groups.
"""

def __init__(self, params, base_optimizer, lr_sigma=0.01, sigma_prior = 0.01, **kwargs):
    {rho}")
        if not lr_Sigma >= 0.0:
            raise ValueError(f"Invalid lr_Sigma, should be non-negative: {lr_Sigma}")
        if not sigma_prior > 0.0:
            raise ValueError(f"Invalid lr_Sigma, should be positive: {lr_Sigma}")

        self.sigma_prior = sigma_prior

        defaults = dict(lr_Sigma=lr_Sigma, **kwargs)
        super(MeanFieldOptimizer, self).__init__(params, defaults)

        self.M_param_groups = []
        for param_group in self.param_groups:
            M_param_group = param_group.copy()
            M_param_group["params"] = [
                torch.ones_like(tensor, requires_grad=True) 
                for tensor in param_group['params']
                ]
            M_param_group['lr'] = M_param_group['lr_Sigma']
            M_param_group.pop('lr_Sigma')
            param_group.pop('lr_Sigma')
            self.M_param_groups.append(M_param_group)

        self.base_optimizer = base_optimizer(
            self.param_groups + self.M_param_groups, **kwargs)

        self.eps = max(torch.finfo(
            self.param_groups[0]['params'][0].dtype).eps, 1e-12)

        self.shared_device = self.param_groups[0]["params"][0].device


    def step(self, closure):        
        self.populate_gradients_for_mean(closure)
        self.populare_gradients_for_Sigma()
        self.base_optimizer.step()


    def _populate_gradients_for_mean(self, closure()):
        """This function populates the gradients of the mean parameter in
        `param_groups`. It first saves the original parameter values for later,
        applies the perturbation, zeroes out the gradients, calls the closure to
        backpropagate, then restores the mean parameters to their original values.
        """

        self.perturbation_groups = self.get_perturbation()

        for param_group, M_param_group, perturbation_group in zip(self.param_groups, self.M_param_groups, self.perturbation_groups):
            for param, M, perturbation in zip(param_group['params'], M_param_group['params'], perturbation_group['params']):
                if param.grad is None:
                    continue
                self.state[param]["old_p"] = param.data.clone()
                param._add(totch.abs(M)*perturbation)
        
        self.base_optimizer.zero_grads()

        closure()
        
        for p in self.param_groups[0]["params"]:
            if p.grad is None:
                continue
            p.data = self.state[p]["old_p"]
        

    def _populate_gradients_for_Sigma(self):
        """This function computes the gradients with respect to the field parameters
        in `M_param_group`. It does so by multiplying the corresponding mean parameter
        gradients by the perturbations, and adding this value to the gradient of the KL
        divergence."""

        
        kl_div = 0 #TODO:implement KL-divergence
        kl_div.backward()

        for param_group, M_param_group, perturbation_group in zip(self.param_groups, self.M_param_groups, self.perturbation_groups):
            for param, M, perturbation in zip(param_group['params'], M_param_group['params'], perturbation_group['params']):
                if param.grad is None:
                    continue
                M.grad._add(param.grad*perturbation)

    @abstractmethod
    def get_perturbation(self):
        """Abstract method all subclasses must implement for calculating the
        perturbation direction in the Mean-Field optimization algorithm."""
        pass


class MFVI(MeanFieldOptimizer):

    def get_perturbation(self):
        """Calculates a standard normal perturbation for each parameter."""
        perturbation_groups = []

        for param_group in self.param_groups:
            perturbation_group = {'params': []}
            for param in param_group['params']:
                if param.grad is None:
                    continue
                perturbation_group['params'].append(torch.randn_like(param))
            perturbation_groups.append(perturbation_group)

        return perturbation_groups


class VariationalSAM(MeanFieldOptimizer):

    def get_perturbation(self):

        perturbation_groups = []
        squared_norm = torch.tensor(0.0)
        num_params = 0

        for param_group, M_param_group in zip(self.param_groups, self.M_param_groups):
            perturbation_group = {'params':[]}
            for param, M in zip(param_group['params'], M_param_group['params']):
                if param.grad is None:
                    continue
                perturbation = M*param.grad
                squared_norm._add((perturbation**2).sum())
                num_params+=torch.numel(perturbation)
                perturbation_group['params'].append(perturbation)
            perturbation_groups.append(perturbation_group)

        scale = torch.sqrt(num_params / squared_norm)

        for perturbation_group in perturbation_groups:
            for perturbation in perturbation_group:
                perturbation*= scale

        return perturbation_groups
