import torch
from torch.optim import Optimizer
from abc import ABC, abstractmethod
from torch.nn.init import constant_

class MeanFieldOptimizer(Optimizer, ABC):
    """Abstract base class for Mean-Field Type Optimizers, including
    Mean-Field Variational Inference or Variational SAM.

    Subclasses must implement the get_perturbation method that returns
    a D-dimensional parameter vector of norm sqrt(D), shaped in the same
    format as in param_groups."""

    def __init__(self, params, base_optimizer, lr_Sigma = 0.01, sigma_prior = 1, init_scale_M = 0.1, kl_div_weight = 0.01, **kwargs):
        if not lr_Sigma >= 0.0:
            raise ValueError(f"Invalid lr_Sigma, should be non-negative: {lr_Sigma}")
        if not sigma_prior > 0.0:
            raise ValueError(f"Invalid sigma_prior, should be positive: {sigma_prior}")
        if not kl_div_weight >= 0.0:
            raise ValueError(f"Invalid kl_div_weight, should be positive: {kl_div_weight}")
          
        self.sigma_prior = sigma_prior
        self.kl_div_weight = kl_div_weight

        defaults = dict(lr_Sigma=lr_Sigma, **kwargs)
        super(MeanFieldOptimizer, self).__init__(params, defaults)

        self.M_param_groups = []
        for param_group in self.param_groups:
            M_param_group = param_group.copy()
            M_param_group["params"] = [
                torch.ones_like(tensor, requires_grad=tensor.requires_grad)
                for tensor in param_group['params']
            ]
            for M in M_param_group["params"]:
                constant_(M, init_scale_M)
            M_param_group['lr'] = M_param_group['lr_Sigma']
            M_param_group.pop('lr_Sigma')
            param_group.pop('lr_Sigma')
            self.M_param_groups.append(M_param_group)

        self.base_optimizer = base_optimizer(self.param_groups + self.M_param_groups, **kwargs)

        self.eps = max(torch.finfo(
            self.param_groups[0]['params'][0].dtype).eps, 1e-12)

        self.shared_device = self.param_groups[0]["params"][0].device


    def step(self, closure):        
        self._populate_gradients_for_mean(closure)
        self._populate_gradients_for_Sigma()
        self.base_optimizer.step()

    @torch.no_grad()
    def _populate_gradients_for_mean(self, closure):
        """This function populates the gradients of the mean parameter in
        `param_groups`. It first saves the original parameter values for later,
        applies the perturbation, zeroes out the gradients, calls the closure to
        backpropagate, then restores the mean parameters to their original values.
        """

        self.perturbation_groups = self._get_perturbation()

        for param_group, M_param_group, perturbation_group in zip(self.param_groups, self.M_param_groups, self.perturbation_groups):
            for param, M, perturbation in zip(param_group['params'], M_param_group['params'], perturbation_group['params']):
                if param.requires_grad:
                    self.state[param]["old_p"] = param.data.clone()
                    param.add_(torch.abs(M)*perturbation)
        
        self.base_optimizer.zero_grad()

        with torch.enable_grad():
          closure()
        
        
        for param_group in self.param_groups:
            for param in param_group["params"]:
                if param.requires_grad:
                    param.data = self.state[param]["old_p"]
        

    @torch.no_grad()
    def _populate_gradients_for_Sigma(self):
        """This function computes the gradients with respect to the field parameters
        in `M_param_group`. It does so by multiplying the corresponding mean parameter
        gradients by the perturbations, and adding this value to the gradient of the KL
        divergence."""

        with torch.enable_grad():
            kl_div = torch.tensor(0.0)
            for M_param_group in self.M_param_groups:
                for M in M_param_group['params']:
                    if M.requires_grad:
                        kl_div+= (M**2).sum()/2/self.sigma_prior**2 + torch.log(torch.abs(M)).sum()
            (self.kl_div_weight * kl_div).backward()

        for param_group, M_param_group, perturbation_group in zip(self.param_groups, self.M_param_groups, self.perturbation_groups):
            for param, M, perturbation in zip(param_group['params'], M_param_group['params'], perturbation_group['params']):
                if param.requires_grad:
                    M.grad.add_(param.grad * perturbation * torch.sign(M))


    def zero_grad(self):
        self.base_optimizer.zero_grad()

    @abstractmethod
    def _get_perturbation(self):
        """Abstract method all subclasses must implement for calculating the
        perturbation direction in the Mean-Field optimization algorithm."""
        pass


class MFVI(MeanFieldOptimizer):
    """Implements Mean Field Variational Optimization.
    Args:

        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        base_optimizer (torch.optim.Optimizer): base optimizer to make gradient
            updates to mean and field parameters
        lr (float, optional): learning rate for the mean parameter
            (default: 1e-3)
        lr_Sigma (float, optional): learning rate for the field parameter
            (default: 1e-3)
        sigma_prior (float, optional): standard deviation of the prior, controls
            weight decay of mean parameter and strength of regularisation on
            the field parameter (default: 10)
        kl_div_weight (float, optional): weight given to the KL divergence term.
            Should be chosen roughly proportional to batchsize/N where N is the
            total number of datasets (default: 0.01), assuming loss function
            calculates mean log loss.
        """
    def _get_perturbation(self):
        """Calculates a standard normal perturbation for each parameter."""
        perturbation_groups = []

        for param_group in self.param_groups:
            perturbation_group = {'params': []}
            for param in param_group['params']:
                if param.requires_grad:
                  perturbation_group['params'].append(torch.randn_like(param))
                else:
                  perturbation_group['params'].append(None)
            perturbation_groups.append(perturbation_group)

        return perturbation_groups


class VariationalSAM(MeanFieldOptimizer):

    def _get_perturbation(self):
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
