import torch
import numpy as np


class LagrangeOpt(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, lr_2=0.01, lr_3=0.01, rho=1.,  **kwargs):
        if not lr_2 >= 0.0:
            raise ValueError(f"Invalid eta2, should be non-negative: {lr_2}")
        if not lr_3 >= 0.0:
            raise ValueError(f"Invalid eta2, should be non-negative: {lr_3}")

        self.rho = rho

        defaults = dict(lr_2=lr_2, lr_3=lr_3, **kwargs)
        super(LagrangeOpt, self).__init__(params, defaults)

        self.tilde_param_groups = []
        for param_group in self.param_groups:
            tilde_param_group = param_group.copy()
            tilde_param_group['params'] = [torch.ones_like(
                tensor, requires_grad=True) for tensor in param_group['params']]
            tilde_param_group['lr'] = tilde_param_group['lr_2']
            tilde_param_group.pop('lr_2')
            param_group.pop('lr_2')
            self.tilde_param_groups.append(tilde_param_group)
        
        #lambda is a number
        self.lambd_param_groups = [{"params": [torch.tensor([1], requires_grad=True)], "lr": lr_3}] 

        self.base_optimizer = base_optimizer(
            self.param_groups + self.tilde_param_groups + self.lambd_param_groups, **kwargs)

        self.eps = max(torch.finfo(
            self.param_groups[0]['params'][0].dtype).eps, 1e-12)

        self.shared_device = self.param_groups[0]["params"][0].device

    #next two steps take care of the theta equation (i.e. substitute tilde into thet theta grad of the loss)
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        for theta_param_group, tilde_param_group in zip(self.theta_param_groups, self.tilde_param_groups):
            for theta, tilde in zip(theta_param_group['params'], tilde_param_group['params']):
                if theta.grad is None:
                    continue
                self.state[theta]["old_theta"] = theta.data.clone()
                theta = tilde
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self):
        for theta in self.theta_param_groups[0]["params"]:
            if theta.grad is None:
                continue
            theta.data = self.state[theta]["old_theta"]  #get back to take grad wrt theta"

        self.base_optimizer.step()
    
    @torch.no_grad()
    def tilde_step(self):
        for theta, tilde in zip(self.theta_param_groups[0]["params"], self.tilde_param_groups[0]["params"]):
            if tilde.grad is None:
                continue
            #ez itt tuti nem jó (sztem ez theta gradja a theta helyen), de valami ilyesmit kéne
            tilde += self.tilde_param_groups[0]["lr"] * theta.grad

    def tildeloss(self):
        """Theta grad of loss at tilde missing.
        Since the gradient is wrt to theta not tilde, I think this term should be
        added to the new tilde after the optimizer update. See tilde_step"""
        return - self.lambd_param_groups[0]["params"].detach().to(self.shared_device) * self.dist()
    
    def lambdloss(self):
        # dist here is a constant so need to detach
        dist = self.dist().detach().to(self.shared_device)
        return torch.tanh(self.rho - dist)

    @torch.no_grad()
    def step(self, closure):
        with torch.enable_grad():
            tildeloss = self.tildeloss()
            tildeloss.backward()

            lambdloss = self.lambdloss()
            lambdloss.backward()

        self.first_step(zero_grad=True)
        with torch.enable_grad():
            closure()
        self.second_step()
        self.tilde_step()

    def dist(self):
        dist = torch.tensor(0.0, device=self.shared_device)
        for theta_param_group, tilde_param_group in zip(self.param_groups, self.tilde_param_groups):
            for theta, tilde in zip(theta_param_group['params'], tilde_param_group['params']):
                # dist is constant in theta, so need to detach
                theta = theta.detach().to(self.shared_device)
                dist.add_(torch.norm(theta-tilde, p=2))
        return dist
