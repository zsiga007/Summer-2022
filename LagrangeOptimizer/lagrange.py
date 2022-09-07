import torch
import numpy as np


class LagrangeOpt(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, lr_2=0.05, lr_3=0.05, rho=0.05,  **kwargs):
        if not lr_2 >= 0.0:
            raise ValueError(f"Invalid learning rate 2, should be non-negative: {lr_2}")
        if not lr_3 >= 0.0:
            raise ValueError(f"Invalid learning rate 3, should be non-negative: {lr_3}")

        self.rho = rho

        defaults = dict(lr_2=lr_2, lr_3=lr_3, **kwargs) #user wont change them
        super(LagrangeOpt, self).__init__(params, defaults)

        #initialize tilde param,should have the same structure as theta
        self.tilde_param_groups = []
        for param_group in self.param_groups:
            tilde_param_group = param_group.copy()
            tilde_param_group['params'] = [
                torch.tensor(tensor + 
                torch.nn.init.normal_(torch.ones_like(tensor), mean=0.0, std=1.0), requires_grad=True)
                for tensor in param_group['params']]
            tilde_param_group['lr'] = tilde_param_group['lr_2']
            tilde_param_group.pop('lr_2')
            param_group.pop('lr_2')
            self.tilde_param_groups.append(tilde_param_group)
        
        #lambda is a number, it is initialized here
        self.lambd_param_groups = [{"params": [torch.tensor([0.0])], "lr": lr_3}] 

        self.base_optimizer = base_optimizer(
            self.param_groups + self.tilde_param_groups + self.lambd_param_groups, **kwargs)

        self.eps = max(torch.finfo(
            self.param_groups[0]['params'][0].dtype).eps, 1e-12)

        self.shared_device = self.param_groups[0]["params"][0].device

    #next two steps take care of the theta equation (i.e. substitute tilde into thet theta grad of the loss)
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        for theta_param_group, tilde_param_group in zip(self.param_groups, self.tilde_param_groups):
            for theta, tilde in zip(theta_param_group['params'], tilde_param_group['params']):
                if theta.grad is None:
                    continue
                self.state[theta]["old_theta"] = theta.data.clone()
                theta.data = tilde.data
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self):
        for theta in self.param_groups[0]["params"]:
            if theta.grad is None:
                continue
            theta.data = self.state[theta]["old_theta"]  #get back to take grad wrt theta"

        self.base_optimizer.step()

    def tildeloss(self, closure):
        #here we were trying to avoid nans
        return - closure()[0]/(self.dist().detach()+self.eps) + 0.000001*self.dist().detach()*torch.abs(self.lambd_param_groups[0]["params"][0]).detach().to(self.shared_device) * self.dist()
           
    def lambdloss(self):
        # dist here is a constant so need to detach
        dist = self.dist().detach().to(self.shared_device)
        return torch.tanh(self.rho - dist)

    @torch.no_grad()
    def step(self, closure):
        self.first_step(zero_grad=True)
        with torch.enable_grad():
            tildeloss = self.tildeloss(closure)
            tildeloss.backward(retain_graph=True)
        self.second_step()
        #step with lambda, this does not require gradients
        self.lambd_param_groups[0]["params"][0] = self.lambd_param_groups[0]["params"][0].to(self.shared_device) - self.lambd_param_groups[0]["lr"] * self.lambdloss()
        print(f"Distance: {self.dist()}")

    def dist(self):
        "so far we are using euclidean distance"
        dist = torch.tensor(0.0, device=self.shared_device)
        for theta_param_group, tilde_param_group in zip(self.param_groups, self.tilde_param_groups):
            for theta, tilde in zip(theta_param_group['params'], tilde_param_group['params']):
                # dist is constant in theta, so need to detach
                theta = theta.detach().to(self.shared_device)
                dist.add_(torch.norm(theta-tilde, p=2)**2)
        #print(f"Distance: {torch.sqrt(dist)}")
        return torch.sqrt(dist)
