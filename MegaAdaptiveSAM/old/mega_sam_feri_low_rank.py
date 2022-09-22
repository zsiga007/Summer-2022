import torch
import numpy as np


class MegaSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, lr_M=0.01, rho=0.05, alpha=np.sqrt(1 / 254), trace_penalty=True, **kwargs):
        if not rho >= 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")
        if not lr_M >= 0.0:
            raise ValueError(f"Invalid eta2, should be non-negative: {lr_M}")
        if not alpha >= 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {alpha}")

        self.trace_penalty = trace_penalty
        self.alpha = alpha
        self.rho = rho

        defaults = dict(lr_M=lr_M, **kwargs)
        super(MegaSAM, self).__init__(params, defaults)

        num_params = 0
        for group in self.param_groups:
            for tensor in group["params"]:
                num_params += torch.numel(tensor)
        #num_params = torch.sum([torch.sum([tensor.size() for tensor in param_group["params"]]) for param_group in self.param_groups])

        self.shared_device = self.param_groups[0]["params"][0].device
        A = torch.ones((5, num_params), requires_grad=True, device=self.shared_device)
        torch.nn.init.normal_(A,  mean = 1.0, std = 1)
        #A = torch.normal(mean= 0.3 * torch.ones((num_params, 5)), std= 0.1 * torch.ones((num_params, 5)), requires_grad=True).to(self.shared_device)
        D = torch.ones(num_params, requires_grad=True).to(self.shared_device)
        self.M_param_groups = [{'params': [A, D], 'lr': lr_M}]
   

        self.base_optimizer = base_optimizer(
            self.param_groups + self.M_param_groups, **kwargs)

        self.eps = max(torch.finfo(
            self.param_groups[0]['params'][0].dtype).eps, 1e-12)

    def mloss(self):
        squared_norm, _ = self._grad_norm()
        return self.rho * torch.sqrt(squared_norm)

    def mpenalty(self):
        A = self.M_param_groups[0]['params'][0]
        D = self.M_param_groups[0]['params'][1]
        #trace_Minv = torch.sum(1/D) - torch.trace(A.T @ torch.inverse((torch.eye(5) + (A / D) @ A.T)) @ A @ torch.diag(1/D**2))
        X = torch.linalg.solve((torch.eye(5) + (A / D) @ A.T), A)
        trace_Minv = torch.sum(1/D) - torch.trace(A.T @ X @ torch.diag(1/D**2))
       
        logdet_M = torch.log(torch.determinant(torch.eye(5) + (A / D) @ A.T) * torch.prod(D))
        return self.alpha * trace_Minv + 1 * self.alpha * logdet_M

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        squared_norm, grads = self._grad_norm()
        scale = self.rho / (torch.sqrt(squared_norm) + self.eps)
        A = self.M_param_groups[0]['params'][0]
        D = self.M_param_groups[0]['params'][1]

        eps = grads / D - A.T @ torch.inverse((torch.eye(5) + A @ A.T / D)) @ (A @ grads.T) / (D**2)
        #SOLVE_link_slack
        for group in self.param_groups:
          for p in group["params"]:
            if p.grad is None: continue
            self.state[p]["old_p"] = p.data.clone()
            eps_chunk = eps[:torch.numel(p)]
            eps = eps[torch.numel(p):]
            eps_reshaped = eps_chunk.reshape(torch.shape(p))
            p.add_(scale * eps_reshaped)


        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self):
        for p in self.param_groups[0]["params"]:
            if p.grad is None:
                continue
            p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        # do the actual "sharpness-aware" update, this updates M as well.
        self.base_optimizer.step()

    @torch.no_grad()
    def step(self, closure):
        self._zero_M_grad()
        with torch.enable_grad():
            penalized_mloss = self.mloss() + self.mpenalty()
            penalized_mloss.backward()
        self.first_step(zero_grad=True)
        with torch.enable_grad():
            closure()
        self.second_step()

    def _grad_norm(self):
        A = self.M_param_groups[0]['params'][0]
        D = self.M_param_groups[0]['params'][1]
        #M_inv = 1 / D - A.T @ (torch.eye(5) + A @ A.T / D).inv() @ A / (D**2)
        tensor_list = []
        for param_group in self.param_groups:
            for param in param_group['params']:
                if param.grad is None:
                    continue
                tensor_list.append(torch.flatten(param.grad))
        grads = torch.cat(tensor_list).detach().to(self.shared_device)
        print(f"A shape {A.size()}")
        print(f"D shape {D.size()}")
        print(f"grads shape {grads.size()}")
        print(f"Inv size: {torch.inverse((torch.eye(5) + (A / D) @ A.T ))}")
        squared_norm = (grads ** 2) / D - (grads @ A.T) @ torch.inverse((torch.eye(5) + (A / D) @ A.T )) @ (A @ grads.T) / (D**2)

        return squared_norm, grads

    def _zero_M_grad(self, set_to_none=False):
        for matrix in self.M_param_groups[0]["params"]:
            if set_to_none:
                matrix.grad = None
            else:
                if matrix.grad is not None:
                    torch.zero_(matrix.grad)

