import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.notebook import tqdm, trange
from tqdm.notebook import tqdm


class MegaSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, M=None, eta2=0.01, rho=0.05, alpha=0.05, trace_penalty=True, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        assert eta2 >= 0.0, f"Invalid eta2, should be non-negative: {eta2}"
        assert alpha >= 0.0, f"Invalid rho, should be non-negative: {alpha}"

        defaults = dict(rho=rho, trace_penalty=trace_penalty, **kwargs)
        super(MegaSAM, self).__init__(params, defaults)

        if M is None:
            self.M = [torch.ones_like(tensor, requires_grad=True) for tensor in self.param_groups[0]['params']]
        else:
            self.M = M
        M_dict = {"params": self.M}
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)  # add M here
        # print(self.param_groups)
        self.base_optimizer_M = base_optimizer([M_dict], **kwargs)

        self.param_groups = [self.base_optimizer.param_groups[0],
                            M_dict]
        self.base_optimizer_M.defaults.update(dict(alpha=alpha, lr=eta2))
        self.defaults.update(self.base_optimizer.defaults)
        self.defaults.update(self.base_optimizer_M.defaults)
        # print(self.param_groups)
        # print(self.defaults)
        # print(self.param_groups)
    def mloss(self, zero_grad=False):
        rho = self.defaults['rho']
        alpha = self.defaults['alpha']
        norm, _ = self._grad_norm()
        M_inv_trace = sum([torch.sum(1 / M) for M in self.M])
        loss = rho * norm + alpha * torch.sqrt(M_inv_trace)
        return loss
        
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm, grads_list = self._grad_norm()
        scale = self.param_groups[0]['rho'] / (grad_norm + 1e-12)  # torch.eps
        M_inv = [1 / M for M in self.M]
        for num, p in enumerate(self.param_groups[0]["params"]):
            if p.grad is None: continue
            self.state[p]["old_p"] = p.data.clone()

            # M_inv_flat_chunk = M_inv[:torch.numel(p.grad)]

            # M_inv_chunk = M_inv_flat_chunk.reshape(p.grad.size())
            p.add_(scale * M_inv[num] * p.grad)

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for p in self.param_groups[0]["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()
    
    @torch.no_grad()
    def m_step(self, zero_grad=False):
      alpha = self.defaults['alpha']
    #   eta2 = self.param_groups[1]['eta2']
      grad_norm, _ = self._grad_norm()
    #   M_inv = 1 / self.M
      #for M in self.param_groups[1]:
      #  M.grad
      self.base_optimizer_M.step()
      if zero_grad: self.zero_grad()
  


    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
        self.first_step(zero_grad=False)
        self.m_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        
        M_inv = [1 / M for M in self.M]
        grads_list = [ 
                  p.grad.to(shared_device)   # removed p=2 norm here
                  for p in self.param_groups[0]["params"]
                  if p.grad is not None
                    ]
        # grads_flattened = torch.cat([
        #                 item.flatten().to(shared_device)   # removed p=2 norm here
        #                 for item in grads_list
        #             ])
        norm = torch.sqrt(sum([torch.sum(M_inv[num] * grads_list[num]**2) for num in range(len(M_inv))]))
               
        return norm, grads_list

    def _reshape(self, my_item, target):
        #print(f"target: {target}")
        target_shapes = [i.size() for i in target]
        target_sizes = [torch.numel(i) for i in target]
        #print(f"target  sizes: {sum(target_sizes)}, shapes {target_shapes}")
        #print(f"len of my item: {torch.numel(my_item)}")
        #assert torch.numel(my_item) == sum(target_sizes)

        chunked_item = torch.tensor_split(my_item, tuple(np.cumsum(target_sizes))[:-1])
        reshaped_item = [item.reshape(target_shapes[i]) for i, item in enumerate(chunked_item)]
        return reshaped_item

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups



