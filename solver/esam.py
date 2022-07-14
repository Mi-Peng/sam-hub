import random
import torch
import torch.nn.functional as F
from utils.configurable import configurable

from solver.build import OPTIMIZER_REGISTRY
from solver.utils import Sync_Perturbation

@OPTIMIZER_REGISTRY.register()
class ESAM(torch.optim.Optimizer):
    @configurable
    def __init__(self, params, base_optimizer, rho=0.05, beta=1.0, gamma=1.0, **kwargs):
        assert isinstance(base_optimizer, torch.optim.Optimizer), f"base_optimizer must be an `Optimizer`"
        self.base_optimizer = base_optimizer
        
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        assert 0.0 <= beta <= 1.0, f"Invalid beta, should between 0 and 1: {beta}"
        assert 0.0 <= gamma <= 1.0, f"Invalid gamma, should between 0 and 1: {gamma}"
        self.rho = rho
        self.beta = beta
        self.gamma = gamma

        super(ESAM, self).__init__(params, dict(rho=rho))
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
            group["beta"] = beta
            group["gamma"] = gamma

    @classmethod
    def from_config(cls, cfg):
        return {
            "rho": cfg.optimizer.rho,
            "beta": cfg.optimizer.beta,
            "gamma": cfg.optimizer.gamma,
        }


    @torch.no_grad()
    def perturb_weights(self):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7)
            for p in group["params"]:
                if p.grad is None: continue

                prob = torch.rand(p.shape, device=p.device, dtype=torch.double)
                mask = torch.where(prob > self.beta, 1., 0.).float()

                e_w = p.grad * scale * mask
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def unperturb_weights(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"
            
    @torch.no_grad()
    def step(
        self, 
        closure_first = None, 
        closure_second = None,
        *,
        sync_perturbation = False,
        model = None,
        criterion = None,
        full_images = None, full_targets = None,
        **kwargs
    ):
        assert closure_first is not None, "SAM-type requires closure, which is not provided."
        assert closure_second is not None, "SAM-type requires closure, which is not provided."
        
        with Sync_Perturbation(model, sync_perturbation):
            with torch.enable_grad():
                loss_before, output = closure_first(per_data=True)
            self.perturb_weights()

            loss_after = criterion(model(full_images), full_targets)
            sharpness_per_data = loss_after - loss_before
            position = int(len(full_targets) * self.gamma)
            threshold, _ = torch.topk(sharpness_per_data, position)
            threshold = threshold[-1]
            indices = [sharpness_per_data > threshold]
            with torch.enable_grad():
                closure_second(indices=indices)
            self.unperturb_weights()
        
        self._sync_grad()
        self.base_optimizer.step()
        return loss_before.mean(), output
        
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        #original sam 
                        # p.grad.norm(p=2).to(shared_device)
                        #asam 
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm
    
    def _sync_grad(self):
        if torch.distributed.is_initialized(): # synchronize final gardients on weight update
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)
        return