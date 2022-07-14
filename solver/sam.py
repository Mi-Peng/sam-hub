import torch
from torch.optim import Optimizer
from utils.configurable import configurable

from solver.build import OPTIMIZER_REGISTRY
from solver.utils import Sync_Perturbation

@OPTIMIZER_REGISTRY.register()
class SAM(Optimizer):
    @configurable
    def __init__(self, params, base_optimizer, rho):
        assert isinstance(base_optimizer, torch.optim.Optimizer), f"base_optimizer must be an `Optimizer`"
        self.base_optimizer = base_optimizer

        assert 0 <= rho, f"rho should be non-negative:{rho}"
        self.rho = rho
        
        super(SAM, self).__init__(params, dict(rho=rho))
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho

    @classmethod
    def from_config(cls, cfg):
        return {
            "rho": cfg.optimizer.rho
        }
    
    @torch.no_grad()
    def perturb_weights(self):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-7)
            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale
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
        **kwargs,
    ):
        assert closure_first is not None, "SAM-type requires closure, which is not provided."
        assert closure_second is not None, "SAM-type requires closure, which is not provided."

        with Sync_Perturbation(model, sync_perturbation):
            with torch.enable_grad():
                loss, output = closure_first()
            self.perturb_weights()
            with torch.enable_grad():
                closure_second()
            self.unperturb_weights()
        
        self._sync_grad()
        self.base_optimizer.step()
        return loss, output

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
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