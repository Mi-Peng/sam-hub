import torch
from torch.optim import Optimizer
from utils.configurable import configurable

from solver.build import OPTIMIZER_REGISTRY
from solver.utils import Sync_Perturbation



@OPTIMIZER_REGISTRY.register()
class LookSAM(Optimizer):
    @configurable
    def __init__(self, params, base_optimizer, rho: float, k: int, alpha: float) -> None:
        assert isinstance(base_optimizer, torch.optim.Optimizer), f"base_optimizer must be an `Optimizer`"
        self.base_optimizer = base_optimizer

        assert 0 <= rho, f"rho should be non-negative:{rho}"
        assert 0 < k and isinstance(k, int), f"k should be non-negative and integer:{k}"
        assert 0 <= alpha, f"alpha should be non-negative:{alpha}"

        self.rho = rho
        self.k = k 
        self.alpha = alpha
        super(LookSAM, self).__init__(params, dict(rho=rho))
        self.param_groups = self.base_optimizer.param_groups
        for group in self.param_groups:
            group["rho"] = rho
            group["k"] = k
            group["alpha"] = alpha

        self.grad_norm = {}

    @classmethod
    def from_config(cls, cfg):
        return {
            "rho": cfg.optimizer.rho,
            "k": cfg.optimizer.k,
            "alpha": cfg.optimizer.alpha,
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
        global_step = None,
        **kwargs,
    ):
        assert closure_first is not None, "SAM-type requires closure, which is not provided."
        assert closure_second is not None, "SAM-type requires closure, which is not provided."

        with Sync_Perturbation(model, sync_perturbation):
            with torch.enable_grad():
                loss, output = closure_first()
            self.grad_norm['sgd'] = self._grad_norm()
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    self.state[p]["sgd_grad"] = p.grad.clone().detach()
            
            if global_step % self.k == 0:
                self.perturb_weights()
                with torch.enable_grad():
                    closure_second()
                self.update_gv()
                self.unperturb_weights()
            else:
                self.reuse_gv()
        self._sync_grad()
        self.base_optimizer.step()
        return loss, output

    @torch.no_grad()
    def update_gv(self):
        self.grad_norm['sam'] = self._grad_norm()
        cosine_theta = self._cosine_theta_sgdsam()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["vertical_grad"] = p.grad.clone().detach() - self.grad_norm['sam'] * cosine_theta * self.state[p]['sgd_grad'] / self.grad_norm['sgd']
    
    @torch.no_grad()
    def reuse_gv(self):
        '''
            p.grad is the gradient of SGD. Should be after `closure_first`.
        '''
        norm_of_vertical = self._vertical_norm()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                vertical_grad = self.state[p]['vertical_grad'] * self.grad_norm['sgd'] / norm_of_vertical
                p.grad.data.add_(vertical_grad, alpha=self.alpha)

    @torch.no_grad()
    def _cosine_theta_sgdsam(self):
        '''
            Calculate the cosine theta, between the gradient of sgd and the gradient of sam. Should be after `closure_second`.
        '''
        dot_production = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                dot_production += torch.sum(self.state[p]["sgd_grad"] * p.grad.clone().detach())
        return dot_production / (self.grad_norm['sgd'] * self.grad_norm['sam'])


    @torch.no_grad()
    def _vertical_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        self.state[p]["vertical_grad"].norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

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




'''
    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        assert closure is not None, "SAM-type requires closure, which is not provided."
        global_step = kwargs["global_step"]

        self.norm['sgd'] = self._grad_norm()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                self.state[p]["sgd_grad"] = p.grad.clone().detach()

        if global_step % self.k == 0:
            self.first_step(True)  # calculate epsilon
            with torch.enable_grad():
                closure() # get the gradient of `w + eps`
            self.update_gv()
        else:
            self.reuse_gv()
        self.second_step(backew=(global_step % self.k == 0))
'''