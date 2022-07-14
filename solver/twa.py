import os
from re import L
import torch
from torch.optim import Optimizer
import numpy as np

from utils.configurable import configurable

from solver.build import OPTIMIZER_REGISTRY


@OPTIMIZER_REGISTRY.register()
class TWA(Optimizer):
    @configurable
    def __init__(self, params, base_optimizer, start_id, end_id, model, model_dir):
        super(TWA, self).__init__(params)
        self.base_optimizer = base_optimizer
        self.start_id = start_id
        self.end_id = end_id
        self.model = model
        
        W = self.load_model_get_W(model_dir)
        P = self.Schmidt_subspace(W)
        self.P = P

    @classmethod
    def from_config(cls, cfg):
        return {
            
        }

    def Schmidt_subspace(self, W):
        P = torch.from_numpy(np.array(W)).cuda() # P: [n, w]
        n, w = P.size()
        for i in range(n):
            if i > 0:
                angle = torch.mm(P[:i,:], P[i,:].reshape(-1, 1)) # [i, w] x [w, 1] = [i, 1]
                P[i,:] -= torch.mm(angle.reshape(1, -1), P[:i,:]) # [1, i] x [i, w] = [1, w]
            P[i,:] /= torch.linalg.vector_norm(P[i,:])
        return P

    def load_model_get_W(self, model_dir):
        def get_model_param_vec(model):
            """
            Return model parameters as a vector
            """
            vec = []
            for name,param in model.named_parameters():
                vec.append(param.detach().cpu().numpy().reshape(-1))
            return np.concatenate(vec, 0)
        W = []
        for sample_id in range(self.start_id, self.end_id):
            self.model.load_state_dict(torch.load(os.path.join(model_dir, 'checkpoint_{}.pth'.format(sample_id))))
            W.append(get_model_param_vec(self.model))
        W = np.array(W)
        return W

    @torch.no_grad()
    def bn_update(self, model):
        pass

    @torch.no_grad()
    def step(self):
        grad_vec = []
        for name, param in self.model.named_parameters():
            grad_vec.append(param.grad.data.detach().reshape(-1))
        grad_vec = torch.cat(grad_vec, 0) # [, w]

        grad_projection = torch.mm(self.P.T, torch.mm(self.P, grad_vec.reshape(-1, 1)))

        idx = 0
        for name, param in self.model.named_parameters():
            param_shape = param.data.shape
            param_size = param_shape.numel()
            param.grad.data = grad_projection[idx: idx+param_size].reshape(param_shape)
            idx += param_size
        
        self.base_optimizer.step()