from utils.registry import Registry 

MODELS_REGISTRY = Registry("models")

def build_model(cfg):
    model = MODELS_REGISTRY.get(cfg.model.name)(cfg)
    model = model.cuda()
    return model