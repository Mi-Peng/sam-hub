import torch
import numpy as np
import matplotlib.pyplot as plt

import platform

def vis_image(image_tensor: torch.Tensor, image_file: str):
    assert isinstance(image_tensor, torch.Tensor) and image_tensor.ndim  == 3
    image_numpy = image_tensor.numpy()
    image = np.transpose(image_numpy, (1, 2, 0))
    if platform.system().lower() == 'windows':
        plt.imshow(image)
        plt.show()
    elif platform.system().lower() == 'linux':
        plt.imsave(image_file, image)