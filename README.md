<h1 align="center"><b> SAM Hub</b></h1>
<p align="center">
  <i>~ in Pytorch ~</i>
</p> 

## Introduction
This is an **unofficial** repo implements the SAM optimizer and its variants. The following table displays the optimizer and their paper url.

| Optimizer | Official Code | Paper(Conference) | This repo |
| :-------: | :-----------: | :---: | :-------: |
| SAM | https://github.com/google-research/sam | [Sharpness-Aware Minimization for Efficiently Improving Generalization (ICLR2021)](https://arxiv.org/abs/2010.01412) | ... |
| ASAM | https://github.com/SamsungLabs/ASAM | [ASAM: Adaptive Sharpness-Aware Minimization for Scale-Invariant Learning of Deep Neural Networks (ICML2021)](https://arxiv.org/abs/2102.11600) | ... |
| ESAM | https://github.com/dydjw9/Efficient_SAM | [Efficient Sharpness-aware Minimization for Improved Training of Neural Networks (ICLR2022)](https://arxiv.org/abs/2110.03141) | ... |
| GSAM | https://github.com/juntang-zhuang/GSAM | [Surrogate Gap Minimization Improves Sharpness-Aware Training (ICLR2022)](https://arxiv.org/abs/2203.08065) | ... |
| LookSAM | Not Found | [Towards Efficient and Scalable Sharpness-Aware Minimization (CVPR2022)](https://arxiv.org/abs/2203.02714) | ... |

## Usage
### Installation
- Clone this repo
```bash
git clone git@github.com:Mi-Peng/sam-hub.git
cd sam-hub
```
- Create an anaconda virtual envrionment
```bash
conda create -n samhub python=3.8
conda activate samhub
```

- Install requirements
```bash
pip install hydra-core --upgrade
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

- Install wandb (Optional)
```bash
pip install wandb
```

### Data preparation
- Download the ImageNet dataset from http://image-net.org/
- Sort out the dataset file to make sure that the structure of dataset file look like:
```bash
$ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
```
> If you have no idea, the file `setup.sh` would help you.

### Evaluation
```bash
python eval.py 
```

### Training
- Training the default setting in file `configs/your-config-file`.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 1234 --use_env main.py --config-dir configs/ --config-name [your-config-file]
```



## Other Reference
> https://github.com/davda54/sam