cuda1=0,1,2,3
cuda2=4,5,6,7
port1=1289
port2=9821
cuda_use=$cuda1
port_use=$port1
# configyaml="config_imagenet.yaml"
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 1289 --use_env main.py --config-dir configs/ --config-name $configyaml
configyaml="config_imagenet_gsam.yaml"
CUDA_VISIBLE_DEVICES=$cuda_use python -m torch.distributed.launch --nproc_per_node 4 --master_port $port_use --use_env main.py --config-dir configs/ --config-name $configyaml optimizer.rho=0.05
CUDA_VISIBLE_DEVICES=$cuda_use python -m torch.distributed.launch --nproc_per_node 4 --master_port $port_use --use_env main.py --config-dir configs/ --config-name $configyaml optimizer.rho=0.05
CUDA_VISIBLE_DEVICES=$cuda_use python -m torch.distributed.launch --nproc_per_node 4 --master_port $port_use --use_env main.py --config-dir configs/ --config-name $configyaml optimizer.rho=0.05
CUDA_VISIBLE_DEVICES=$cuda_use python -m torch.distributed.launch --nproc_per_node 4 --master_port $port_use --use_env main.py --config-dir configs/ --config-name $configyaml optimizer.rho=0.05
# configyaml="config_imagenet_sam.yaml"
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 1289 --use_env main.py --config-dir configs/ --config-name $configyaml optimizer.rho=0.1 optimizer.name=ASAM
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 1289 --use_env main.py --config-dir configs/ --config-name $configyaml optimizer.rho=0.2 optimizer.name=ASAM
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 1289 --use_env main.py --config-dir configs/ --config-name $configyaml optimizer.rho=0.05 optimizer.name=ASAM
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 1289 --use_env main.py --config-dir configs/ --config-name $configyaml optimizer.rho=0.15 optimizer.name=ASAM
# configyaml="config_imagenet_sam.yaml"
# CUDA_VISIBLE_DEVICES=$cuda_use python -m torch.distributed.launch --nproc_per_node 4 --master_port $port_use --use_env main.py --config-dir configs/ --config-name $configyaml optimizer.rho=0.5 optimizer.name=ASAM
# CUDA_VISIBLE_DEVICES=$cuda_use python -m torch.distributed.launch --nproc_per_node 4 --master_port $port_use --use_env main.py --config-dir configs/ --config-name $configyaml optimizer.rho=1.0 optimizer.name=ASAM
# CUDA_VISIBLE_DEVICES=$cuda_use python -m torch.distributed.launch --nproc_per_node 4 --master_port $port_use --use_env main.py --config-dir configs/ --config-name $configyaml optimizer.rho=2.0 optimizer.name=ASAM
# CUDA_VISIBLE_DEVICES=$cuda_use python -m torch.distributed.launch --nproc_per_node 4 --master_port $port_use --use_env main.py --config-dir configs/ --config-name $configyaml optimizer.rho=1.5 optimizer.name=ASAM