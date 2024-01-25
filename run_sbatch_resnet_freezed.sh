#!/bin/bash
#SBATCH --gres=gpu:1 --cpus-per-task=4 -t 06:00:00 --mail-type=ALL,TIME_LIMIT_80

eval "$(conda shell.bash hook)"
conda activate comp3340gp
cd ~/COMP3340-gp

python main.py device=cuda model.backbone.target=model.backbone.ResNet18 model.backbone.params.frozen=True model.classifier.params.in_features=512
