# COMP3340 Group Project - Image Classifier on Flowers17
This is the repository for the group project of COMP3340 Applied Machine Learning @ HKU.
## Environment Setup
```
conda create -n comp3340gp python=3.11
conda activate comp3340gp
pip install -r requirement.txt
```

## Configuration
Please kindly check the configuration files in `./configs`. For attention visualization, it's only available for VisionTransformer models.

If you want to train the model (except GroupMixFormer, which you need to download the pre-trained checkpoint manually), directly run the following script, pre-trained models and data for training will be downloaded automatically.
```
python main.py -config resnet18.yaml
```

## Web UI
We provide a Web UI for inference and some analysis. You could directly run
```
python webui.py
```
to start the Web UI. It will check the checkpoints saved in `./lightning_logs`. You can download our trained checkpoints from [OneDrive](https://cloudstorage-intl.jdscript.app/Models/COMP3340). Please keep the folder structure like `ViT-Base-16/checkpoints/*.ckpt`.