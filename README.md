# acne-segmentation

# Install

```sh
conda create -n "AcneSegmentation" python=3.9 -y
conda activate AcneSegmentation

pip install torch torchvision torchaudio
pip install segmentation-models-pytorch opencv-python pandas albumentations tqdm matplotlib
```

# Train

```sh
python train_script.py
```

# Inference

```sh
python evaluate_unet.py \
    --model ./checkpoint/best_model_unet_mobilenet_v2_epoch118_iou0.4578.pth \
    --input_txt ./data/with_mask_test.txt \
    --img_dir ./data/JPEGImages/ \
    --output_dir ./results/ \
    --img_size 256 256 \
    --threshold 0.4 \
    --darken 0.4
```

# Encoders

- Slightly Heavier Mobile-Optimized:
    - 'efficientnet-b0' through 'efficientnet-b4' (Start with b0 or b1 and increase if needed. Generally good balance.)

- Standard ResNets (Proven performers):
    - 'resnet18'
    - 'resnet34' (Commonly used, good baseline)    
    - 'resnet50' (More powerful, significantly more parameters)    

- More Advanced CNNs:
    - 'resnest50d' (ResNeSt variant, often performs well)
    - 'seresnext50_32x4d' (SE-ResNeXt variant)
    - 'convnext_tiny' or 'convnext_small' (More recent CNN architecture)    
