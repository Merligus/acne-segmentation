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