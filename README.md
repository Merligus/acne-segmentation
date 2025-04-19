# acne-segmentation

# Install

```sh
conda create -n "AcneSegmentation" python=3.9 -y
conda activate AcneSegmentation

pip install torch torchvision torchaudio
pip install segmentation-models-pytorch opencv-python pandas albumentations tqdm matplotlib transformers
```

# Train

```sh
# Train Unet (efficientnet-b4)
python train_script.py --architecture Unet --unet_encoder efficientnet-b4 --checkpoint ./checkpoint/best_model_unet_efficientnet-b4_epoch78_iou0.4648.pth --epochs 50 --lr 0.00001 --batch_size 8 --img_size 256 256

# Train SegFormer (nvidia/segformer-b0...)
python train_script.py --architecture SegFormer --segformer_model nvidia/segformer-b0-finetuned-ade-512-512 --epochs 50 --lr 6e-5 --batch_size 8 --img_size 256 256
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
