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

# Train SegFormer (nvidia/segformer-b1...)
python train_script.py --architecture SegFormer --segformer_model nvidia/segformer-b1-finetuned-ade-512-512 --checkpoint ./checkpoint/best_model_segformer_nvidia_segformer-b1-finetuned-ade-512-512_epoch45_iou0.4737.pth --epochs 50 --lr 6e-5 --batch_size 8 --img_size 512 512
```

# Inference and Evaluation

```sh
# Evaluate a Unet model
python evaluate.py \
    --architecture Unet \
    --unet_encoder mobilenet_v2 \
    --model ./checkpoint/best_model_unet_mobilenet_v2_epoch118_iou0.4578.pth \
    --input_txt ./data/with_mask_test.txt \
    --img_dir ./data/JPEGImages/ \
    --mask_dir ./data/mask/ \
    --output_dir ./results_unet/ \
    --img_size 256 256 \
    --threshold 0.5 \
    --darken 0.3

# Or inference a SegFormer model (no masks given)
python evaluate.py \
    --architecture SegFormer \
    --segformer_model nvidia/segformer-b1-finetuned-ade-512-512 \
    --model ./checkpoint/best_model_segformer_nvidia_segformer-b1-finetuned-ade-512-512_epoch45_iou0.4737.pth \
    --input_txt ./data/with_mask_test.txt \
    --img_dir ./data/JPEGImages/ \
    --output_dir ./results_segformer/ \
    --img_size 512 512 \
    --threshold 0.5 \
    --darken 0.3
```
