# Road-lanes classification 

## Requirements
- Rye

## Installation
```bash
cd hogehoge/lane-case-classification
rye sync
```


## Dataset

## Training 
```bash
. .venv/bin/activate
python src/lane_case_classification/main.py --data_dir data/ --batch_size 8 --epochs 100 --save_dir logs --checkpoint_epoch 2 --img_size 224
```

## Inference
```bash
. .venv/bin/activate
python src/lane_case_classification/inference.py --infer_dir hoge/ --model_path logs/hoge.pth --num_classes 8 --output_file inference.txt
```
