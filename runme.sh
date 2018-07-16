# Dataset directory
DATASET_DIR="/vol/vssp/datasets/audio/dcase2018/task3"

# You need to modify this path
WORKSPACE="/vol/vssp/msos/qk/workspaces/dcase2018_task3"

# Create validation csv
python create_validation.py --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

# Extract features
python features.py logmel --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

# Train
CUDA_VISIBLE_DEVICES=2 python main_pytorch.py train --workspace=$WORKSPACE

# Inference bottleneck feature
CUDA_VISIBLE_DEVICES=2 python main_pytorch.py inference_bottleneck --workspace=$WORKSPACE --iteration=1000
