# Dataset directory
DATASET_DIR="/vol/vssp/datasets/audio/dcase2018/task3"

# You need to modify this path
WORKSPACE="/vol/vssp/msos/qk/workspaces/pub_dcase2018_task3"

BACKEND="pytorch"
HOLDOUT_FOLD=1
GPU_ID=1

# Create validation csv
python create_validation.py --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE

# Extract features
python utils/features.py logmel --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type=development
python utils/features.py logmel --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE --data_type=test

############ Development ############
# Train
CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_pytorch.py train --workspace=$WORKSPACE --data_type=development --validate --holdout_fold=$HOLDOUT_FOLD --cuda

# Inference validation
CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_pytorch.py inference_validation --workspace=$WORKSPACE --holdout_fold=$HOLDOUT_FOLD --iteration=5000 --cuda

############ Full train ############
# Train using all development data
CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_pytorch.py train --workspace=$WORKSPACE --data_type=development --cuda

# Predict on test data
CUDA_VISIBLE_DEVICES=$GPU_ID python $BACKEND/main_pytorch.py inference_testing_data --workspace=$WORKSPACE --iteration=3000 --cuda