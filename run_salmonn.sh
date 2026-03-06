#!/bin/bash -l
#SBATCH --job-name=salmonn-gpu
#SBATCH --account=AIRR-P39-DAWN-GPU
#SBATCH --partition=pvc9
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

module purge
module load rhel8/default-dawn
module load intelpython-conda/2025.0

conda activate /home/dn-rana1/rds/conda_envs/llm_exp

export HF_HOME=/home/dn-rana1/rds/rds-airr-p39-JpwWyPZa2Oc/hf_home/
export HF_TOKEN=""
pip install "transformers>=4.36.0,<4.45.0" "tokenizers>=0.15.0,<0.21.0" --break-system-packages

# ============================================================
# Paths
# ============================================================
RDS_DIR=/home/dn-rana1/rds/rds-airr-p39-JpwWyPZa2Oc
SALMONN_DIR=${RDS_DIR}/models/SALMONN
MODEL_DIR=${RDS_DIR}/models/salmonn_weights

# ============================================================
# Step 1: Clone SALMONN repo (only on first run)
# ============================================================
if [ ! -d "${SALMONN_DIR}" ]; then
    echo "Cloning SALMONN repo..."
    mkdir -p ${RDS_DIR}/models
    git clone -b salmonn https://github.com/bytedance/SALMONN.git ${SALMONN_DIR}
    pip install -r ${SALMONN_DIR}/requirements.txt --break-system-packages
else
    echo "SALMONN repo already exists at ${SALMONN_DIR}"
fi

# ============================================================
# Step 2: Download model weights (only on first run)
# ============================================================
mkdir -p ${MODEL_DIR}

# 2a. Whisper Large V2
if [ ! -d "${MODEL_DIR}/whisper-large-v2" ]; then
    echo "Downloading Whisper Large V2..."
    git lfs install
    git clone https://huggingface.co/openai/whisper-large-v2 ${MODEL_DIR}/whisper-large-v2
else
    echo "Whisper already downloaded"
fi

# 2b. BEATs encoder
if [ ! -f "${MODEL_DIR}/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt" ]; then
    echo "Downloading BEATs..."
    wget -O ${MODEL_DIR}/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
      "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt?sv=2020-08-04&st=2023-03-01T07:51:05Z&se=2033-03-02T07:51:00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D"
else
    echo "BEATs already downloaded"
fi

# 2c. Vicuna 7B v1.5
if [ ! -d "${MODEL_DIR}/vicuna-7b-v1.5" ]; then
    echo "Downloading Vicuna 7B v1.5..."
    git lfs install
    git clone https://huggingface.co/lmsys/vicuna-7b-v1.5 ${MODEL_DIR}/vicuna-7b-v1.5
else
    echo "Vicuna already downloaded"
fi

# 2d. SALMONN 7B checkpoint
if [ ! -f "${MODEL_DIR}/salmonn_7b_v0.pth" ]; then
    echo "Downloading SALMONN 7B checkpoint..."
    wget -O ${MODEL_DIR}/salmonn_7b_v0.pth \
      "https://huggingface.co/tsinghua-ee/SALMONN-7B/resolve/main/salmonn_7b_v0.pth"
else
    echo "SALMONN checkpoint already downloaded"
fi

echo "============================================"
echo "All models ready. Listing weights:"
ls -lh ${MODEL_DIR}/
echo "============================================"

# ============================================================
# Step 3: Run inference
# ============================================================
export SALMONN_REPO=${SALMONN_DIR}
export SALMONN_CKPT=${MODEL_DIR}/salmonn_7b_v0.pth
export WHISPER_PATH=${MODEL_DIR}/whisper-large-v2
export BEATS_PATH=${MODEL_DIR}/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
export VICUNA_PATH=${MODEL_DIR}/vicuna-7b-v1.5

python -m salmonn