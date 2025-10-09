#!/bin/bash
# The config is optimized for 8xA100
set -x
cd ./verl

export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=xxx
export WANDB_MODE=offline

# Color definitions
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ------------------------------------------------------------------------------------------------
# Cleanup function to kill the vllm server process
cleanup() {
    echo "Cleaning up background processes..."
    # Kill the vllm server process
    jobs -p | xargs -r kill
    exit 0
}

# Set trap to call cleanup function on script exit, interrupt, or termination
trap cleanup EXIT INT TERM

# ------------------------------------------------------------------------------------------------
# Get the parameters from the command line
BASE_MODEL=${1:-"your model path"}
MAX_EPOCHS=${2:-"12"}
DATASET=${3:-"polaris"}
NO_ENTROPY_LOSS_AND_KL=${4:-"True"}
VERIFIER_MODEL=${5:-"xxx"}
VERIFIER_SETUP=${6:-"vanilla"} # addon
VERIFIER_WEIGHT=${7:-"1.0"}

# MAIN CONFIG
MODEL_PATH=$BASE_MODEL
ROLLOUT_BATCH_SIZE=512
ROLLOUT_N_SAMPLE=8
PPO_MINI_BATCH_SIZE=64

MODEL_NICKNAME=$(basename "$MODEL_PATH")
VERIFIER_MODEL_NICKNAME=$(basename "$VERIFIER_MODEL")
# MODEL_NICKNAME=$(echo $MODEL_PATH | cut -d'/' -f2)
# VERIFIER_MODEL_NICKNAME=$(echo $VERIFIER_MODEL | cut -d'/' -f2)
TIMESTAMP=$(date +%Y%m%d%H%M%S)
if [[ "$DATASET" == *_tinyv* ]]; then
    RUN_NAME=${MODEL_NICKNAME}-${DATASET}-${VERIFIER_SETUP}-ppo
else
    RUN_NAME=${MODEL_NICKNAME}-${DATASET}-${VERIFIER_SETUP}-ppo
    # RUN_NAME=${MODEL_NICKNAME}-${DATASET}-3
fi

if [ "$NO_ENTROPY_LOSS_AND_KL" == "True" ]; then
    RUN_NAME=${RUN_NAME}-NEKL
    USE_KL_LOSS=FALSE
    USE_KL_IN_REWARD=FALSE
    KL_COEF=0.000
else
    USE_KL_LOSS=True
    USE_KL_IN_REWARD=True
    KL_COEF=0.001
fi

export RUN_NAME=${RUN_NAME}
export GLOBAL_RUN_NAME=${RUN_NAME}

TOTAL_SAMPLES=$(( PPO_MINI_BATCH_SIZE * ROLLOUT_N_SAMPLE )) # Number of experiences
echo "Number of experiences: $TOTAL_SAMPLES"

SAVED_DIR="./models_rl/${RUN_NAME}"

mkdir -p .checkpoints/$RUN_NAME
mkdir -p $SAVED_DIR

MAX_PROMPT_LEN=1024
MAX_RESPONSE_LEN=4096
PPO_MAX_TOKEN_LEN_PER_GPU=$(( 3 * $(( $MAX_PROMPT_LEN + $MAX_RESPONSE_LEN )) ))
MAX_NUM_BATCHED_TOKENS=$(( 1 * $(( $MAX_PROMPT_LEN + $MAX_RESPONSE_LEN )) ))

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    GPUS_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
fi

echo -e "${BLUE}[PPO Trainer] Running PPO training...${NC}"

python3 -m verl.trainer.main_ppo \
    algorithm.use_kl_in_reward=$USE_KL_IN_REWARD \
    data.train_files=/data/train_polaris.parquet \
    data.val_files=data/test_math.parquet \
    data.train_batch_size=$ROLLOUT_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_RESPONSE_LEN \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=$USE_KL_LOSS \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.n=$ROLLOUT_N_SAMPLE \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    +trainer.rollout_data_dir=$SAVED_DIR \
    algorithm.kl_ctrl.kl_coef=0.001 \
    critic.optim.lr=9e-6 \
    critic.model.path=$MODEL_PATH \
    critic.model.use_remove_padding=True \
    critic.ppo_max_token_len_per_gpu=24000 \
    critic.forward_max_token_len_per_gpu=36000 \
    reward_model.enable=True \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.micro_batch_size_per_gpu=32 \
    reward_model.reward_manager=naive \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='TinyV-0429' \
    trainer.experiment_name=${RUN_NAME} \
    trainer.nnodes=1 \
    trainer.default_local_dir=$SAVED_DIR \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.save_freq=10 \
    trainer.test_freq=2 \
    trainer.validation_data_dir=$SAVED_DIR \
    trainer.resume_mode=auto \
    trainer.total_epochs=$MAX_EPOCHS 12>&1 | tee .checkpoints/$RUN_NAME/train.log
