#!/usr/bin/env bash
# FaithMed no-SFT core recipe. Stop at 300 updates; evaluate the final checkpoint.
set +x
set -euo pipefail
root=${FAITHMED_ROOT:-/data/fby/faithmed}
: "${FAITHMED_MEDCORP_URL:?set a real MedCorp retrieval base URL}"
[[ -f "$root/private/judge.env" ]] || { echo 'Missing private/judge.env' >&2; exit 2; }
source "$root/private/judge.env"
: "${ZC_GPT_API_KEY:?judge API key is missing}"
: "${ZC_BASE_URL:?judge base URL is missing}"
export OPENAI_API_KEY="$ZC_GPT_API_KEY"
case "${ZC_BASE_URL%/}" in
  */chat/completions) export OPENAI_API_URL="${ZC_BASE_URL%/}" ;;
  *) export OPENAI_API_URL="${ZC_BASE_URL%/}/chat/completions" ;;
esac
source "$root/env/bin/activate"
export CUDA_VISIBLE_DEVICES=0,1,2,3
if [[ -n "${FAITHMED_SMOKE_GPU:-}" ]]; then
  [[ "$FAITHMED_SMOKE_GPU" == 4 ]] || { echo 'Only GPU4 is allocated to the bounded single-GPU smoke' >&2; exit 2; }
  export CUDA_VISIBLE_DEVICES=4
fi
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export FAITHMED_MEDCORP_URL
run_name=${FAITHMED_RUN_NAME:-faithmed-qwen3-1.7b-gpt6luna}
output="$root/outputs/$run_name"
export FAITHMED_JUDGE_AUDIT_DIR="$output/judge"
export TMPDIR="$root/tmp/$run_name"
export RAY_TMPDIR="$TMPDIR/ray"
mkdir -p "$output" "$RAY_TMPDIR" "$FAITHMED_JUDGE_AUDIT_DIR"
export PYTHONPATH="$root/code:$root/code/verl-agent${PYTHONPATH:+:$PYTHONPATH}"
export NCCL_DEBUG=WARN RAY_DEDUP_LOGS=1
export no_proxy="localhost,127.0.0.1,0.0.0.0${no_proxy:+,$no_proxy}"
unset ROCR_VISIBLE_DEVICES

# Configuration-only Hydra inspection does not allocate a GPU or call a judge.
config_only=false
for argument in "$@"; do [[ "$argument" == --cfg ]] && config_only=true; done
python - "$root" "$config_only" <<'PREFLIGHT'
import json,os,subprocess,sys
from pathlib import Path
from urllib.parse import urlsplit
root=Path(sys.argv[1])
for name in ['FAITHMED_MEDCORP_URL','OPENAI_API_URL']:
    url=os.environ[name]
    parsed=urlsplit(url)
    if parsed.scheme not in ('http','https') or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment or any(x in url.lower() for x in ('your-host','placeholder','.example')):
        raise ValueError(name+' is not a real credential-free endpoint')
model=root/'models/Qwen3-1.7B'
with (model/'config.json').open() as handle: config=json.load(handle)
if config.get('model_type')!='qwen3':raise ValueError('student must be Qwen3-1.7B')
for split in ('train','test'):
    if not (root/'data/medmix'/f'{split}.parquet').is_file():raise FileNotFoundError(split+' parquet missing')
if sys.argv[2]!='true':
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    gpu={int(fields[0]):(fields[1].strip(),int(fields[2])) for line in rows.splitlines() if (fields:=line.split(','))}
    requested=[int(x) for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    if not all(index in gpu for index in requested):raise RuntimeError('Requested GPUs are missing')
    busy=[index for index in requested if gpu[index][1]>1024]
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    active={line.split(',')[0].strip() for line in apps.splitlines() if line.strip()}
    busy+= [index for index in requested if gpu[index][0] in active]
    if busy:raise RuntimeError('Requested GPUs are occupied: '+str(sorted(set(busy))))
PREFLIGHT
cd "$root/code/verl-agent"
RUN_NAME="$run_name"
PROJECT_NAME=faithmed
TRAIN_BATCH_SIZE=32
VAL_BATCH_SIZE=512
MAX_PROMPT_LENGTH=10000
MAX_RESPONSE_LENGTH=3072
LEARNING_RATE=1e-6
PPO_MINI_BATCH_SIZE=128
N_GPUS_PER_NODE=4
SEQUENCE_PARALLEL_SIZE=2
[[ -z "${FAITHMED_SMOKE_GPU:-}" ]] || { N_GPUS_PER_NODE=1; SEQUENCE_PARALLEL_SIZE=1; }
CLIP_RATIO=0.2
KL_LOSS_COEF=0.001
ENTROPY_COEFFIENT=0.001
KL_LOSS_TYPE=low_var_kl
TEMPERATURE=1.0
LOG_PROB_MICRO_BATCH_SIZE=128
ROLLOUT_N=8
KL_COEF=0.001
DATASET_NAME=medmix
ROLLOUT_GPU_MEMORY_UTIL=0.8
LOG_VAL_GENERATIONS=3
SAVE_FREQ=300
REMOVE_CLIP=False
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=2
[[ -z "${FAITHMED_SMOKE_GPU:-}" ]] || ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1
ENV_MAX_STEPS=4
NUM_CPUS_PER_ENV_WORKER=0.125
SEARCH_ENGINE=medcorp
NUM_DOCS=6
PROCESS_REWARD_LAMBDA=0.05
PROCESS_REWARD_DEBUG=0
GIGPO_ANCHOR_SOURCE=query
USE_INVALID_ACTION_PENALTY=True
INVALID_ACTION_PENALTY_COEF=0.05
VERBOSE_FREQ=-1
VALIDATION_PROCESS_REWARD_FREQ=300
RUBRIC_DISABLED_DIMS=""
GAMMA=1.0
PPO_MAX_TOKEN_LEN_PER_GPU=32000
HDFS_DATA_PATH="$root/data"
HDFS_CHECKPOINT_PATH="$output/checkpoints"
LOG_FILE_PATH="$output/train.log"
MODEL_PATH="$root/models/Qwen3-1.7B"
mode=mean_std_norm
enable_similarity=True
SIMILARITY_THRESH=0.8
ARNOLD_WORKER_NUM=1
ENV_LOG_DIR="$output/trajectories"
ENV_ANSWER_DIR="$output/answers"
ENV_GROUPS_DIR="$output/groups"
ENV_RUBRICS_DIR="$output/rubrics"
ENV_VAL_LOG_DIR="$output/final_eval/trajectories"
ENV_VAL_ANSWER_DIR="$output/final_eval/answers"
ENV_VAL_RUBRICS_DIR="$output/final_eval/rubrics"
mkdir -p "$HDFS_CHECKPOINT_PATH" "$ENV_LOG_DIR" "$ENV_ANSWER_DIR" "$ENV_GROUPS_DIR" "$ENV_RUBRICS_DIR" "$ENV_VAL_LOG_DIR" "$ENV_VAL_ANSWER_DIR" "$ENV_VAL_RUBRICS_DIR"

python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=gigpo \
  algorithm.use_kl_in_reward=False \
  algorithm.norm_adv_by_std_in_grpo=False \
  algorithm.compute_mean_std_cross_all_data=True \
  data.train_files=$HDFS_DATA_PATH/$DATASET_NAME/train.parquet \
  data.val_files=$HDFS_DATA_PATH/$DATASET_NAME/test.parquet \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  data.val_batch_size=$VAL_BATCH_SIZE \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.return_raw_chat=True \
  data.filter_overlong_prompts=True \
  data.truncation='left' \
  actor_rollout_ref.model.path=$MODEL_PATH \
  actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
  actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFFIENT \
  actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
  actor_rollout_ref.actor.use_invalid_action_penalty=$USE_INVALID_ACTION_PENALTY \
  actor_rollout_ref.actor.invalid_action_penalty_coef=$INVALID_ACTION_PENALTY_COEF \
  actor_rollout_ref.model.enable_activation_offload=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=$SEQUENCE_PARALLEL_SIZE \
  actor_rollout_ref.rollout.temperature=$TEMPERATURE \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTIL \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=True  \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  algorithm.gamma=$GAMMA \
  algorithm.gigpo.step_advantage_w=1.0 \
  algorithm.gigpo.mode=$mode \
  algorithm.gigpo.enable_similarity=$enable_similarity \
  algorithm.gigpo.similarity_thresh=$SIMILARITY_THRESH \
  env.env_name=medrm \
  env.seed=0 \
  env.rollout.n=$ROLLOUT_N \
  env.search_engine=$SEARCH_ENGINE \
  env.num_docs=$NUM_DOCS \
  env.max_steps=$ENV_MAX_STEPS \
  env.process_reward_enable=1 \
  env.process_reward_lambda=$PROCESS_REWARD_LAMBDA \
  env.process_reward_debug=$PROCESS_REWARD_DEBUG \
  env.gigpo_anchor_source=$GIGPO_ANCHOR_SOURCE \
  env.gemini_model=gpt-6-luna \
  env.verbose_freq=$VERBOSE_FREQ \
  env.step_scoring=True \
  env.rubric_disabled_dims=$RUBRIC_DISABLED_DIMS \
  env.log_dir=$ENV_LOG_DIR \
  env.answer_dir=$ENV_ANSWER_DIR \
  env.groups_dir=$ENV_GROUPS_DIR \
  env.rubrics_dir=$ENV_RUBRICS_DIR \
  env.val_log_dir=$ENV_VAL_LOG_DIR \
  env.val_answer_dir=$ENV_VAL_ANSWER_DIR \
  env.val_rubrics_dir=$ENV_VAL_RUBRICS_DIR \
  env.resources_per_worker.num_cpus=$NUM_CPUS_PER_ENV_WORKER \
  trainer.critic_warmup=0 \
  trainer.logger=['console'] \
  trainer.project_name=$PROJECT_NAME \
  trainer.experiment_name=$RUN_NAME \
  trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
  trainer.nnodes=$ARNOLD_WORKER_NUM \
  trainer.log_val_generations=$LOG_VAL_GENERATIONS \
  trainer.validation_process_reward_freq=$VALIDATION_PROCESS_REWARD_FREQ \
  trainer.save_freq=$SAVE_FREQ \
  trainer.test_freq=300 \
  trainer.default_local_dir=$HDFS_CHECKPOINT_PATH/$RUN_NAME \
  trainer.max_actor_ckpt_to_keep=5 \
  trainer.max_critic_ckpt_to_keep=5 \
  trainer.total_epochs=300 \
  trainer.total_training_steps=300 \
  trainer.val_before_train=False "$@" 2>&1 | tee -a $LOG_FILE_PATH
