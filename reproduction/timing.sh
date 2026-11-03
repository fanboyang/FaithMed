#!/usr/bin/env bash
# Five real updates on the full pool; 40-document retrieval timing, not a benchmark.
set +x
set -euo pipefail
root=${FAITHMED_ROOT:-/data/fby/faithmed}
[[ -z "${FAITHMED_SMOKE_GPU:-}" ]] || { echo 'Timing and integration smoke GPU settings conflict' >&2; exit 2; }
export FAITHMED_TRAIN_DEVICES=0
export FAITHMED_RUN_NAME=faithmed-timing-v1
export FAITHMED_MEDCORP_URL=http://127.0.0.1:18601
output="$root/outputs/$FAITHMED_RUN_NAME"
[[ ! -e "$output" ]] || { echo 'Timing output already exists; inspect prior run' >&2; exit 2; }
source "$root/env/bin/activate"
export CUDA_VISIBLE_DEVICES=''
python - "$root" "$output" <<'PY'
import json,sys,urllib.request
from pathlib import Path
import pyarrow.parquet as pq
root=Path(sys.argv[1]);output=Path(sys.argv[2])
with urllib.request.urlopen('http://127.0.0.1:18601/health',timeout=15) as response:health=json.load(response)
if health.get('scope')!='integration_smoke_subset_exact_search' or health.get('documents')!=40 or len(health.get('sources',[]))!=5:
    raise ValueError('Timing requires the explicit existing 40-document five-source retrieval service')
train=root/'data/medmix_rl/train.parquet'
count=pq.ParquetFile(train).metadata.num_rows
if count!=33906:raise ValueError('Timing requires the entire frozen 33906-row training pool')
output.mkdir(parents=True,exist_ok=False)
(output/'timing_manifest.json').write_text(json.dumps({'scope':'timing_pilot_40_docs_not_paper_result',
 'train_rows':count,'train_path':str(train),'steps':5,'batch_size':32,'rollouts_per_prompt':8,
 'ppo_mini_batch_size':128,'student':'original Qwen3-1.7B (no resume)',
 'retrieval':health,'retrieval_url':'http://127.0.0.1:18601','validation_enabled':False,
 'tensor_parallel_size':1,'sequence_parallel_size':1,'logprob_micro_batch_size':1,
 'max_prompt_length':10000,'max_response_length':3072,'ppo_max_token_len_per_gpu':32000,
 'rollout_max_num_batched_tokens':16384,'max_turns':4,'budget_seconds':7200,'seed':0,
 'learning_rate':1e-6,'lr_warmup_steps':0,'resume_mode':'disable',
 'checkpoint_save_freq':5,'gpu_devices':[0],
 'limitation':'Existing 40-document five-source retrieval subset; timing pilot only, not full MedCorp or paper benchmark'},indent=2))
PY
bash "$root/code/reproduction/train.sh" \
  data.train_batch_size=32 actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  env.rollout.n=8 env.max_steps=4 actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
  trainer.total_training_steps=5 trainer.total_epochs=5 \
  trainer.test_freq=-1 trainer.save_freq=5 trainer.val_before_train=False \
  trainer.validation_process_reward_freq=-1 trainer.resume_mode=disable
