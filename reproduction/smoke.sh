#!/usr/bin/env bash
# One training iteration on eight fixed examples; no benchmark claims.
set +x
set -euo pipefail
root=${FAITHMED_ROOT:-/data/fby/faithmed}
export FAITHMED_SMOKE_GPU=4
smoke_id=${FAITHMED_SMOKE_ID:-v1}
[[ "$smoke_id" =~ ^v[1-9][0-9]*$ ]] || { echo 'Invalid smoke version' >&2; exit 2; }
export FAITHMED_SMOKE_ID="$smoke_id"
export FAITHMED_RUN_NAME="faithmed-integration-smoke-$smoke_id"
export FAITHMED_MEDCORP_URL=http://127.0.0.1:18601
source "$root/env/bin/activate"
export CUDA_VISIBLE_DEVICES=''
python - "$root" "$smoke_id" <<'PY'
import json,sys,urllib.request
from pathlib import Path
import pyarrow.parquet as pq
root=Path(sys.argv[1]); out=root/('data/integration-smoke-'+sys.argv[2])
if out.exists():raise FileExistsError('smoke data already exists; inspect prior run before continuing')
with urllib.request.urlopen('http://127.0.0.1:18601/health',timeout=15) as response: health=json.load(response)
assert health['scope']=='integration_smoke_subset_exact_search' and len(health['sources'])==5
rows=pq.read_table(root/'data/medmix_rl/train.parquet')
sources=rows.column('data_source').to_pylist(); counts={}; selected=[]
for i,source in enumerate(sources):
    if counts.get(source,0)<2:selected.append(i);counts[source]=counts.get(source,0)+1
assert len(selected)==8 and len(counts)==4
out.mkdir()
pq.write_table(rows.take(selected),out/'train.parquet')
pq.write_table(rows.take(selected[:2]),out/'validation.parquet')
(out/'manifest.json').write_text(json.dumps({'scope':'integration_smoke_not_benchmark','train_indices':selected,'sources':counts,'validation':'first two selected training rows; no held-out accuracy claim','retrieval':health,'training_iterations':1,'rollouts_per_prompt':8,'ppo_minibatch_rows':64},ensure_ascii=False,indent=2))
PY
bash "$root/code/reproduction/train.sh" \
  data.train_files="$root/data/integration-smoke-$smoke_id/train.parquet" \
  data.val_files="$root/data/integration-smoke-$smoke_id/validation.parquet" \
  data.train_batch_size=8 data.val_batch_size=2 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
  trainer.total_training_steps=1 trainer.total_epochs=1 \
  trainer.test_freq=-1 trainer.save_freq=1 trainer.val_before_train=False
