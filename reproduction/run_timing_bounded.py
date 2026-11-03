"""GPU0-only, five-update timing pilot; isolated owner cleanup and a two-hour budget."""
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import uuid

# Reuse the tested exact-owner cleanup. The historical environment-key name is
# shared, but this task's unique UUID owner never matches an integration smoke.
from reproduction.run_smoke_bounded import cleanup_run

RUN_NAME='faithmed-timing-v1'
BUDGET_SECONDS=7200


def collect_gpu0(path,env,stop):
    with path.open('x') as handle:
        handle.write('timestamp, utilization.gpu, memory.used, memory.total, power.draw\n')
        handle.flush()
        while not stop.is_set():
            try:
                result=subprocess.run(['nvidia-smi','--id=0',
                    '--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,power.draw',
                    '--format=csv,noheader,nounits'],env=env,capture_output=True,text=True,timeout=8)
                if result.returncode:
                    csv.writer(handle).writerow([time.time(),'sampling_error',result.returncode])
                else:
                    handle.write(result.stdout if result.stdout.endswith('\n') else result.stdout+'\n')
                handle.flush()
            except Exception as exc:
                csv.writer(handle).writerow([time.time(),'sampling_error',type(exc).__name__]);handle.flush()
            stop.wait(10)


def main():
    root=Path(os.environ.get('FAITHMED_ROOT','/data/fby/faithmed'))
    logs=root/'logs';logs.mkdir(exist_ok=True)
    output=root/'outputs'/RUN_NAME
    if output.exists():raise FileExistsError('Timing output already exists; never resume this pilot')
    with (logs/'timing.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.monotonic()
        owner=RUN_NAME+':'+str(uuid.uuid4())
        env={**os.environ,'FAITHMED_RUN_NAME':RUN_NAME,'FAITHMED_SMOKE_OWNER':owner,
             'FAITHMED_TRAIN_DEVICES':'0'}
        gpu_log=logs/('gpu0-'+RUN_NAME+'-'+owner.split(':')[1]+'.csv')
        stop=threading.Event()
        sampler=threading.Thread(target=collect_gpu0,args=(gpu_log,env,stop),daemon=True)
        code=None;cleaned=[];cleanup_error=None
        def interrupted(signum,frame):
            nonlocal code
            code=128+signum
            raise SystemExit(code)
        signal.signal(signal.SIGTERM,interrupted)
        signal.signal(signal.SIGINT,interrupted)
        try:
            sampler.start()
            process=subprocess.Popen(['bash',str(root/'code/reproduction/timing.sh')],env=env,start_new_session=True)
            try:code=process.wait(timeout=BUDGET_SECONDS)
            except subprocess.TimeoutExpired:code=124
        finally:
            try:
                stop.set();sampler.join(timeout=12)
                if sampler.is_alive():raise RuntimeError('GPU sampler did not stop')
                for _ in range(3):
                    found=cleanup_run(owner);cleaned.extend(found)
                    if not found:break
                else:raise RuntimeError('timing owner processes remain after cleanup')
                if 'process' in locals():process.wait(timeout=30)
            except BaseException as exc:
                cleanup_error=type(exc).__name__
                raise
            finally:
                (logs/(RUN_NAME+'-bounded-status.json')).write_text(json.dumps({
                    'scope':'timing_pilot_40_docs_not_paper_result','run_name':RUN_NAME,'owner':owner,
                    'returncode':code,'elapsed_seconds':time.monotonic()-started,
                    'budget_seconds':BUDGET_SECONDS,'cleaned_tagged_pids':cleaned,
                    'cleanup_error':cleanup_error,'gpu0_csv':str(gpu_log)},indent=2))
        raise SystemExit(code)

if __name__=='__main__':main()
