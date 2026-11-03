"""Run the one-update integration test with a 30-minute resource budget.

Cleanup selects only processes carrying this exact run's environment tag;
unrelated projects and the CPU retrieval service are never selected.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import signal
import uuid
import time
import psutil

RUN_NAME='faithmed-integration-smoke-v1'


def cleanup_run(run_name):
    targets=[]
    for process in psutil.process_iter(['pid']):
        if process.pid==os.getpid():continue
        try:
            if process.environ().get('FAITHMED_SMOKE_OWNER')==run_name:
                targets.append(process)
        except psutil.NoSuchProcess:
            continue
    for process in targets:
        try:process.terminate()
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs(targets,timeout=15)
    for process in alive:
        try:process.kill()
        except psutil.NoSuchProcess:pass
    _,remaining=psutil.wait_procs(alive,timeout=15)
    if remaining:raise RuntimeError('tagged smoke processes did not stop: '+str([p.pid for p in remaining]))
    return [p.pid for p in targets]


def main():
    root=Path(os.environ.get('FAITHMED_ROOT','/data/fby/faithmed'))
    logdir=root/'logs';logdir.mkdir(exist_ok=True)
    with (logdir/'smoke.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        started=time.time()
        owner=RUN_NAME+':'+str(uuid.uuid4())
        env={**os.environ,'FAITHMED_RUN_NAME':RUN_NAME,'FAITHMED_SMOKE_OWNER':owner}
        def interrupted(signum,frame):raise SystemExit(128+signum)
        signal.signal(signal.SIGTERM,interrupted)
        signal.signal(signal.SIGINT,interrupted)
        code=None;cleaned=[];cleanup_error=None
        try:
            process=subprocess.Popen(['bash',str(root/'code/reproduction/smoke.sh')],env=env,start_new_session=True)
            try:code=process.wait(timeout=1800)
            except subprocess.TimeoutExpired:code=124
        finally:
            try:
                for _ in range(3):
                    found=cleanup_run(owner)
                    cleaned.extend(found)
                    if not found:break
                else:raise RuntimeError('tagged processes persisted after three cleanup sweeps')
                if 'process' in locals():process.wait(timeout=30)
            except BaseException as exc:
                cleanup_error=type(exc).__name__
                raise
            finally:
                (logdir/'smoke-bounded-status.json').write_text(json.dumps({'scope':'integration_smoke_not_benchmark','run_name':RUN_NAME,'owner':owner,'returncode':code,'elapsed_seconds':time.time()-started,'budget_seconds':1800,'cleaned_tagged_pids':cleaned,'cleanup_error':cleanup_error},indent=2))
        raise SystemExit(code)

if __name__=='__main__':main()
