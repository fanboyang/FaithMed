import ast
import base64
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]

class RetrievalProtocolTests(unittest.TestCase):
    def load(self):
        # Execute only the real query_medcorp function; unrelated optional web search
        # providers are deliberately outside this data-free protocol test.
        source=ROOT/'search_agent/retrieval.py'
        tree=ast.parse(source.read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='query_medcorp')
        import requests,time,random
        scope={'requests':requests,'time':time,'random':random,'os':os,'json':json,'base64':base64,
               'MAX_RETRIES':1,'medcorp_time_log':str(Path(self.tmp.name)/'time.log'),
               'medcorp_error_log':str(Path(self.tmp.name)/'error.log')}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),scope)
        return scope['query_medcorp'],requests

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)

    def test_url_is_required_not_placeholder_and_query_is_encoded(self):
        query,requests=self.load()
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(ValueError):query('x')
        with patch.dict(os.environ,{'FAITHMED_MEDCORP_URL':'https://medrag.your-host.com'},clear=True):
            with self.assertRaises(ValueError):query('x')
        response=unittest.mock.Mock()
        response.json.return_value={'results':[]}
        with patch.dict(os.environ,{'FAITHMED_MEDCORP_URL':'http://127.0.0.1:18700'},clear=True),patch.object(requests,'get',return_value=response) as get:
            self.assertEqual(query('a & b?',6),[])
            self.assertEqual(get.call_args.args[0],'http://127.0.0.1:18700/search')
            self.assertEqual(get.call_args.kwargs['params'],{'query':'a & b?','k':6})

    def test_valid_document_and_missing_results_failure(self):
        query,requests=self.load()
        doc=base64.b64encode(json.dumps({'Clean-Text':'Book. evidence','title':'Book','data_source':'textbooks'}).encode()).decode()
        response=unittest.mock.Mock()
        with patch.dict(os.environ,{'FAITHMED_MEDCORP_URL':'http://127.0.0.1:18700'}),patch.object(requests,'get',return_value=response):
            response.json.return_value={'results':[doc]}
            self.assertIn('evidence',query('x')[0])
            for raw in [{},{'results':None},{'results':['not base64']}]:
                response.json.return_value=raw
                with self.subTest(raw=raw),self.assertRaises((ValueError,KeyError,TypeError)):query('x')

class LauncherTests(unittest.TestCase):
    def fake_root(self,directory):
        root=Path(directory)
        for folder in ['private','env/bin','models/Qwen3-1.7B','data/medmix_rl','stubs']:
            (root/folder).mkdir(parents=True,exist_ok=True)
        (root/'code').symlink_to(ROOT,target_is_directory=True)
        (root/'models/Qwen3-1.7B/config.json').write_text('{"model_type":"qwen3"}')
        for name in ['train','test']:(root/f'data/medmix_rl/{name}.parquet').write_text('fixture')
        (root/'private/judge.env').write_text("ZC_BASE_URL=https://api.zcloudapi.com/v1\nZC_GPT_API_KEY=test-private-credential\n")
        (root/'env/bin/activate').write_text('export PATH='+shlex.quote(str(root/'stubs'))+':"$PATH"\n')
        python=root/'stubs/python'
        python.write_text('#!/bin/bash\nif [[ "$1" == - ]]; then exec '+shlex.quote(sys.executable)+' "$@"; fi\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n')
        python.chmod(0o755)
        gpu=root/'stubs/nvidia-smi'
        gpu.write_text('#!/bin/bash\nif [[ "$1" == --query-gpu=* ]]; then printf "0, GPU-0, 0\\n1, GPU-1, 0\\n2, GPU-2, 0\\n3, GPU-3, 0\\n"; elif [[ "${SIMULATE_BUSY:-0}" == 1 ]]; then echo "GPU-0, 99"; fi\n')
        gpu.chmod(0o755)
        return {**os.environ,'FAITHMED_ROOT':str(root),'FAITHMED_MEDCORP_URL':'http://127.0.0.1:18700',
                'CAPTURE_ARGS':str(root/'captured.txt')}

    def test_cfg_job_and_busy_gpu_guard_without_running_training(self):
        with tempfile.TemporaryDirectory() as directory:
            env=self.fake_root(directory)
            completed=subprocess.run(['bash','-x',str(ROOT/'reproduction/train.sh'),'--cfg','job'],
                                     env=env,capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stderr)
            self.assertNotIn('test-private-credential',completed.stdout+completed.stderr)
            args=Path(env['CAPTURE_ARGS']).read_text()
            self.assertIn('trainer.total_training_steps=300',args)
            self.assertIn('--cfg\njob',args)
            Path(env['CAPTURE_ARGS']).unlink()
            env['SIMULATE_BUSY']='1'
            failed=subprocess.run(['bash',str(ROOT/'reproduction/train.sh')],env=env,capture_output=True,text=True)
            self.assertNotEqual(failed.returncode,0)
            self.assertIn('occupied',failed.stderr)
            self.assertFalse(Path(env['CAPTURE_ARGS']).exists())

    def test_explicit_single_gpu_smoke_keeps_core_default(self):
        with tempfile.TemporaryDirectory() as directory:
            env=self.fake_root(directory)
            gpu=Path(directory)/'stubs/nvidia-smi'
            gpu.write_text('#!/bin/bash\nif [[ "$1" == --query-gpu=* ]]; then echo "4, GPU-4, 0"; fi\n')
            env['FAITHMED_SMOKE_GPU']='4'
            completed=subprocess.run(['bash',str(ROOT/'reproduction/train.sh')],env=env,capture_output=True,text=True)
            self.assertEqual(completed.returncode,0,completed.stderr)
            args=Path(env['CAPTURE_ARGS']).read_text()
            self.assertIn('trainer.n_gpus_per_node=1',args)
            self.assertIn('actor_rollout_ref.rollout.tensor_model_parallel_size=1',args)
            self.assertIn('actor_rollout_ref.actor.ulysses_sequence_parallel_size=1',args)

    def test_core_config_and_safe_launch(self):
        script=(ROOT/'reproduction/train.sh').read_text()
        for expected in ['CUDA_VISIBLE_DEVICES=0,1,2,3','trainer.total_training_steps=300',
                         'trainer.val_before_train=False','trainer.test_freq=300',
                         'env.gemini_model=gpt-6-luna','env.process_reward_enable=1',
                         'env.step_scoring=True','actor_rollout_ref.rollout.max_num_batched_tokens=16384','FAITHMED_MEDCORP_URL','ZC_GPT_API_KEY']:
            self.assertIn(expected,script)
        self.assertNotIn('set -x',script)
        self.assertNotIn("'console','wandb'",script)
        self.assertIn('"$@"',script)

if __name__=='__main__':unittest.main()
