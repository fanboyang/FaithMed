import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from reproduction import test_launch_retrieval as fixtures

ROOT=Path(__file__).resolve().parents[1]

class TimingPilotTests(unittest.TestCase):
    def test_gpu_zero_only_and_conflicting_smoke_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            env=fixtures.LauncherTests.fake_root(self,directory)
            (Path(directory)/'stubs/nvidia-smi').write_text('#!/bin/bash\nif [[ "$1" == --query-gpu=* ]]; then printf "0, GPU-0, 0\\n1, GPU-1, 5000\\n5, GPU-5, 5000\\n"; else echo "GPU-5, 99"; fi\n')
            env['FAITHMED_TRAIN_DEVICES']='0'
            result=subprocess.run(['bash',str(ROOT/'reproduction/train.sh')],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            args=Path(env['CAPTURE_ARGS']).read_text()
            for flag in ['trainer.n_gpus_per_node=1','actor_rollout_ref.rollout.tensor_model_parallel_size=1',
                         'actor_rollout_ref.actor.ulysses_sequence_parallel_size=1',
                         'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1',
                         'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1']:
                self.assertIn(flag,args)
            env['FAITHMED_SMOKE_GPU']='4'
            result=subprocess.run(['bash',str(ROOT/'reproduction/train.sh'),'--cfg','job'],env=env,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('conflict',result.stderr.lower())

    def test_timing_scope_and_existing_output_fail(self):
        script=ROOT/'reproduction/timing.sh'
        self.assertTrue(script.exists(),'timing entry is missing')
        content=script.read_text()
        for value in ['FAITHMED_TRAIN_DEVICES=0','33906','documents','40','trainer.total_training_steps=5',
                      'trainer.test_freq=-1','trainer.save_freq=5','data.train_batch_size=32',
                      'actor_rollout_ref.actor.ppo_mini_batch_size=128','env.rollout.n=8', 'actor_rollout_ref.actor.optim.lr_warmup_steps=0']:
            self.assertIn(value,content)
        for key in ['tensor_parallel_size', 'sequence_parallel_size', 'logprob_micro_batch_size',
                    'max_prompt_length', 'max_response_length', 'budget_seconds', 'seed',
                    'learning_rate', 'lr_warmup_steps', 'resume_mode']:
            self.assertIn(repr(key),content)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'outputs/faithmed-timing-v1').mkdir(parents=True)
            result=subprocess.run(['bash',str(script)],env={**os.environ,'FAITHMED_ROOT':directory},capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('already exists',result.stderr)

    def test_runner_is_bounded_and_monitors_gpu_zero_only(self):
        path=ROOT/'reproduction/run_timing_bounded.py'
        self.assertTrue(path.exists(),'timing runner missing')
        source=path.read_text()
        self.assertIn('7200',source)
        self.assertIn('--id=0',source)
        self.assertNotIn('ray stop',source)
        self.assertIn('cleanup_run(owner)',source)

if __name__=='__main__':unittest.main()
