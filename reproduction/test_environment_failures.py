"""Execute real small environment methods without importing Ray/gym model runners."""
import ast
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
ENV=ROOT/'verl-agent/agent_system/environments/env_package/medrm'

def method(filename,name,scope=None):
    tree=ast.parse((ENV/filename).read_text())
    node=next(n for cls in tree.body if isinstance(cls,ast.ClassDef) for n in cls.body
              if isinstance(n,ast.FunctionDef) and n.name==name)
    namespace={} if scope is None else scope
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(ENV/filename),'exec'),namespace)
    return namespace[name]

class EnvironmentFailures(unittest.TestCase):
    def test_medcorp_error_is_not_evidence_text(self):
        def unavailable(*args,**kwargs):raise RuntimeError('offline')
        search=method('env.py','_search',{'_SEARCH_SEMAPHORE':threading.Semaphore(1),'query_medcorp':unavailable})
        with self.assertRaises(RuntimeError):search(SimpleNamespace(search_engine='medcorp',num_docs=6),'q')

    def test_step_and_episode_judge_errors_propagate(self):
        from rewards import rubrics_judge,step_rubric_judge
        rubrics=[{'id':'R1','type':'binary'}]
        for function,step_mode in [('compute_process_rewards',False),('compute_step_process_rewards',True)]:
            future=Future();future.set_exception(RuntimeError('judge failed'))
            env=SimpleNamespace(_rubric_future=future,_step_rubric_futures=[future],
                _step_features=[{'full_text':'q'}],log_rubrics=lambda *a:None,log_step_rubrics=lambda *a:None)
            group=SimpleNamespace(num_processes=1,_actual_n=1,workers=[SimpleNamespace(env=env)],
                                  group_n=1,config={'step_scoring':step_mode})
            with patch.object(rubrics_judge,'load_active_rubrics',return_value=rubrics),patch.object(step_rubric_judge,'load_step_rubrics',return_value=rubrics):
                with self.subTest(function=function),self.assertRaises(RuntimeError):method('envs.py',function)(group)

if __name__=='__main__':unittest.main()
