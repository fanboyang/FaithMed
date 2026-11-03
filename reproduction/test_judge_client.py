from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError
from io import BytesIO
from reproduction.judge_client import JudgeClient, JudgeResponseError, JudgeTransportError, JudgeInFlightError

RUBRICS=[{'id':'R1','type':'binary'},{'id':'R2','type':'ordinal3'}]
def raw(content='{"R1":1,"R2":{"score":2}}'):
    return json.dumps({'model':'gpt-6-luna','choices':[{'message':{'content':content}}],
                       'usage':{'prompt_tokens':10,'completion_tokens':5}})

class JudgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.db=Path(self.tmp.name)/'judge.sqlite'
        self.client=JudgeClient('test-secret',self.db,max_attempts=3)

    def test_success_resume_exact_request_and_audit_without_key(self):
        with patch('reproduction.judge_client.post_raw',return_value=raw()) as request:
            self.assertEqual(self.client.score('p',RUBRICS),{'R1':1,'R2':2})
            self.assertEqual(JudgeClient('other-secret',self.db,timeout=7,max_attempts=1).score('p',RUBRICS),{'R1':1,'R2':2})
            self.assertEqual(request.call_count,1)
            body=request.call_args.args[1]
            self.assertEqual(body['model'],'gpt-6-luna')
            self.assertEqual(body['reasoning_effort'],'none')
            self.assertEqual(body['temperature'],0)
        with closing(sqlite3.connect(self.db)) as conn:
            audit=str(conn.execute('SELECT * FROM requests').fetchall())+str(conn.execute('SELECT * FROM attempts').fetchall())
        self.assertNotIn('test-secret',audit)
        self.assertIn('completion_tokens',audit)
        self.assertIn('gpt-6-luna',audit)

    def test_missing_or_invalid_score_is_terminal_without_resampling(self):
        for content in ['not JSON','{"R1":1}','{"R1":4,"R2":2}','{"R1":0.5,"R2":2}']:
            with self.subTest(content=content):
                prompt=content
                with patch('reproduction.judge_client.post_raw',return_value=raw(content)) as request:
                    with self.assertRaises(JudgeResponseError):self.client.score(prompt,RUBRICS)
                    self.assertEqual(request.call_count,1)
                with patch('reproduction.judge_client.post_raw') as request:
                    with self.assertRaises(JudgeResponseError):JudgeClient('test-secret',self.db).score(prompt,RUBRICS)
                    request.assert_not_called()

    def test_transient_only_retries(self):
        for failure in [URLError('offline'),HTTPError('local',503,'busy',{},BytesIO(b'busy')),
                        HTTPError('local',429,'limit',{},BytesIO(b'rate limit'))]:
            with self.subTest(failure=failure):
                prompt=str(type(failure))+str(getattr(failure,'code','net'))
                with patch('reproduction.judge_client.post_raw',side_effect=[failure,raw()]) as request, patch('reproduction.judge_client.time.sleep'):
                    self.assertEqual(self.client.score(prompt,RUBRICS)['R1'],1)
                    self.assertEqual(request.call_count,2)
        with patch('reproduction.judge_client.post_raw',side_effect=HTTPError('local',400,'bad',{},BytesIO(b'bad request'))) as request:
            with self.assertRaises(JudgeTransportError):self.client.score('400',RUBRICS)
            self.assertEqual(request.call_count,1)
        with patch('reproduction.judge_client.post_raw') as request:
            with self.assertRaises(JudgeResponseError):self.client.score('400',RUBRICS)
            request.assert_not_called()

    def test_exhausted_transport_can_resume_without_resampling_valid_outputs(self):
        with patch('reproduction.judge_client.post_raw',side_effect=URLError('offline')) as request, patch('reproduction.judge_client.time.sleep'):
            with self.assertRaises(JudgeTransportError):self.client.score('retry',RUBRICS)
            self.assertEqual(request.call_count,3)
        with patch('reproduction.judge_client.post_raw',return_value=raw()):
            self.assertEqual(self.client.score('retry',RUBRICS)['R2'],2)

    def test_identical_concurrent_request_waits_and_reuses_one_response(self):
        entered=threading.Event()
        release=threading.Event()
        def slow_response(*args):
            entered.set()
            if not release.wait(timeout=3):raise AssertionError('test release timeout')
            return raw()
        with patch('reproduction.judge_client.post_raw',side_effect=slow_response) as request:
            with ThreadPoolExecutor(max_workers=2) as workers:
                first=workers.submit(self.client.score,'same',RUBRICS)
                self.assertTrue(entered.wait(timeout=3))
                second=workers.submit(self.client.score,'same',RUBRICS)
                release.set()
                self.assertEqual(first.result(timeout=3),second.result(timeout=3))
            self.assertEqual(request.call_count,1)

    def test_pending_request_cannot_be_resampled(self):
        self.client.begin('pending-key','{}')
        with self.assertRaises(JudgeInFlightError):self.client.begin('pending-key','{}')

    def test_legitimate_na_and_fenced_nested_output_remain_supported(self):
        with patch('reproduction.judge_client.post_raw',return_value=raw('```json\n{"R1":-1,"R2":{"score":"2"}}\n```')):
            self.assertEqual(self.client.score('na',RUBRICS),{'R1':-1,'R2':2})

    def test_environment_requires_audited_destination_and_key(self):
        from reproduction.judge_client import from_environment
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaises(ValueError):from_environment()
        with patch.dict(os.environ,{'FAITHMED_JUDGE_AUDIT_DIR':self.tmp.name},clear=True):
            with self.assertRaises(ValueError):from_environment()

    def test_rubric_prompt_preconditions_preserved_and_missing_is_not_na(self):
        from rewards import rubrics_judge as rubric
        specs=[{'id':'R1','type':'binary','name':'a','description':'b','scale_labels':{0:'no',1:'yes'}},
               {'id':'skip','type':'binary','precondition':'search_performed','name':'a','description':'b','scale_labels':{0:'no',1:'yes'}}]
        features={'full_text':'reasoning','search_performed':False}
        with patch.object(rubric,'_get_judge_client',return_value=self.client),patch('reproduction.judge_client.post_raw',return_value=raw('{"R1":1}')):
            result,_=rubric._score_one(features,specs)
            self.assertEqual(result,{'R1':1,'skip':-1})
        with patch.object(rubric,'_get_judge_client',return_value=self.client),patch('reproduction.judge_client.post_raw',return_value=raw('{}')):
            with self.assertRaises(JudgeResponseError):rubric._score_one({'full_text':'other','search_performed':False},specs)

if __name__=='__main__':unittest.main()
