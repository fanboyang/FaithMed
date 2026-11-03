"""FaithMed no-SFT数据入口最小离线检查。"""
import importlib.util
import ast
from types import SimpleNamespace
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from unittest.mock import patch


def module():
    path = Path(__file__).resolve().parents[1]/'data_preparation/medmix_preparation.py'
    spec = importlib.util.spec_from_file_location('faithmed_prepare', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class PrepareDataTests(unittest.TestCase):
    def test_required_split_is_never_silently_ignored(self):
        m = module()
        with self.assertRaisesRegex(ValueError, 'validation'):
            m.select_splits({'train': [1]}, ['train', 'validation'], 'source')

    def test_all_seven_evaluation_and_four_training_sources_required(self):
        m = module()
        tests = {s['name']: 1 for s in m.DATASET_SPECS}
        train = {s['name']: 1 for s in m.DATASET_SPECS if s['train_splits']}
        m.validate_source_counts(train, tests)
        tests.pop('TsinghuaC3I/MedXpertQA')
        with self.assertRaises(ValueError):
            m.validate_source_counts(train, tests)
        tests['TsinghuaC3I/MedXpertQA'] = 0
        with self.assertRaises(ValueError):
            m.validate_source_counts(train, tests)

    def test_headqa_official_archive_and_answer_mapping(self):
        m = module()
        question = {'qid': '3', 'qtext': 'Which choice?', 'ra': '2', 'image': '',
                    'answers': [{'aid': '2', 'atext': 'two'}, {'aid': '1', 'atext': 'one'}]}
        exam = {'name': 'exam', 'year': '2017', 'category': 'medicine', 'data': [question, {**question, 'qid': '4', 'image': 'HEAD/image.png'}]}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'official.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                for prefix in ['train', 'dev', 'test']:
                    archive.writestr(f'HEAD_EN/{prefix}_HEAD_EN.json', json.dumps({'exams': {'exam': exam}}))
            splits = m.load_headqa_archive(path)
            self.assertEqual(set(splits), {'train', 'validation', 'test'})
            self.assertIsNone(splits['train'][0]['image'])
            self.assertEqual(len(m.filter_headqa_no_image(splits['train'])), 1)
            spec = next(s for s in m.DATASET_SPECS if s['name']=='dvilares/head_qa')
            converted = m.map_validated_row(spec, splits['train'][0], 0, 'train', 'train')
            self.assertEqual(converted['reward_model']['ground_truth'], 'B')
            self.assertIn('A one; B two', converted['prompt'][0]['content'])
            self.assertNotIn('ground_truth', converted['prompt'][0]['content'])

    def test_full_no_sft_pipeline_and_atomic_publication(self):
        m = module()
        from datasets import Dataset, DatasetDict
        def load(spec, **kwargs):
            name = spec['name']
            if name == 'openlifescienceai/medqa':
                row = {'id':'q1','data':{'Question':'short same question','Correct Option':'A','Options':{'A':'a','B':'b'}}}
            elif name == 'dvilares/head_qa':
                row = {'name':'exam','year':'2017','category':'medicine','qid':1,'qtext':'short head question',
                       'ra':1,'image':None,'answers':[{'aid':1,'atext':'a'},{'aid':2,'atext':'b'}]}
            elif name == 'openlifescienceai/medmcqa':
                row = {'id':'q1','question':'short mc question','opa':'a','opb':'b','opc':'c','opd':'d','cop':0}
            elif name == 'ncbi/MedCalc-Bench':
                row = {'Question':'calculate value','Patient Note':'patient note','Ground Truth Answer':'1','Note ID':'n1'}
            elif name == 'JesseLiu/medbulltes5op':
                row = {'question':'short bullets question','answer_idx':'A', **{'choices'+c:c for c in 'ABCDE'}}
            elif name == 'TIGER-Lab/MMLU-Pro':
                row = {'question_id':0,'question':'short health question','options':['a','b'],'answer':'A','answer_index':0,'category':'health'}
            else:
                row = {'id':'Text-1','question':'short expert question Answer Choices: (A) a (B) b','options':{'A':'a','B':'b'},'label':'A'}
            result = {}
            for split in dict.fromkeys(spec['train_splits']+spec['test_splits']):
                count = 1001 if name == 'JesseLiu/medbulltes5op' else 1
                result[split] = Dataset.from_list([row]*count)
            return DatasetDict(result)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'prepared'
            manifest = m.prepare_data(output, loader=load)
            self.assertEqual(len(manifest['test_counts']), 7)
            self.assertEqual(manifest['test_counts']['JesseLiu/medbulltes5op'], 1001)
            self.assertEqual(manifest['train_rows'], 6)
            self.assertIsNone(manifest['test_limit'])
            self.assertGreater(manifest['deduplication']['audit_final']['exact_overlap_train_rows'], 0)
            self.assertEqual(manifest['deduplication']['removed_train_rows'], 0)
            from pyarrow.parquet import ParquetFile
            self.assertEqual(ParquetFile(output/'test.parquet').metadata.num_rows, 1007)
            self.assertEqual(json.loads((output/'manifest.json').read_text())['status'], 'complete')
            with self.assertRaisesRegex(ValueError, 'not empty'):
                m.prepare_data(output, loader=load)
            failed = Path(temp)/'failed'
            with patch.object(m, 'load_source', side_effect=RuntimeError('download failed')):
                with self.assertRaises(RuntimeError):
                    m.prepare_data(failed)
            self.assertFalse(failed.exists())
            original_write = Dataset.to_parquet
            def fail_second_write(dataset, path, *args, **kwargs):
                if Path(path).name == 'test.parquet':
                    raise RuntimeError('test parquet write failed')
                return original_write(dataset, path, *args, **kwargs)
            with patch.object(Dataset, 'to_parquet', new=fail_second_write):
                with self.assertRaisesRegex(RuntimeError, 'write failed'):
                    m.prepare_data(failed, loader=load)
            self.assertFalse(failed.exists())
            self.assertFalse(list(Path(temp).glob('.failed.partial-*')))

    def test_generated_env_kwargs_reaches_real_environment_reset(self):
        m = module()
        spec = next(s for s in m.DATASET_SPECS if s['name']=='openlifescienceai/medqa')
        row = {'id':'q1','data':{'Question':'q','Options':{'A':'a','B':'b'},'Correct Option':'A'}}
        converted = m.map_validated_row(spec, row, 3, 'dev', 'train')
        environment = converted['env_kwargs']
        self.assertEqual(set(environment), {'question','question_id','ground_truth','data_source','extra_info'})
        self.assertEqual(environment['question_id'], converted['extra_info']['uid'])
        self.assertEqual(environment['ground_truth'], 'A')
        self.assertEqual(environment['question'], converted['extra_info']['question'])
        source = Path(__file__).resolve().parents[1]/'verl-agent/agent_system/environments/env_manager.py'
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='MedRMEnvironmentManager')
        reset = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name=='reset')
        future = ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
        program = ast.fix_missing_locations(ast.Module(body=[future,reset],type_ignores=[]))
        scope = {}
        exec(compile(program,str(source),'exec'),scope)
        seen = []
        def receive(*args):
            seen.append(args)
            return ['observation'], [{}]
        obs, infos = scope['reset'](SimpleNamespace(envs=SimpleNamespace(reset=receive)), [environment])
        self.assertEqual(seen[0][1], [converted['extra_info']['uid']])
        self.assertEqual(seen[0][2], ['A'])
        self.assertEqual(obs['text'], ['observation'])

    def test_adapter_preserves_existing_parquet_columns_and_manifest(self):
        path = Path(__file__).with_name('prepare_rl_adapter.py')
        if not path.exists():
            raise AssertionError('prepare_rl_adapter.py 尚未实现')
        spec = importlib.util.spec_from_file_location('prepare_rl_adapter', path)
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        import pyarrow as pa
        import pyarrow.parquet as pq
        row = {'data_source':'openlifescienceai/medqa','prompt':[{'role':'user','content':'original prompt'}],
               'reward_model':{'style':'rule','ground_truth':'A'},
               'extra_info':{'qid':'q','uid':'stable:dev:3','question':'original question','source_split':'dev'}}
        with tempfile.TemporaryDirectory() as temp:
            source, output = Path(temp)/'medmix', Path(temp)/'medmix_rl'
            source.mkdir()
            from datasets import Dataset
            original = Dataset.from_list([row]).data.table
            for split in ['train','test']:
                pq.write_table(original,source/f'{split}.parquet')
            metadata = {'status':'complete','protocol':'original protocol','train_rows':1,'test_rows':1,
                        'sources':[{'name':'original','revision':'pinned'}]}
            (source/'manifest.json').write_text(json.dumps(metadata))
            result = adapter.prepare_adapter(source, output)
            self.assertEqual(result['protocol'], metadata['protocol'])
            self.assertEqual(result['sources'], metadata['sources'])
            self.assertEqual(json.loads((output/'source_manifest.json').read_text()), metadata)
            for split in ['train','test']:
                actual = pq.read_table(output/f'{split}.parquet')
                self.assertTrue(actual.drop(['env_kwargs']).equals(original))
                from datasets import Features
                self.assertIn('env_kwargs', Features.from_arrow_schema(actual.schema))
                environment = actual['env_kwargs'].to_pylist()[0]
                self.assertEqual(environment['question_id'], 'stable:dev:3')
                self.assertEqual(environment['extra_info'], row['extra_info'])
                self.assertNotIn('env_kwargs', pq.read_schema(source/f'{split}.parquet').names)
            with self.assertRaises(ValueError):
                adapter.prepare_adapter(source, output)
            invalid = {**row, 'extra_info': {**row['extra_info'], 'uid': None}}
            pq.write_table(pa.Table.from_pylist([invalid]),source/'test.parquet')
            failed = Path(temp)/'failed_rl'
            with self.assertRaisesRegex(ValueError, 'uid'):
                adapter.prepare_adapter(source, failed)
            self.assertFalse(failed.exists())
            self.assertFalse(list(Path(temp).glob('.failed_rl.partial-*')))

    def test_invalid_gold_is_explicit_error(self):
        m = module()
        spec = next(s for s in m.DATASET_SPECS if s['name']=='openlifescienceai/medqa')
        row = {'id': 'q1', 'data': {'Question': 'q', 'Options': {'A':'a','B':'b'}, 'Correct Option':'Z'}}
        with self.assertRaisesRegex(ValueError, 'label'):
            m.map_validated_row(spec, row, 0, 'train', 'train')


if __name__ == '__main__':
    unittest.main()
