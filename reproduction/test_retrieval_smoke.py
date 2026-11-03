"""五源检索smoke语料的最小离线检查。"""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


def module():
    path = Path(__file__).with_name('prepare_retrieval_smoke.py')
    if not path.exists():
        raise AssertionError('prepare_retrieval_smoke.py 尚未实现')
    spec = importlib.util.spec_from_file_location('prepare_retrieval_smoke', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


XML = b'''<book-part><book-part-meta><title-group><title>Original title</title></title-group>
<permissions><copyright-statement>Copyright original author</copyright-statement>
<license xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="https://creativecommons.org/licenses/by-nc-nd/4.0/">
<license-p>Attribution NonCommercial NoDerivatives</license-p></license></permissions></book-part-meta>
<body><sec><title>Section</title><p>First original <bold>paragraph</bold>.</p><p>Second original paragraph.</p></sec></body></book-part>'''


class RetrievalSmokeTests(unittest.TestCase):
    def test_medrag_fields_preserved_and_empty_content_fails(self):
        m = module()
        row = {'id':'original-id','title':'Title','content':'A short true text.'}
        result = m.medrag_record('pubmed', row, 'https://source.test/data')
        self.assertEqual(set(result), {'id','title','content','data_source','source_url'})
        self.assertEqual(result['content'], row['content'])
        self.assertEqual(result['data_source'], 'pubmed')
        with self.assertRaises(ValueError):
            m.medrag_record('pubmed', {**row,'content':''}, 'https://source.test/data')

    def test_guidelines_uses_clean_original_paragraphs(self):
        m = module()
        row = {'id':'doc','title':'None','clean_text':'First paragraph.\n\nSecond paragraph.',
               'raw_text':'wrong raw field','url':'https://source.test/guideline'}
        records = m.guideline_records(row, limit=2)
        self.assertEqual([r['content'] for r in records], ['First paragraph.','Second paragraph.'])
        self.assertTrue(all(r['title']=='None' for r in records))
        self.assertEqual(records[0]['source_url'], row['url'])

    def test_statpearls_keeps_original_paragraphs_and_license(self):
        m = module()
        records, metadata, artifacts = m.statpearls_records(XML, 'archive/article-1.nxml', limit=8)
        self.assertEqual([r['content'] for r in records], ['First original paragraph.','Second original paragraph.'])
        self.assertTrue(all(r['data_source']=='statpearls' for r in records))
        self.assertIn('by-nc-nd/4.0', metadata['license'])
        self.assertEqual(artifacts['raw/statpearls/original.nxml'], XML)
        self.assertIn(b'Copyright original author', artifacts['raw/statpearls/permissions.xml'])

    def test_missing_source_or_more_than_eight_fails(self):
        m = module()
        records = [{'id':s,'title':'t','content':'c','data_source':s,'source_url':'https://source.test'}
                   for s in m.SOURCE_ORDER]
        m.validate_records(records)
        with self.assertRaises(ValueError):
            m.validate_records(records[:-1])
        with self.assertRaises(ValueError):
            m.validate_records(records+[dict(records[0], id=f'extra-{i}') for i in range(8)])

    def test_streams_stop_at_subset_and_archive_read_budget(self):
        m = module()
        class Response:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def raise_for_status(self):
                pass
            def iter_lines(self, **kwargs):
                for i in range(8):
                    yield json.dumps({'id':str(i),'title':'title','content':'real source text'}).encode()
                raise AssertionError('stream read beyond selected subset')
        with patch('requests.get', return_value=Response()) as request:
            records, metadata = m.fetch_hf_records('pubmed', 8)
            self.assertEqual(len(records), 8)
            self.assertTrue(request.call_args.kwargs['stream'])
            self.assertEqual(metadata['revision'], m.HF_SOURCES['pubmed']['revision'])
        reader = m.LimitedReader(io.BytesIO(b'123456789'), 5)
        self.assertEqual(reader.read(100), b'12345')
        with self.assertRaises(ValueError):
            reader.read(1)

    def test_atomic_publication_and_scope(self):
        m = module()
        def hf(source, limit):
            return [{'id':source,'title':'title','content':'true text','data_source':source,'source_url':'https://source.test'}], {'revision':'fixed','license':'upstream'}
        def stat(limit):
            return m.statpearls_records(XML, 'archive/article-1.nxml', limit)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp)/'smoke'
            manifest = m.prepare_smoke(output, hf_loader=hf, stat_loader=stat)
            self.assertEqual(manifest['scope'], 'integration_smoke_subset')
            self.assertEqual(set(manifest['source_counts']), set(m.SOURCE_ORDER))
            self.assertEqual((output/'raw/statpearls/original.nxml').read_bytes(), XML)
            rows = [json.loads(line) for line in (output/'corpus.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 6)
            self.assertEqual(json.loads((output/'manifest.json').read_text())['scope'], 'integration_smoke_subset')
            failed = Path(temp)/'failed'
            def broken(source, limit):
                raise RuntimeError('source download failed')
            with self.assertRaises(RuntimeError):
                m.prepare_smoke(failed, hf_loader=broken, stat_loader=stat)
            self.assertFalse(failed.exists())
            self.assertFalse(list(Path(temp).glob('.failed.partial-*')))


if __name__ == '__main__':
    unittest.main()
