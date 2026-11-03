import base64
import json
import unittest
import numpy as np
from reproduction.retrieval_service_smoke import ranked_results

class RetrievalSmokeTests(unittest.TestCase):
    def test_rank_and_wire_format(self):
        docs=[dict(id=str(i),title='Title',content='Text '+str(i),data_source='pubmed',source_url='https://example.org') for i in range(3)]
        result=ranked_results(docs,np.array([[0.,1.],[1.,0.],[.6,.8]]),np.array([1.,0.]),2)
        self.assertEqual(result['scores'],[1.,.6])
        decoded=json.loads(base64.b64decode(result['results'][0]))
        self.assertEqual(decoded['Clean-Text'],'Text 1')
        self.assertEqual(decoded['data_source'],'pubmed')
        self.assertEqual(result['scope'],'integration_smoke_subset_exact_search')
    def test_invalid_vectors_fail(self):
        with self.assertRaises(ValueError):ranked_results([{}],np.array([[float('nan'),1]]),np.array([1.,0.]),1)
        with self.assertRaises(ValueError):ranked_results([{}],np.array([[1.,0.]]),np.array([1.,0.]),0)

if __name__=='__main__':unittest.main()
