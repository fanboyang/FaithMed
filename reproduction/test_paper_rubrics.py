from collections import Counter
from pathlib import Path
import unittest
import yaml
from rewards import rubrics_judge,step_rubric_judge

ROOT=Path(__file__).resolve().parents[1]

class PaperRubricTests(unittest.TestCase):
    def setUp(self):
        rubrics_judge._rubrics_cache=None
        step_rubric_judge._step_rubrics_cache=None

    def test_both_loaders_use_paper_fifteen_binary_rubrics(self):
        episode=rubrics_judge.load_active_rubrics()
        steps=step_rubric_judge.load_step_rubrics()
        self.assertEqual(episode,steps)
        self.assertEqual(len(steps),15)
        self.assertEqual(Counter(r['dimension'] for r in steps),{1:3,2:3,3:3,4:3,5:3})
        self.assertTrue(all(r['type']=='binary' and r['scale_labels']==['UNMET','MET'] for r in steps))
        source=yaml.safe_load((ROOT/'rubric-refining/rubrics_step.yaml').read_text())['source']
        self.assertEqual(source['tables'],[6,7])
        self.assertEqual(source['pdf_pages'],[13,14])
        self.assertEqual(steps[2]['name'],'Question specificity')
        self.assertEqual(steps[5]['name'],'Query constraint preservation')
        self.assertIn('No explicit hypothesis-checking or self-correction pattern appears before the final answer.',steps[13]['description'])

    def test_step_applicability_and_binary_scores(self):
        rubrics=step_rubric_judge.load_step_rubrics()
        seen=[]
        def score(features,applicable):
            seen.append([r['dimension'] for r in applicable])
            return {r['id']:1 for r in applicable},None
        reset,_=step_rubric_judge.score_step('<think>q</think><search>x</search>','<search>x</search>',[],
                                            'reset',rubrics=rubrics,score_one_fn=score)
        self.assertEqual(set(seen[-1]),{1,2})
        self.assertEqual(sum(v==1 for v in reset.values()),6)
        final,_=step_rubric_judge.score_step('<think>x</think><answer>A</answer>','<answer>A</answer>',
                                            ['[T1-R1] evidence'],'answer',rubrics=rubrics,score_one_fn=score)
        self.assertEqual(set(seen[-1]),{3,4,5})
        self.assertEqual(sum(v==1 for v in final.values()),9)
        invalid,_=step_rubric_judge.score_step('invalid',None,[],'invalid',rubrics=rubrics,score_one_fn=score)
        self.assertTrue(all(v==-1 for v in invalid.values()))

if __name__=='__main__':unittest.main()
