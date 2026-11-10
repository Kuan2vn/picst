import json
from pathlib import Path
import unittest
import numpy as np
import torch
from scripts.gsd_scout import project_probabilities, macro_regions, contradiction, filter_bank_components
from scripts.dtst_run_control import stop_decision


class ScoutTests(unittest.TestCase):
    def test_projection_identity_and_partial_support(self):
        p=torch.randn(1,6,8,8).softmax(1)
        g=dict(native_size=32,native_x=0,native_y=0,view_size=32,view_x=0,view_y=0)
        projected,valid=project_probabilities(p,[g],(8,8))
        torch.testing.assert_close(projected,p)
        self.assertTrue(valid.all())
        q,valid=project_probabilities(p,[dict(g,view_size=64,view_x=-16,view_y=-16)],(8,8))
        self.assertEqual(int(valid.sum()),16)
        torch.testing.assert_close(q.sum(1)[valid],torch.ones(16))

    def test_macro_small_and_truncated_are_neutral(self):
        labels=np.zeros((1,20,20),np.uint8);labels[:,5:15,5:15]=3;labels[:,2:4,2:4]=4
        result=macro_regions(labels,np.ones_like(labels,bool),1,10)
        self.assertTrue(result[0,9,9]);self.assertFalse(result[0,2,2]);self.assertFalse(result[0,0,0])

    def test_source_contradiction_and_component_gate(self):
        source=torch.tensor([[[[.94,.94,.04]],[[.02,.02,.94]],[[.01,.01,.005]],
                             [[.01,.01,.005]],[[.01,.01,.005]],[[.01,.01,.005]]]])
        labels=torch.tensor([[[1,0,255]]])
        self.assertEqual(contradiction(source,labels).tolist(),[[[True,False,False]]])
        mask=np.array([[1,1,255,2,2]],np.uint8)
        veto=np.array([[True,False,False,False,False]])
        self.assertEqual(filter_bank_components(mask,veto,.25).tolist(),[[255,255,255,2,2]])

    def test_scout_two_nonimprovements_ties_and_recovery(self):
        policy=json.loads((Path(__file__).parents[1]/'configs/dtst_sfuda/early_stopping_scout_2eval.json').read_text())
        def curve(values):return [dict(iteration=i*1000,metrics={'miou':v}) for i,v in enumerate(values)]
        for values in ([.52,.55,.55,.55],[.52,.55,.54,.549],[.55,.54,.54]):
            self.assertIsNotNone(stop_decision(curve(values),policy))
        for values in ([.52,.55,.54],[.52,.55,.54,.551],[.52,.55,.55001,.55]):
            self.assertIsNone(stop_decision(curve(values),policy))

if __name__=='__main__':unittest.main()
