import json
from pathlib import Path
import unittest
from unittest.mock import patch
from scripts.dtst_run_control import process_running
from scripts.dtst_run_control import stop_decision


class EarlyStoppingTests(unittest.TestCase):
    policy = json.loads((Path(__file__).parents[1]/'configs/dtst_sfuda/early_stopping.json').read_text())

    def curve(self, values):
        return [dict(iteration=step, metrics={'miou': value}) for step, value in values]

    def test_continuing_decline_stops_at_second_bad_eval(self):
        curve = self.curve([(0,.52),(3000,.56),(6000,.58),(9000,.575),(12000,.568)])
        self.assertEqual(stop_decision(curve,self.policy)['iteration'],12000)
        self.assertEqual(stop_decision(curve,self.policy)['peak_iteration'],6000)

    def test_one_drop_recovery_and_small_noise_do_not_stop(self):
        for values in [(.58,.579,.5785),(.58,.575,.583),(.58,.575,.575)]:
            self.assertIsNone(stop_decision(self.curve([(6000,values[0]),(9000,values[1]),(12000,values[2])]),self.policy))

    def test_no_early_or_fixed_endpoint_relabeling(self):
        self.assertIsNone(stop_decision(self.curve([(0,.6),(3000,.5)]),self.policy))
        self.assertIsNone(stop_decision(self.curve([(15000,.6),(18000,.55),(20000,.5)]),self.policy))

    def test_termination_wait_never_reads_process_environment(self):
        with patch('pathlib.Path.read_text',return_value='7 (trainer) Z 1 0 0'):
            self.assertFalse(process_running(7))
        with patch('pathlib.Path.read_text',return_value='7 (trainer) S 1 0 0'):
            self.assertTrue(process_running(7))
        with patch('pathlib.Path.read_text',side_effect=FileNotFoundError):
            self.assertFalse(process_running(7))

    def test_1k_patience_stops_a_low_plateau_after_three_observations(self):
        policy = json.loads((Path(__file__).parents[1]/'configs/dtst_sfuda/early_stopping_1k.json').read_text())
        curve = self.curve([(0,.52),(1000,.56),(2000,.575),(3000,.58),
                            (4000,.575),(5000,.5755),(6000,.576)])
        decision = stop_decision(curve, policy)
        self.assertEqual(decision['iteration'],6000)
        self.assertEqual(decision['peak_iteration'],3000)
        self.assertIsNone(stop_decision(curve[:-1],policy))
        self.assertIsNone(stop_decision(curve[:-1]+self.curve([(6000,.578)]),policy))
        self.assertIsNone(stop_decision(self.curve([(16000,.58),(17000,.575),(18000,.575),
                                                   (19000,.575),(20000,.575)]),policy))


if __name__ == '__main__':
    unittest.main()
