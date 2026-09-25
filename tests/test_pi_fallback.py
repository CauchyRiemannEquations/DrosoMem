import json
import unittest
from pathlib import Path
import numpy as np
from flying.data.pi_digits import decimal_pi_digits,pi_digits


class PiFallbackTests(unittest.TestCase):
    def test_matches_previously_recorded_training_digits(self):
        root=Path(__file__).resolve().parents[1]
        row=json.loads((root/'results/phase5_bptt/recalls.jsonl').read_text().splitlines()[0])
        expected=np.array([int(c) for c in row['prompt']+row['target']])
        self.assertTrue(np.array_equal(decimal_pi_digits(200),expected))
        self.assertTrue(np.array_equal(pi_digits(200),expected))

    def test_known_prefix_and_precision_consistency(self):
        known='314159265358979323846264338327950288419716939937510'
        self.assertEqual(''.join(map(str,decimal_pi_digits(len(known)))),known)
        self.assertTrue(np.array_equal(decimal_pi_digits(200),decimal_pi_digits(1000)[:200]))

    def test_invalid_requests(self):
        for invalid in [0,1,-5,3.5]:
            with self.assertRaises(ValueError):decimal_pi_digits(invalid)
            with self.assertRaises(ValueError):pi_digits(invalid)
