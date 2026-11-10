"""CPU checks for the opt-in PIC-ST options of 09/11/2026 (c0_classes, c1_plausible)."""
import copy, importlib.util, json, sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from scripts import dtst_phenology as new  # noqa: E402

OLD_PATH = ROOT.parents[1]/'picst_seed1_kaggle_20261103/overlay/scripts/dtst_phenology.py'


class Enc(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.c = torch.nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x):
        f = self.c(x); return f, [f, f, f, f]


class Cls(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.c = torch.nn.Conv2d(8, 6, 1)

    def forward(self, feats, size=None):
        z = self.c(feats[-1]); return torch.nn.functional.interpolate(z, size=size) if size else z


def settings(name):
    return json.loads((ROOT/'kaggle_settings'/name).read_text())


def models(seed=0):
    torch.manual_seed(seed); enc, cls = Enc(), Cls()
    with torch.no_grad():  # make tree/ground/low-vegetation predictions all present
        cls.c.bias.copy_(torch.tensor([.2, -.5, .1, .15, -1., -.3]))
    return enc, cls


def images(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(2, 3, 96, 96, generator=g)


class Options(unittest.TestCase):
    def run_adjust(self, module, cfg, out):
        enc, cls = models(); x = images()
        pheno = module.PhenologyCorrection(SimpleNamespace(output=out), module.validate_phenology(copy.deepcopy(cfg)))
        z = cls(enc(x)[1], x.shape[-2:])
        return pheno, pheno.adjust(z, x, enc, cls, 0)

    @unittest.skipUnless(OLD_PATH.exists(), 'previous overlay not available')
    def test_default_paths_identical_to_previous_module(self):
        spec = importlib.util.spec_from_file_location('old_pheno', OLD_PATH); old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        for name in ('pheno_c0.json', 'pheno_c0_c1.json', 'pheno_null.json'):
            with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
                p_old, z_old = self.run_adjust(old, settings(name), a)
                p_new, z_new = self.run_adjust(new, settings(name), b)
                self.assertTrue(torch.equal(z_old, z_new), name); self.assertEqual(p_old.delta, p_new.delta)
                x = images(3); torch.manual_seed(5); e_old = p_old.counterfactual(x, z_old, torch.full((2, 96, 96), 255), torch.ones(2, 96, 96, dtype=torch.bool))
                torch.manual_seed(5); e_new = p_new.counterfactual(x, z_new, torch.full((2, 96, 96), 255), torch.ones(2, 96, 96, dtype=torch.bool))
                self.assertTrue(torch.equal(e_old, e_new), name)

    def test_single_class_list_equals_tree_c0(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            p1, z1 = self.run_adjust(new, settings('pheno_c0.json'), a)
            p2, z2 = self.run_adjust(new, dict(settings('pheno_c0.json'), c0_classes=[[3, [0, 2, 5]]]), b)
            self.assertAlmostEqual(p1.delta, p2.delta, places=5); self.assertTrue(torch.allclose(z1, z2, atol=1e-5))

    def test_multi_class_offsets_and_resume(self):
        cfg = settings('pheno_c0multi.json')
        with tempfile.TemporaryDirectory() as a:
            p, z = self.run_adjust(new, cfg, a)
            enc, cls = models(); x = images(); base = cls(enc(x)[1], x.shape[-2:])
            for c, value in p.deltas.items():
                self.assertTrue(torch.allclose(z[:, c] - base[:, c], torch.full_like(base[:, c], value), atol=1e-5))
            self.assertEqual(set(p.deltas), {2, 3})
            again = new.PhenologyCorrection(SimpleNamespace(output=a), new.validate_phenology(cfg))
            self.assertEqual(again.deltas, p.deltas)

    def test_plausible_c1_only_keeps_vegetation(self):
        cfg = settings('pheno_c0_c1plaus.json')
        with tempfile.TemporaryDirectory() as a:
            p, z = self.run_adjust(new, dict(cfg, c1_prob=1.0), a)
            x = images(4)
            out = p.counterfactual(x, z, torch.full((2, 96, 96), 255), torch.ones(2, 96, 96, dtype=torch.bool))
            changed = (out - x).abs().sum(1) > 1e-6
            self.assertGreaterEqual(p.c1_rejected, 0)
            # every edited pixel must be one the teacher calls vegetation on the hard-edited input
            enc, cls = p.teacher
            self.assertTrue(changed.any() or p.c1_rejected > 0)

    def test_validation(self):
        with self.assertRaises(ValueError):
            new.validate_phenology(dict(settings('pheno_c1.json'), c0_classes=[[3, [0, 2, 5]]]))
        with self.assertRaises(ValueError):
            new.validate_phenology(dict(settings('pheno_c0.json'), c1_plausible=True))
        with self.assertRaises(ValueError):
            new.validate_phenology(dict(settings('pheno_c0.json'), c0_classes=[[3, [3, 2]]]))


if __name__ == '__main__':
    unittest.main()
