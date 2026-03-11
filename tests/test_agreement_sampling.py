import random
import unittest

from scfg.scfg import SCFG, CFGParams, SCFGParams


class AgreementSamplingTest(unittest.TestCase):
    def setUp(self):
        self.params = SCFGParams(
            a=CFGParams(
                agreement_enabled=True,
                verbs=3,
                nouns=3,
                propns=2,
                prons=2,
                adjs=2,
                det_def=1,
                det_indef=1,
                comps=1,
                rng_seed=11,
            ),
            b=CFGParams(
                agreement_enabled=True,
                verbs=3,
                nouns=3,
                propns=2,
                prons=2,
                adjs=2,
                det_def=1,
                det_indef=1,
                comps=1,
                rng_seed=12,
            ),
        )
        self.scfg = SCFG(self.params)

    def test_sample_contains_agreement_metadata(self):
        sample = self.scfg.sample(min_depth=0, max_depth=1, rng=random.Random(5))
        self.assertTrue(sample["agreement_ok"])
        self.assertIn("subject_features", sample)
        self.assertIn("verb_features", sample)
        self.assertEqual(sample["subject_features"], sample["verb_features"])

    def test_proper_nouns_are_third_person_singular(self):
        for seed in range(50):
            sample = self.scfg.sample(min_depth=0, max_depth=0, rng=random.Random(seed))
            if sample["subject_features"] == {"person": "3", "number": "sg"}:
                self.assertEqual(sample["subject_features"], sample["verb_features"])
                return
        self.fail("Did not encounter a proper-noun subject in 50 draws")

    def test_gender_can_be_sampled_and_preserved(self):
        params = SCFGParams(
            a=CFGParams(
                agreement_enabled=True,
                latent_gender=True,
                realize_gender=False,
                verbs=2,
                nouns=2,
                propns=2,
                prons=2,
                adjs=1,
                det_def=1,
                det_indef=1,
                comps=1,
                rng_seed=21,
            ),
            b=CFGParams(
                agreement_enabled=True,
                latent_gender=True,
                realize_gender=True,
                verbs=2,
                nouns=2,
                propns=2,
                prons=2,
                adjs=1,
                det_def=1,
                det_indef=1,
                comps=1,
                rng_seed=22,
            ),
        )
        scfg = SCFG(params)
        for seed in range(100):
            sample = scfg.sample(min_depth=0, max_depth=0, rng=random.Random(seed))
            if "gender" in sample["subject_features"]:
                self.assertEqual(sample["subject_features"], sample["verb_features"])
                return
        self.fail("Did not encounter a gendered nominal subject in 100 draws")


if __name__ == "__main__":
    unittest.main()
