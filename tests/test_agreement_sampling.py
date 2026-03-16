import random
import unittest
from typing import Any, cast

from scfg.agreement import FeatureBundle
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

    def test_verb_sampling_uses_aligned_paradigm_entries(self):
        self.assertIsNotNone(self.params.a.verb_paradigms)
        self.assertIsNotNone(self.params.b.verb_paradigms)
        a_verb_paradigms = cast(list[dict[str, Any]], self.params.a.verb_paradigms)
        b_verb_paradigms = cast(list[dict[str, Any]], self.params.b.verb_paradigms)
        plural_third_key = FeatureBundle(person="3", number="pl").key(
            self.params.a.latent_axes
        )
        expected_pairs = {
            (
                a_verb_paradigms[index]["forms"][plural_third_key],
                b_verb_paradigms[index]["forms"][plural_third_key],
            )
            for index in range(
                min(
                    len(a_verb_paradigms),
                    len(b_verb_paradigms),
                )
            )
        }
        for seed in range(20):
            derivation = self.scfg._sample_agreement_recursive(
                "V",
                rng=random.Random(seed),
                current_depth=0,
                min_depth=0,
                max_depth=0,
                inherited_features=FeatureBundle(person="3", number="pl"),
            )
            self.assertIn((derivation.left_full, derivation.right_full), expected_pairs)

    def test_simple_lexical_categories_use_aligned_pairs(self):
        adj_pairs = set(zip(self.params.a.adj_lex, self.params.b.adj_lex))
        comp_pairs = set(zip(self.params.a.comp_lex, self.params.b.comp_lex))
        tense_pairs = set(zip(self.params.a.tense_lex, self.params.b.tense_lex))
        det_pairs = set(zip(self.params.a.det_def_lex, self.params.b.det_def_lex))

        for seed in range(10):
            rng = random.Random(seed)
            self.assertIn(self.scfg._choose_aligned_adj(rng), adj_pairs)
            self.assertIn(self.scfg._choose_aligned_comp(rng), comp_pairs)
            self.assertIn(self.scfg._choose_aligned_tense(rng), tense_pairs)
            self.assertIn(self.scfg._choose_aligned_det(rng, True), det_pairs)

    def test_possible_right_answers_expand_when_source_collapses_number(self):
        params = SCFGParams(
            a=CFGParams(
                agreement_enabled=False,
                verbs=2,
                nouns=2,
                propns=2,
                prons=6,
                adjs=1,
                det_def=1,
                det_indef=1,
                comps=1,
                rng_seed=31,
            ),
            b=CFGParams(
                agreement_enabled=True,
                verbs=2,
                nouns=2,
                propns=2,
                prons=6,
                adjs=1,
                det_def=1,
                det_indef=1,
                comps=1,
                rng_seed=32,
            ),
        )
        scfg = SCFG(params)
        left_surface, _ = scfg._noun_entry(params.a, 0, "sg")
        possible = scfg._possible_noun_right_surfaces(0, left_surface)
        self.assertEqual(2, len(possible))
        self.assertNotEqual(possible[0], possible[1])


if __name__ == "__main__":
    unittest.main()
