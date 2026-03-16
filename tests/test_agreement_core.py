import unittest
from unittest.mock import patch

from scfg.agreement import FeatureBundle, FeatureUnifier
from scfg.scfg import CFGParams


class AgreementCoreTest(unittest.TestCase):
    def test_feature_bundle_round_trip(self):
        bundle = FeatureBundle(person="3", number="pl")
        self.assertEqual(bundle, FeatureBundle.from_dict(bundle.to_dict()))

    def test_feature_unifier_detects_conflict(self):
        left = FeatureBundle(person="3", number="sg")
        right = FeatureBundle(person="3", number="pl")
        self.assertIsNone(FeatureUnifier.unify(left, right))

    def test_feature_unifier_merges_underspecified_bundle(self):
        left = FeatureBundle(person="1")
        right = FeatureBundle(number="pl")
        merged = FeatureUnifier.unify(left, right)
        self.assertEqual(merged, FeatureBundle(person="1", number="pl"))

    def test_agreement_paradigms_are_built(self):
        params = CFGParams(
            agreement_enabled=True,
            verbs=2,
            nouns=2,
            propns=2,
            prons=2,
            adjs=1,
            det_def=1,
            det_indef=1,
            comps=1,
        )
        self.assertEqual(6, len(params.pronoun_paradigms))
        self.assertEqual(2, len(params.verb_paradigms))
        self.assertEqual(2, len(params.noun_paradigms))
        self.assertIn("number=sg", params.noun_paradigms[0]["forms"])
        self.assertIn("number=pl", params.noun_paradigms[0]["forms"])
        self.assertEqual(6, len(params.verb_paradigms[0]["forms"]))

    def test_gender_can_be_latent_without_surface_realization(self):
        hidden = CFGParams(
            agreement_enabled=True,
            latent_gender=True,
            realize_gender=False,
            verbs=1,
            nouns=1,
            propns=1,
            prons=2,
            adjs=1,
            det_def=1,
            det_indef=1,
            comps=1,
        )
        overt = CFGParams(
            agreement_enabled=True,
            latent_gender=True,
            realize_gender=True,
            verbs=1,
            nouns=1,
            propns=1,
            prons=2,
            adjs=1,
            det_def=1,
            det_indef=1,
            comps=1,
        )
        self.assertIn("gender", hidden.noun_paradigms[0]["features"].to_dict())
        hidden_forms = hidden.verb_paradigms[0]["forms"]
        overt_forms = overt.verb_paradigms[0]["forms"]
        self.assertEqual(
            hidden_forms["number=sg|person=3|gender=masc"],
            hidden_forms["number=sg|person=3|gender=fem"],
        )
        self.assertNotEqual(
            overt_forms["number=sg|person=3|gender=masc"],
            overt_forms["number=sg|person=3|gender=fem"],
        )

    def test_sample_string_respects_sampled_syllable_count(self):
        params = CFGParams(
            syllable_structure="CV",
            avg_syllables_per_word=1,
            syllable_max=3,
            verbs=1,
            nouns=1,
            propns=1,
            prons=1,
            adjs=1,
            det_def=1,
            det_indef=1,
            comps=1,
        )
        with (
            patch("numpy.random.uniform", return_value=0.8),
            patch("numpy.random.poisson", return_value=1),
            patch("numpy.random.beta", return_value=0.0),
            patch("numpy.random.binomial", return_value=0),
            patch.object(params, "_generate_syllable", return_value="ba"),
        ):
            self.assertEqual("baba", params._sample_string())


if __name__ == "__main__":
    unittest.main()
