import random
import unittest
from unittest.mock import patch

from scfg.agreement import FeatureBundle
from scfg.scfg import SCFG, CFGParams, SCFGParams


class SCFGGenerationTest(unittest.TestCase):
    def test_cfg_round_trip_preserves_agreement_metadata(self):
        params = CFGParams(
            agreement_enabled=True,
            latent_gender=True,
            realize_gender=True,
            gender_values=("fem", "masc"),
            verbs=2,
            nouns=2,
            propns=2,
            prons=2,
            adjs=1,
            det_def=1,
            det_indef=1,
            comps=1,
            rng_seed=13,
        )

        cloned = CFGParams.from_dict(params.to_dict())

        self.assertEqual(params.agreement_axes, cloned.agreement_axes)
        self.assertEqual(params.gender_values, cloned.gender_values)
        self.assertEqual(params.agreement_suffixes, cloned.agreement_suffixes)
        self.assertEqual(params.verb_paradigms, cloned.verb_paradigms)
        self.assertEqual(params.noun_paradigms, cloned.noun_paradigms)
        self.assertEqual(params.pronoun_paradigms, cloned.pronoun_paradigms)

    def test_sample_morpheme_falls_back_when_optional_template_is_empty(self):
        params = CFGParams(
            syllable_structure="C*",
            max_consonants=0,
            verbs=1,
            nouns=1,
            propns=1,
            prons=1,
            adjs=1,
            det_def=1,
            det_indef=1,
            comps=1,
            rng_seed=17,
        )

        with patch.object(
            params,
            "_generate_syllable",
            side_effect=["", "ba"],
        ) as mock_generate:
            self.assertEqual("ba", params._sample_morpheme())

        self.assertEqual(2, mock_generate.call_count)
        self.assertEqual(["C", "V"], mock_generate.call_args_list[1].args[0])

    def test_choose_verb_uses_first_gender_value_when_gender_is_unspecified(self):
        params = CFGParams(
            agreement_enabled=True,
            latent_gender=True,
            realize_gender=True,
            gender_values=("fem", "masc"),
            verbs=["lem"],
            nouns=1,
            propns=1,
            prons=2,
            adjs=1,
            det_def=1,
            det_indef=1,
            comps=1,
            rng_seed=19,
        )

        surface, features = params.choose_verb(
            random.Random(0),
            FeatureBundle(person="3", number="sg"),
        )

        expected_key = FeatureBundle(person="3", number="sg", gender="fem").key(
            params.latent_axes
        )
        self.assertEqual("fem", features.gender)
        self.assertEqual(params.verb_paradigms[0]["forms"][expected_key], surface)

    def test_possible_verb_right_surfaces_expand_when_source_lacks_agreement(self):
        params = SCFGParams(
            a=CFGParams(
                agreement_enabled=False,
                verbs=["stem"],
                nouns=1,
                propns=1,
                prons=1,
                adjs=1,
                det_def=1,
                det_indef=1,
                comps=1,
                rng_seed=23,
            ),
            b=CFGParams(
                agreement_enabled=True,
                verbs=["stem"],
                nouns=1,
                propns=1,
                prons=6,
                adjs=1,
                det_def=1,
                det_indef=1,
                comps=1,
                rng_seed=24,
            ),
        )
        scfg = SCFG(params)

        possible = scfg._possible_verb_right_surfaces(0, "stem")

        self.assertGreater(len(possible), 1)
        self.assertEqual(len(possible), len(set(possible)))

    def test_pro_drop_subject_sampling_preserves_null_surface_features(self):
        params = SCFGParams(
            a=CFGParams(
                agreement_enabled=True,
                pro_drop=True,
                verbs=1,
                nouns=1,
                propns=1,
                prons=1,
                adjs=1,
                det_def=1,
                det_indef=1,
                comps=1,
                rng_seed=29,
            ),
            b=CFGParams(
                agreement_enabled=True,
                pro_drop=True,
                verbs=1,
                nouns=1,
                propns=1,
                prons=1,
                adjs=1,
                det_def=1,
                det_indef=1,
                comps=1,
                rng_seed=30,
            ),
        )
        scfg = SCFG(params)

        for seed in range(200):
            subject = scfg._sample_agreement_recursive(
                "NP_SUBJ",
                rng=random.Random(seed),
                current_depth=0,
                min_depth=0,
                max_depth=0,
                role="subject",
            )
            if subject.trace == "pro":
                self.assertEqual("", subject.left_phonetic)
                self.assertEqual("", subject.right_phonetic)
                self.assertEqual(("∅",), subject.possible_right_full)
                self.assertEqual("3", subject.features.person)
                self.assertEqual("sg", subject.features.number)
                return

        self.fail("Did not encounter a pro-drop subject in 200 draws")

    def test_sample_respects_recursive_depth_bounds(self):
        scfg = SCFG(
            SCFGParams(
                a=CFGParams.english(),
                b=CFGParams.german(),
            )
        )

        sample = scfg.sample(min_depth=1, max_depth=1, rng=random.Random(31))

        self.assertEqual(1, sample["depth"])
        self.assertIn("that", sample["left_phonetic"])
        self.assertIn("dass", sample["right_phonetic"])


if __name__ == "__main__":
    unittest.main()
