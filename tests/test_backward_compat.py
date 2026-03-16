import unittest

from scfg.scfg import SCFG, CFGParams, SCFGParams


class BackwardCompatTest(unittest.TestCase):
    def test_old_style_cfg_dict_loads(self):
        old_dict = {
            "head_initial": True,
            "spec_initial": True,
            "pro_drop": False,
            "proper_with_det": False,
            "syllable_structure": "CVC",
            "avg_syllables_per_word": 2,
            "max_consonants": 2,
            "rng_seed": 1,
            "verbs": ["verb"],
            "nouns": ["noun"],
            "propns": ["name"],
            "prons": ["pron"],
            "adjs": ["adj"],
            "det_def": ["the"],
            "det_indef": ["a"],
            "comps": ["that"],
            "tenses": ["∅_T_pres"],
            "asps": ["∅_Asp_prog"],
            "orthography": "latin",
            "space_alpha": 0.5,
            "space_beta": 3.0,
            "syllable_max": 4,
        }
        params = CFGParams.from_dict(old_dict)
        self.assertFalse(params.agreement_enabled)
        self.assertEqual(["verb"], params.verb_lemmas)

    def test_agreement_disabled_grammar_still_samples(self):
        params = SCFGParams(a=CFGParams(rng_seed=3), b=CFGParams(rng_seed=4))
        sample = SCFG(params).sample(min_depth=0, max_depth=0)
        self.assertIn("left_phonetic", sample)
        self.assertEqual([sample["right_phonetic"]], sample["possible_right_phonetic"])
        self.assertNotIn("agreement_ok", sample)

    def test_legacy_yiddish_orthography_alias_still_samples(self):
        params = CFGParams(
            rng_seed=5,
            orthography="yiddish",
            verbs=2,
            nouns=2,
            propns=2,
            prons=2,
            adjs=1,
            det_def=1,
            det_indef=1,
            comps=1,
        )
        self.assertEqual("yiddish", params.orthography)
        self.assertTrue(
            any("\u0590" <= char <= "\u05ff" for char in params.verb_lemmas[0])
        )


if __name__ == "__main__":
    unittest.main()
