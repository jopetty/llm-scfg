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
        self.assertNotIn("agreement_ok", sample)


if __name__ == "__main__":
    unittest.main()
