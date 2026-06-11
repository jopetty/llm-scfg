import random
import unittest

from scfg.scfg import SCFG, CFGParams, SCFGParams


def _cfg(agreement_enabled: bool, rng_seed: int) -> CFGParams:
    return CFGParams(
        agreement_enabled=agreement_enabled,
        rng_seed=rng_seed,
        verbs=20,
        nouns=20,
        propns=10,
        prons=6,
        adjs=10,
        det_def=2,
        det_indef=2,
        comps=2,
    )


def make_scfg(agreement_a: bool, agreement_b: bool) -> SCFG:
    return SCFG(
        SCFGParams(
            a=_cfg(agreement_a, rng_seed=7),
            b=_cfg(agreement_b, rng_seed=8),
        )
    )


class AgreementPossibleRightTest(unittest.TestCase):
    def test_full_and_phonetic_answers_are_aligned(self):
        scfg = make_scfg(False, True)
        rng = random.Random(0)
        for depth in range(4):
            for _ in range(50):
                sample = scfg.sample(min_depth=depth, max_depth=depth, rng=rng)
                self.assertEqual(
                    len(sample["possible_right"]),
                    len(sample["possible_right_phonetic"]),
                )

    def test_gold_answer_is_always_acceptable(self):
        scfg = make_scfg(False, True)
        rng = random.Random(1)
        for depth in range(6):
            for _ in range(100):
                sample = scfg.sample(min_depth=depth, max_depth=depth, rng=rng)
                self.assertIn(
                    sample["right_phonetic"], sample["possible_right_phonetic"]
                )

    def test_unmarked_source_answer_set_stays_small(self):
        # An unmarked source -> marked target is the configuration that used to
        # explode (millions of answers per sentence) because every word's
        # realization was treated independently. Tying the subject to its verb
        # keeps the acceptable set proportional to the number of independent
        # agreement domains, not their product.
        scfg = make_scfg(False, True)
        rng = random.Random(2)
        worst = 0
        for depth in range(6):
            for _ in range(200):
                sample = scfg.sample(min_depth=depth, max_depth=depth, rng=rng)
                worst = max(worst, len(sample["possible_right_phonetic"]))
        self.assertLessEqual(worst, 256)

    def test_subject_number_ambiguity_is_preserved(self):
        # The unmarked source genuinely underdetermines number, so a model may
        # answer with either reading: the set must sometimes contain more than
        # the single gold answer.
        scfg = make_scfg(False, True)
        rng = random.Random(3)
        self.assertTrue(
            any(
                len(
                    scfg.sample(min_depth=0, max_depth=0, rng=rng)[
                        "possible_right_phonetic"
                    ]
                )
                > 1
                for _ in range(200)
            )
        )

    def test_verb_is_tied_to_subject_not_independent(self):
        # At depth 0 a clause has at most two independent agreement domains (the
        # subject and the object), so the consistent set has at most four
        # answers. If the verb varied independently of its subject the count
        # could reach 2 (subject) x 6 (verb person/number) x 2 (object) = 24.
        scfg = make_scfg(False, True)
        rng = random.Random(4)
        sizes = [
            len(
                scfg.sample(min_depth=0, max_depth=0, rng=rng)[
                    "possible_right_phonetic"
                ]
            )
            for _ in range(300)
        ]
        self.assertLessEqual(max(sizes), 4)
        self.assertIn(4, sizes)

    def test_matched_agreement_has_single_answer(self):
        # When both sides mark agreement (or neither does) the source is
        # unambiguous, so there is exactly one acceptable target.
        for agreement_a, agreement_b in ((True, True), (False, False)):
            scfg = make_scfg(agreement_a, agreement_b)
            rng = random.Random(5)
            for depth in range(4):
                for _ in range(50):
                    sample = scfg.sample(min_depth=depth, max_depth=depth, rng=rng)
                    self.assertEqual(len(sample["possible_right_phonetic"]), 1)


if __name__ == "__main__":
    unittest.main()
