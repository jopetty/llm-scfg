# EACL Camera-Ready Checklist — Submission 2101

Source: ACL ARR 2026 August, OpenReview forum `dzXIEG3M1Q`.
Reviewers: **NVoD** (3.5), **Mr9M** (4.0), **qmUX** (3.5). Meta-review (AC **MVd1**): 3.5, Borderline Conference.

**Scope rule used here:** this list is limited to (a) things we explicitly promised in an
author response, and (b) things the meta-review explicitly directs us to do. Items reviewers
raised that we did *not* promise are kept separate at the bottom, unticked, so we can decide
them deliberately rather than by accident.

The AC's framing matters for triage: *"the remaining issues appear to be addressable primarily
through improved presentation, clarification, and more careful discussion of the scope of the
conclusions."* Exactly one new experiment was promised.

---

## 1. Experiments to run

### 1.1 Examples-only control — **the only promised experiment** 🔴

Promised to **both** NVoD and qmUX:

> *"We will add an example-only null."* (to NVoD)
> *"We agree and will add an examples-only control."* (to qmUX)

NVoD escalated this in their post-rebuttal reply:

> *"in particular, the example-only null setting will contextualise the findings, and I believe
> this is an absolutely necessary control to include."*

**What it is:** a condition with few-shot example pairs **but no grammar in the prompt**. This is
the control that licenses the claim that performance is driven by grammar interpretation rather
than by the exemplars.

**Implementation notes:**
- `scfg/prompt.py:33` `basic_prompt()` hard-codes the grammar block into the prompt string. An
  examples-only variant needs a new prompt builder that drops the grammar section *and* the
  grammar-notation preamble (the long "A grammar is defined by a set of production rules…"
  paragraph), while keeping the task framing and the `Final answer:` output contract.
- Held-out exemplar pools already exist: `shots_*.jsonl` per grammar, surfaced as the `shots`
  split (`scfg/hf_data.py:85`, `scripts/build_hf_dataset.py:336`). No new data generation needed.
- Match the existing few-shot sweep (up to 16 shots) so the two curves are directly comparable on
  one axis; reuse the same 240 samples/condition.
- Cost: roughly the same order as the existing `fewshot` experiment. Well inside budget.

**⚠️ Vocabulary-coverage ceiling — read before running.** The examples-only condition is bounded
from above by a fact about the data, not by model ability. Computed exactly over all 42 fewshot
grammars (no API calls needed):

| k | **sentences fully covered** = exact-match ceiling | measured accuracy |
|---:|---:|---:|
| 1 | **0.00%** | 0.00% |
| 2 | **0.02%** | not yet run |
| 4 | **0.97%** | 0.18% |
| 8 | **8.81%** | not yet run |
| 16 | **25.95%** | 7.30% |

(Earlier drafts of this table quoted higher ceilings — 31.9% at k=16. That swept all 66 grammars
on disk; the ablation ran the 42 named in `fewshot_grammars.txt`, which is the correct
denominator. Figures above are the corrected ones.)

A test sentence can only be exact-matched if *every* one of its source words appears somewhere in
the k exemplars. At k=1 that is true for essentially no sentence; even at k=16 a perfect model
tops out near 26%. So an examples-only run is guaranteed to lose to the grammar condition for
reasons that have nothing to do with grammar interpretation.

This does **not** invalidate the control — it is still the comparison NVoD asked for, and the
honest result. But reporting a bare "examples-only scores ~X%" invites exactly the confound
objection the control was meant to answer. **Report the ceiling alongside the result**, and read
accuracy relative to it. The coverage numbers are cheap and exact; they belong in the paper
whatever we decide about further runs.

**Result (k=1, 4, 16 run; k=2, 8 pending).** Accuracy is **exactly 0.00% on every item whose
source vocabulary is not fully covered** — zero correct out of ~13,800 such items across all
three conditions. The ceiling is not a loose bound, it is binding. Conditional on full coverage,
examples-only reaches 18.4% (k=4, n=49) and 28.1% (k=16, n=1,308), against 47.06% for the grammar
with no examples at all. The grammar is doing the work, and it keeps doing it even on the subset
where the exemplars alone would suffice.

It also reframes NVoD's retrieval request. Retrieval by character overlap (as in Tanzer) raises
coverage by construction, which is presumably *why* they asked for it — a retrieved-exemplar
condition is the version of this control that is actually informative at low k. Worth
reconsidering rather than deferring.

**Caveat — we promised less than NVoD asked for.** NVoD asked for examples-only *retrieved by
character overlap* (as in Tanzer) from a trivially-generated corpus, plus optionally random
unretrieved examples. We promised only the plain null. Our `shots` pools are random, not
retrieved. Decide: ship the random-exemplar null (what we promised) and note the retrieval
variant as future work, or add retrieval. **Recommendation: ship what we promised**, and say in
one sentence that retrieval-based selection is untested here.

### 1.2 Nothing else was promised

We explicitly declined, in writing, to run:
- additional morphological/typological conditions (to NVoD: *"beyond the scope here"*) — replaced
  by a Grambank-framing edit, see §2.6;
- copying/segmenting subtask experiments (to Mr9M: *"new experiments on it would be out-of-scope"*)
  — replaced by a literature-engagement edit, see §2.9;
- a real-language-vocabulary upper-bound control (to NVoD: argued it is ill-posed — solvable from
  priors if the rules are natural, unsolvable if they differ).

Hold this line. None of the three reviewers pushed back after the rebuttal; two explicitly kept
their scores and one raised no further objection.

---

## 2. Edits promised in author responses

### 2.1 Experimental-detail table 🔴
Promised **verbatim and twice** (to NVoD and qmUX): *"We will add a table clarifying these
hyperparameters."* The AC singles this out as a required revision. Content we already committed to:

| Field | Value |
|---|---|
| Samples per condition | 240 (2 grammars × 120 sentences) |
| Runs per item | 1 |
| API sampling | provider default |
| Reasoning effort | provider default; all API models are reasoning models |
| Mean reasoning tokens | 0.8k–3.5k (not comparable across models) |
| Gemma 3 | vLLM defaults |

Also state: handling of truncated outputs, and that `gemma-3-1b-it` is excluded from larger
grammars for context-window reasons (already noted at `main.tex:512`).

### 2.2 Sentence-length cap clarification 🔴
To Mr9M: *"We will amend our writeup to better communicate this."* Two distinct things to fix:
1. The cap is **50 words**, not ~30 — the reviewer misread because
2. the plotted points are **quintile lengths, not maxima**. Say so explicitly in the figure caption
   and in §4.1.

Justification we gave, to be added: 50 words covers the bulk of attested sentence-length
distributions, plus the observation that length effects flatten beyond ~20 words. Citations
promised: Zhou et al. (2025), *Cognitive Science* 49; Sigurd et al. (2004), *Studia Linguistica* 58.

### 2.3 Grammar size vs. prompt length 🔴
To qmUX: *"We were unclear in our description."* Explain that the distractor manipulation qmUX
proposed **is already how grammars are built** — all grammars share core word-order rules, and
size differences come from lexicon entries that may not apply to any given sample. Add our
argument that padding with obvious filler would be unrealistic.

### 2.4 Few-shot claim scoping 🔴
To NVoD: *"We will also scope our claim to more narrowly highlight that while Aycock et al. find
no evidence that LLMs can use grammar explanations, our grammar-only results show otherwise
modulo the size limits."* The claim currently appears in four places — abstract, intro finding 4,
§4.4, Discussion ¶1. Decide how far to propagate (see §4.1 below).

### 2.5 Citation updates 🔴
To NVoD: *"We will update citations and add discussion of Pei et al."*

All titles verified against the publisher or arXiv record (2026-11-23). Add these five to
Paperpile and re-export:

| `\TODO` in `main.tex` | Full reference |
|---|---|
| l.245 Zhou et al. 2025 | Yikai Zhou, Jingyang Jiang & Haitao Liu, "Language Universals in Sentence Length: Comparing Sentence Length Distributions of 10 Languages," *Cognitive Science* 49(9), e70115, 2025. DOI `10.1111/cogs.70115` |
| l.245 Sigurd et al. 2004 | Bengt Sigurd, Mats Eeg-Olofsson & Joost van de Weijer, "Word length, sentence length and frequency: Zipf's law revisited," *Studia Linguistica* 58(1), 37–52, 2004 |
| l.255 Skirgård et al. 2023 | Hedvig Skirgård et al., "Grambank reveals the importance of genealogical constraints on linguistic diversity and highlights the impact of language loss," *Science Advances* 9(16), 2023. DOI `10.1126/sciadv.adg6175` |
| l.330 Jelassi et al. 2024 | Samy Jelassi, David Brandfonbrener, Sham Kakade & Eran Malach, "Repeat After Me: Transformers are Better than State Space Models at Copying," ICML 2024 (PMLR 235). arXiv:2402.01032 |
| l.334 Pei et al. 2026 | Renhao Pei, Yihong Liu, Sampo Pyysalo, Hinrich Schütze & Shaoxiong Ji, "Reasoning over Grammar: Can Synthetic Linguistic Reasoning Traces Enhance Low-Resource Machine Translation?" **EMNLP 2026 Findings**. arXiv:2606.03782 (v1 2 Jun 2026, v2 29 Aug 2026) |

Note that Pei et al. now has a venue — cite EMNLP 2026 Findings, not the bare preprint.

Version swaps, both verified:
- `tanzer-2023-benchmark` → "A Benchmark for Learning to Translate a New Language from One Grammar
  Book," **ICLR 2024** (arXiv:2309.16575). Confirmed.
- `aycock-2024-llms` → "Can LLMs Really Learn to Translate a Low-Resource Language from One Grammar
  Book?" by Seth Aycock, David Stap, Di Wu, Christof Monz & Khalil Sima'an, **ICLR 2025**
  (arXiv:2409.19151, posted Sept 2024). ⚠️ **The year changes 2024 → 2025**, so every in-text
  "Aycock et al. (2024)" becomes "(2025)" — there are several, including in the abstract-adjacent
  framing and §4.4. Decide whether to take the year change or keep citing the preprint.
- `papineni-2001-bleu` → year `2001` → `2002`. The booktitle already says ACL '02; only the `year`
  field is wrong.

⚠️ **`paper/paperpile.bib` is Paperpile-generated.** Hand edits will be clobbered on the next
sync. Make these changes in Paperpile, then re-export.

### 2.5a Our Pei et al. sentence mischaracterized the paper — ✅ **fixed**

`main.tex:334` currently reads: *"Recent work moves beyond pure in-context learning by, for
instance, generating reasoning traces from treebanks in place of grammar books."*

That is backwards. Pei et al. test their synthetic traces three ways — in-context learning,
supervised fine-tuning, and reinforcement fine-tuning — and find the traces help **most as
inference-time guidance in in-context learning**, with the training-data uses giving smaller and
less consistent gains. Their contribution is a richer in-context signal, not a move away from ICL.
Our follow-on sentence ("Such approaches plausibly outperform pure ICL, at a higher cost") rests
on the same misreading.

**Done.** The paragraph now presents Pei et al. as an instance of two of the three levers the
preceding paragraph names — scaffolding that decomposes a derivation into steps, and targeted
training — and reports that they found the two unequal, with the scaffolding working better. That
is both accurate and a stronger connection than the original "moves beyond ICL" framing, since it
makes their result evidence for our own recommendation rather than a contrast with it. Their
bottleneck (generating reliable analyses at all) is presented as the complement of ours, and the
cost asymmetry is kept. Costs about 15 words more than the incorrect version.

All four `\TODO{}` citation placeholders are now gone from `main.tex`; the `\TODO` macro itself is
unused and could be dropped from the preamble.

### 2.6 Grambank contextualization 🟡
To NVoD: *"We will contextualize our findings based on Grambank's paradigms and frequencies, and
note that while running additional morphological experiments is beyond the scope here our
generator for synthetic languages makes it easy for future work to do so."* Prose + citation only.

### 2.7 Implications / "what is the reader meant to do with this" 🟡
NVoD's headline weakness; the AC echoes it. We answered in the rebuttal and should move that
answer into the Discussion: models can translate from formal specifications, but accuracy degrades
*before* the grammar sizes natural languages require; since our grammars fit comfortably in
context, gains will come from better in-context use or targeted training, **not** from longer
context windows.

### 2.8 SCFG formulation note 🟡
To qmUX: *"will add a note clarifying that our more restrictive formulation precludes non-shared
category inventories."* Attaches to the $A = B$ simplifying assumption (`main.tex:176`).

### 2.9 Copying / segmenting literature 🟡
To Mr9M: *"will amend our paper to interact more directly with this line of work."* Cite Jelassi
et al. (2024) and McCoy et al. (2024) — `mccoy-2024-embers` is already in the bib and already cited
at `main.tex:308`; Jelassi is not.

### 2.10 Data and code release 🔴
Both NVoD and Mr9M scored us **Datasets 1/5, Software 1/5** — "no usable datasets/software
submitted" — purely because they didn't find them. qmUX, who did, gave 3/5 on both. This is free
score recovery. Make the release unmissable:
- flip `\usepackage[review]{acl}` → `[final]` (`main.tex:5`), which switches
  `\resourceavailability` from the anonymous repo to `jpetty.org/ict` (`includes/preamble.tex:118`);
- state the HuggingFace dataset explicitly, not just the code link at l.141.

### 2.11 Typos promised to qmUX 🔴
*"These will all be corrected."* All located:

| Reviewer ref | `main.tex` | Fix |
|---|---|---|
| L108 | `:77` | `including including` → `including` |
| L361 | `:234` | `a subset of of the most recently-recent models` → `a subset of the most recent models` |
| L589–591 | `:308` | `the target script that differs from` → `the target script differs from` |
| L896 | `:407` | `tense prhases` → `tense phrases` |
| L927 | `:457` | `assmbled` → `assembled` |
| L964 | `:490` | `making it is more sensitive` → `making it more sensitive` |
| L994–995 | `:500` | `\Cref{…} shows` → `show` |
| App. D.4 | `:560` | `word order experiment` → `morphology experiment` (wrong experiment named in Morphology section) |
| — | bib | BLEU year 2001 → 2002 |
| — | `:247` + figures | `input length` vs `sentence length` — pick one |

Note: the `sentence length`/`input length` split also appears in **figure axis labels**
(Figure 3 vs Figures 8–13), so fixing it fully means regenerating figures, not just editing prose.

---

## 3. Explicitly declined — do not reopen

- **Abstract phrasing.** qmUX asked us to change "the number of vocabulary words" to "grammar size
  in rules". We replied: *"given the above clarification, we believe the current phrasing is
  accurate."* Reviewer did not press. Leave it.

---

## 4. Raised but never promised — decide deliberately

These are real review comments with no author commitment behind them. Listed so the choice is
conscious.

1. **Significance testing / effect sizes.** ✅ **Done** — `scripts/significance_tests.py`,
   written into §4.2 and new Appendix `sec:significance`. Word order is confirmed null with a
   test matched to the design: randomization test *p* = 0.55, randomized-block ANOVA
   *F*(2,33) = 0.71, *p* = 0.50, largest pairwise difference 3.5 pp, equivalence established at
   ±9.1 pp. **The clustering correction mattered**: treating the 5,040 sentences as independent
   returns *p* = 0.06, a near-threshold value that would have invited exactly the over-reading
   the test was meant to prevent. See §7 below for two claims the same tests call into question.
2. **Quantifying the "script tax."** ✅ **Done** — tested, and the result rewrote §4.3 rather
   than decorating it; the full pairwise table is in Appendix `tab:significance-other`. See §7.1.
3. **Tokenization fertility.** We gave numbers in the rebuttal (our vocabularies: fertility 2–4;
   English: ~1.3) but never promised to put them in the paper. `scripts/tokenization_summary.py`
   already exists. One sentence in Limitations. **Recommendation: include** — it is a direct
   answer to a reasonable objection and costs nothing.
4. **Surfacing chrF++/BLEU from the appendix.** We pointed NVoD at Appendix D rather than
   promising a change. NVoD separately asked that more results be pulled out of the appendix
   because "most results presented with single model … clouds findings". Low cost, real benefit.
5. **Upper-bound framing throughout.** NVoD asked that the upper-bound reading be "clearly
   emphasised throughout the paper", not just in Limitations. The AC's single biggest revision ask
   is adjacent: a clearer discussion that performance degrades *before* natural-language
   complexity. The Limitations section already ends on this note (`main.tex:316`) — the work is
   promoting it to the Discussion and the intro. **Recommendation: do it; this is what the AC
   actually asked for.**

---

## 5. ⚠️ Page budget

With the drafted additions in place, the paper builds cleanly (37 pages, no undefined references)
but **content now runs into page 10**: Discussion ends on p.9 and the Limitations section starts
on p.10. Identical in `[review]` and `[final]` mode, so this is not a line-numbering artifact.

EACL allows 9 content pages for a camera-ready long paper; Limitations and References are excluded
from the count. That puts us marginally over. The drafted text is deliberately generous — written
to be cut — so the trim should come from choosing among the drafts rather than from compressing
the existing paper. The two longest additions are the implications passage (§2.7) and the
scope-of-conclusions paragraph in Limitations (§4.5); the latter is free, since Limitations does
not count. **Cut from the Discussion, not from Limitations.**

## 6. Suggested order of work

1. Typos, citations, release statement, `[review]`→`[final]` — mechanical, finishes §2.5/2.10/2.11.
2. Launch the examples-only run (§1.1) — longest latency, start it first.
3. Hyperparameter table (§2.1) and the three clarifications (§2.2, §2.3, §2.8).
4. Discussion/Limitations rewrite: implications (§2.7), scope and upper bound (§4.5), Grambank
   (§2.6), Pei et al. (§2.5), copying literature (§2.9).
5. Fold in examples-only results; scope the few-shot claim (§2.4) to match what the data shows.
6. Significance tests (§4.1) and fertility note (§4.3) if time allows.

---

## 7. ⚠️ What the significance tests revealed — two claims to revisit

`uv run python scripts/significance_tests.py` runs all three condition experiments. Word order
came out exactly as the paper says. The other two did not, and both involve **ordering** claims
rather than the headline effects, which hold up strongly.

### 7.1 Orthography (§4.3): the graded hierarchy is not supported — ✅ **revised**

The paper says models "perform best at translating into Latin script, moderately worse at
translating into Cyrillic, and worst at translating into Hebrew script," and that unpointed
Hebrew sits "below that of Cyrillic and Latin scripts."

Block-adjusted, with grammar as the unit of analysis (gpt-5.6-luna, 70 grammars):

| condition | accuracy |
|---|---:|
| Latin | 58.51% |
| Cyrillic | 53.63% |
| Hebrew (unpointed) | 53.51% |
| Latin + diacritics | 51.43% |
| **Hebrew (pointed)** | **0.12%** |

- **Pointed Hebrew is unambiguous**: −58.4 pp vs Latin, *p* < 0.0001. This finding is safe.
- **Cyrillic vs unpointed Hebrew: +0.12 pp, *p* = 0.99.** These are indistinguishable, so
  "moderately worse ... and worst" misdescribes them.
- **Latin vs Cyrillic: −4.88 pp, *p* = 0.65**; Latin vs Hebrew: −5.00 pp, *p* = 0.65. Neither is
  significant.
- **Latin + diacritics is numerically the lowest of the four non-collapsed conditions**, which
  cuts against "best at translating into Latin script."

The interaction is significant here (*F* = 23.8), so the conservative error term applies and the
intervals are wide (±20 pp). The honest reading: the script effect is carried **entirely by vowel
pointing**, and the remaining differences are not resolvable at this sample size. The subsection
title "Less familiar scripts make translation harder" is defensible only for pointing.

**Done.** The subsection is retitled "Vowel pointing makes translation collapse", the graded
ordering is replaced by the four-way null, and the observation that Latin-with-diacritics is the
lowest of the four — despite not being a script change at all — is used to point at diacritic
sequences rather than script novelty. Also propagated to intro finding 3 and the Discussion's
"performance falls when the target script differs" sentence, which was the same claim.

### 7.2 Agreement (§4.2): "neither language has agreement is easiest" is not supported — ✅ **revised**

The paper says "language pairs where neither language has overt agreement morphology on verbs are
the easiest." Block-adjusted:

| condition | accuracy | vs NoAgr → NoAgr |
|---|---:|---|
| Agr → Agr | 85.42% | +1.67 pp, *p* = 0.85 |
| Agr → NoAgr | 83.47% | +3.33 pp, *p* = 0.66 |
| NoAgr → NoAgr | 80.14% | — |
| **NoAgr → Agr** | **44.79%** | **−35.4 pp, *p* < 0.0001** |

NoAgr → NoAgr is in fact the *lowest* of the three non-collapsed conditions, though none of those
three differ significantly. The real finding — that translating *into* a language with agreement
from one without it is dramatically harder — is strongly supported. Note this experiment is
unbalanced (Agr → Agr is missing one size block), so raw condition means are not comparable
across conditions and the block-adjusted figures above should be used.

**Done.** "Are the easiest" is gone; the paragraph now reports the NoAgr → Agr deficit as
block-adjusted (35.4 points, *p* < 0.0001) and states that the other three conditions are
indistinguishable.

### 7.3 Note on method

Intervals use the randomized-block ANOVA error term, pooling the interaction when it is not
significant. Two earlier approaches were wrong and are recorded in the script's docstring: a
cluster bootstrap within cells (two grammars per cell badly underestimates variance — it produced
intervals excluding zero for differences the randomization test put at *p* = 0.35), and
block-level differences only (valid but discards replication, widening the word-order bound from
±6.8 to ±12.7 pp for no reason).
