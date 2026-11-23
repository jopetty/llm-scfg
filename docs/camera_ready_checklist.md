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

| k | mean source-word coverage | **sentences fully covered** = exact-match ceiling |
|---:|---:|---:|
| 1 | 15.0% | **0.0%** |
| 2 | — | **0.1%** |
| 4 | 40.4% | **1.2%** |
| 8 | — | **11.1%** |
| 16 | 72.0% | **31.9%** |

A test sentence can only be exact-matched if *every* one of its source words appears somewhere in
the k exemplars. At k=1 that is true for essentially no sentence; even at k=16 a perfect model
tops out near 32%. So an examples-only run is guaranteed to lose to the grammar condition for
reasons that have nothing to do with grammar interpretation.

This does **not** invalidate the control — it is still the comparison NVoD asked for, and the
honest result. But reporting a bare "examples-only scores ~X%" invites exactly the confound
objection the control was meant to answer. **Report the ceiling alongside the result**, and read
accuracy relative to it. The coverage numbers are cheap and exact; they belong in the paper
whatever we decide about further runs.

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
- `tanzer-2023-benchmark` → published version (ICLR 2024). Currently `@MISC`/arXiv.
- `aycock-2024-llms` → published version. Currently `@MISC`/arXiv. **Verify the venue** before
  changing.
- `papineni-2001-bleu` → year `2001` → `2002`. The booktitle already says ACL '02; only the `year`
  field is wrong.
- Add **Pei et al. (2026), arXiv:2606.03782** — reasoning traces from UD treebanks in place of
  grammar books — with the discussion we promised: their approach plausibly beats pure ICL at
  higher cost, since our exemplars are cheap but per-sample traces are not.

⚠️ **`paper/paperpile.bib` is Paperpile-generated.** Hand edits will be clobbered on the next
sync. Make these four changes in Paperpile, then re-export.

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

1. **Significance testing / effect sizes.** Raised independently by NVoD ("significance testing is
   missing") and qmUX ("would strengthen the word-order null in particular"). We never answered
   this. It is the strongest unanswered methodological point, and it bears on a **null result we
   rely on** — the word-order finding. Cheap to add from data we already have.
   **Recommendation: do it.**
2. **Quantifying the "script tax."** NVoD asked for the statistical effect of changing script. Not
   answered. Same analysis machinery as (1). Moderate value.
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
