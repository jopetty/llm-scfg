# Agreement Roadmap

## Summary

Implement a reusable agreement core with subject-verb agreement as the first enabled behavior. The implementation should preserve backward compatibility for existing orthography and word-order experiments when agreement is disabled, and it should expose enough structure to extend later to determiner-noun and adjective-noun agreement.

## Key Changes

- Add feature and unification utilities in `scfg/agreement.py`.
  - `FeatureBundle` should carry `person`, `number`, and reserved `gender`.
  - `AgreementConfig` should capture the enablement and active axes.
  - `FeatureUnifier` should provide deterministic compatibility checks.
- Extend `CFGParams` in `scfg/scfg.py`.
  - Add agreement flags and synthetic inflection controls.
  - Generate pronoun, noun, and verb paradigms when agreement is enabled.
  - Persist agreement suffixes and paradigms into the grammar JSON.
- Extend `SCFGParams.to_dict()`.
  - Add `agreement_metadata` to serialized grammars.
  - Keep `grammar_str` readable and append a short agreement summary.
- Refactor sampling in `SCFG.sample()`.
  - Preserve the legacy sampler for grammars without agreement.
  - Use a feature-aware derivation path for grammars with agreement.
  - Carry subject features into verbal realization.
  - Save `subject_features`, `verb_features`, `agreement_ok`, and `agreement_trace` in each sample.
- Extend the CLI in `main.py`.
  - Add `create_agreement_data()`.
  - Allow `create_grammar()` to enable agreement on either language side.
  - Pass agreement metadata through prompt generation.
- Update prompting in `scfg/prompt.py`.
  - Explain that some lexical items inflect for person and number.

## Test Plan

Use `unittest` and keep tests focused on behavior, not notebook outputs.

- `tests/test_agreement_core.py`
  - `FeatureBundle` serialization round-trip.
  - `FeatureUnifier` accepts compatible bundles and rejects incompatible ones.
  - `CFGParams` with agreement enabled builds six pronoun cells and inflected verb/noun paradigms.
- `tests/test_agreement_sampling.py`
  - Samples from agreement grammars always align subject and verb features.
  - Proper nouns behave as third-person singular in v1.
  - Agreement-enabled grammars save debug metadata fields.
- `tests/test_agreement_cli.py`
  - `create_grammar()` writes `agreement_metadata`.
  - `generate_samples()` writes sample JSONL entries with agreement fields.
- `tests/test_backward_compat.py`
  - Loading an old-style grammar dict without agreement metadata still works.
  - Agreement-disabled grammars still sample successfully.

## Assumptions

- V1 agreement is subject-verb agreement only.
- Active v1 axes are `person` and `number`.
- Inflection is synthetic and generated internally rather than delegated to an external morphology package.
- Existing data files do not need migration; backward compatibility is handled in the loader path.
