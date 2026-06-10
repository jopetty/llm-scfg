# Few-Shot Word Order Experiment Data

This dataset varies the number of in-context examples for Latin-script source and target grammars while using the word-order conditions from the main word-order experiment. Each grammar has a dedicated `shots_*.jsonl` pool held out from the evaluated `samples_*.jsonl` rows.

## Experimental Matrix

- source word order: `head_initial=True`, `spec_initial=True`
- target word orders: `head_initial=True, spec_initial=True`, `head_initial=False, spec_initial=True`, `head_initial=False, spec_initial=False`
- source orthography: `latin`
- target orthography: `latin`
- few-shot k values: [0, 1, 2, 4, 8, 16]
- grammar sizes: [25, 50, 100, 1000]
- grammars per size and word-order condition: 2
- depth range: `0..5`
- evaluation samples per depth: 20
- shot-pool samples per depth: 3
- total grammars: 24
- total evaluation samples: 2880
- total shot-pool samples: 432

## Regeneration

```bash
uv run python main.py exp_fewshot
```

## Condition Examples

Examples use the `left_phonetic` and `right_phonetic` surface forms from the sample JSONL files.

### Target matches the source order

- condition: target head_initial=True, target spec_initial=True, lexical size target=25
- grammar: `32189df5a34dd8a5`
- example left: `kecbig ton im dpaf aftnazkavkeb`
- example right: `odgbigavkor od sbawkab oy dawppojgek`

### Target is head-final, spec-initial

- condition: target head_initial=False, target spec_initial=True, lexical size target=25
- grammar: `17570ddaf3352608`
- example left: `svetan ggam pijtagkifpdec paz`
- example right: `ar gag pigul ujtojnov`

### Target is head-final, spec-final

- condition: target head_initial=False, target spec_initial=False, lexical size target=25
- grammar: `b748e0626644d226`
- example left: `duz sbix zpedpfex fob duz sbix fgiwafvzaq`
- example right: `ttuxedbfay fas ngurkex em ttuxedbfay ag`
