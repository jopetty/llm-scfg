# Large Word Order Experiment Data

This dataset expands the word-order experiment with more grammar sizes and more grammar replicates per condition.

## Experimental Matrix

- source word order: `head_initial=True`, `spec_initial=True`
- target word orders: `head_initial=True, spec_initial=True`, `head_initial=False, spec_initial=True`, `head_initial=False, spec_initial=False`
- grammar sizes: [25, 50, 100, 1000, 5000, 7500, 10000]
- grammars per size and word-order condition: 2
- depth range: `0..5`
- samples per depth: 20
- total grammars: 42
- total samples: 5040

## Regeneration

```bash
uv run python main.py exp_wordorder_large
```

## Condition Examples

Examples use the `left_phonetic` and `right_phonetic` surface forms from the sample JSONL files.

### Target matches the source order

- condition: target head_initial=True, target spec_initial=True, lexical size target=25
- grammar: `556e6de808525658`
- example left: `nawebfok aykvuzbpud gnaw fduliz spezuyovkfey`
- example right: `bvinkdapbnay kguxis dzox sgowssej it sbez`

### Target is head-final, spec-initial

- condition: target head_initial=False, target spec_initial=True, lexical size target=25
- grammar: `63f6848845de826d`
- example left: `teqkofdguznef gpijub kabbih btet`
- example right: `efviv urdettel dvay gig sduhkkip`

### Target is head-final, spec-final

- condition: target head_initial=False, target spec_initial=False, lexical size target=25
- grammar: `ba6060385a12891d`
- example left: `gmoqsir kbuptsiksem utestnaz iluzdvoqvkuz ogvap`
- example right: `kgoc gug ev kzulmlifeq kbajbkog iy`
