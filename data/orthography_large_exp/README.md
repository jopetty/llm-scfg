# Large Orthography Experiment Data

This dataset expands the orthography experiment with a larger grammar-size grid and two additional target-side writing systems.

## Experimental Matrix

- source orthography: `latin`
- target orthographies: `latin`, `latin_diacritic`, `cyrillic`, `hebrew`, `hebrew_unpointed`
- grammar sizes: [25, 50, 100, 1000, 5000, 7500, 10000]
- grammars per size and orthography: 2
- depth range: `0..5`
- samples per depth: 20
- total grammars: 70
- total samples: 8400

## Regeneration

```bash
uv run python main.py exp_orthography_large
```

## Condition Examples

Examples use the `left_phonetic` and `right_phonetic` surface forms from the sample JSONL files.

### Latin

- condition: target orthography=`latin`, lexical size target=25
- grammar: `e8013ae6f60f9a5d`
- example left: `kbot efig esdbedof kbot uc`
- example right: `pumstofdalal ofkugzas fir pinffiq pumstofdalal vgufzobpox`

### Latin with diacritics

- condition: target orthography=`latin_diacritic`, lexical size target=25
- grammar: `c0475e3f1baf49c6`
- example left: `kifew undas guqejug biqelbvuhis`
- example right: `žfüř çtükékêsâk öj ódšfínřdösçžäv`

### Cyrillic

- condition: target orthography=`cyrillic`, lexical size target=25
- grammar: `e54f134cb83062cf`
- example left: `ur mviv vtixbpoh ud`
- example right: `фыччыпес ар узюн щрусвгюйжнёв нядаб`

### Hebrew

- condition: target orthography=`hebrew`, lexical size target=25
- grammar: `0165173155009949`
- example left: `bbur sbehkegew ib bbur kic`
- example right: `ילּפללײז ײּ ייִד רראג ילּפללײז ײּ בריתַח`

### Hebrew without nikkud

- condition: target orthography=`hebrew_unpointed`, lexical size target=25
- grammar: `c192b693deee4f1d`
- example left: `zzesfkaw ij tihpekkeggom pgujsviq`
- example right: `בישוס אפ נאריגנחומ פוגוליח`
