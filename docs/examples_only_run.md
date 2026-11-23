# Examples-only control — manual submission guide

The grammar ablation promised to Reviewers NVoD and qmUX. Five batch files are generated and
ready to upload at <https://platform.openai.com/batches>; programmatic submission is currently
blocked by `billing_hard_limit_reached` on the account.

## The files

All in `batches/fewshot_exp/`, model `gpt-5.6-luna`, endpoint `/v1/chat/completions`,
completion window 24h. One file per `k`, so each is a complete, independently analyzable
condition — a partial upload still yields usable cells.

| k | file | requests | size | input tokens |
|---:|---|---:|---:|---:|
| 1 | `inputs_fewshot_examples_only_k1_gpt-5.6-luna_part1_of_1_943f7a.jsonl` | 5,040 | 10.6 MB | 1,501,910 |
| 2 | `inputs_fewshot_examples_only_k2_gpt-5.6-luna_part1_of_1_9ca5a0.jsonl` | 5,040 | 11.1 MB | 1,678,910 |
| 4 | `inputs_fewshot_examples_only_k4_gpt-5.6-luna_part1_of_1_b95640.jsonl` | 5,040 | 12.3 MB | 2,118,830 |
| 8 | `inputs_fewshot_examples_only_k8_gpt-5.6-luna_part1_of_1_bc4ed3.jsonl` | 5,040 | 15.4 MB | 3,323,870 |
| 16 | `inputs_fewshot_examples_only_k16_gpt-5.6-luna_part1_of_1_853f68.jsonl` | 5,040 | 25.3 MB | 7,105,070 |
| | **total** | **25,200** | **74.7 MB** | **15,728,590** |

Every file is well inside OpenAI's per-file limits (50,000 requests, 200 MB). Output is the cost
driver, not input: at the 1,635 completion-tokens/request observed in the existing few-shot run,
expect roughly 41M output tokens across all five.

There is no `k=0` file by design — with no grammar *and* no examples the prompt carries no
information about the target language. The `k=0` cell of the existing few-shot run already serves
as the grammar-only baseline.

## Suggested submission order

**k=16 and k=8 first.** Per the coverage ceiling in
[`camera_ready_checklist.md`](camera_ready_checklist.md), these are the only two cells where the
control can produce a non-trivial number: a perfect model tops out at 31.9% at k=16 and 11.1% at
k=8, against 1.2% at k=4 and ~0% at k=1 and k=2. If budget only covers part of the sweep, those
two carry essentially all the information. The low-k cells are still worth having — they make the
monotonic trend visible — but they are close to foregone conclusions.

## Pulling results back

Dashboard-submitted jobs leave no local manifest, so use the id-based collector. Find the ids:

```bash
uv run python openai_batch.py list_batches --limit=10
```

Then download, newest ids first:

```bash
uv run python openai_batch.py collect_ids --batch_ids='["batch_xxx","batch_yyy"]' --out_dir=batches/fewshot_exp
```

This writes `batch_<id>_output.jsonl` into `batches/fewshot_exp/`, which is the filename the
analysis notebooks already glob for — so a dashboard-submitted run is indistinguishable
downstream from one submitted with `openai_batch.py submit`. Any failed requests are written
alongside as `batch_<id>_output_errors.jsonl`.

If you download the `.jsonl` straight from the dashboard instead, rename it to
`batch_<id>_output.jsonl` and drop it in the same directory; nothing else is needed.

## If billing gets unblocked

The whole thing becomes one command per file:

```bash
uv run python openai_batch.py submit_dir --batch_dir=batches/fewshot_exp --input_glob='inputs_fewshot_examples_only_k*.jsonl'
```

then `status`, and `wait` to block until done and collect automatically.

## Reporting the result

Do not report bare examples-only accuracy. Report it against the coverage ceiling for each k —
the control is bounded from above by vocabulary coverage, not by model ability, and a reviewer
who notices that before we say it will read the comparison as confounded.
