# LM_Dec_Long_Feedback

Feedback - Long family, decoder-only byte-level language model, standardised in the Statera Working Group Language Model library.

## Status

- Verification level: **Smoke-Tested**
- Failure class: Hardware Infeasible: token-by-token recurrence; 3M of 200M tokens in 4 h on an RTX 3070 Ti

## Provenance

- Source repository: LM_2020_feedback-transformer
- Upstream / license: lucidrains/feedback-transformer-pytorch (MIT)

## Differences from the source ([Unified] protocol)

- seq_len (step) 2, mem_len 256 (upstream defaults)
- trained in micro-batches of 2 with gradient accumulation (same update; the token-by-token recurrence keeps a long graph and does not fit batch 16 x 512 in 8 GB)

All models share one protocol: 256-byte vocabulary, dim 512, 6 layers, 8 heads x 64, Adam 2e-4 with 200 warm-up steps, gradient clip 0.5, bf16 autocast (automatic fp32 fallback). Numbers rank structures under one budget on one GPU (RTX 3070 Ti 8 GB); they do not reproduce paper results.

## Results

- Parameters: 22.85 M
- Causality check: pass (largest change of earlier outputs 0.0e+00)
- Overfit check (4 x 256 bytes): 5.728 -> 0.085 nats, pass
- LM-02 enwik8 (200M tokens): valid 2.506 / test 2.482 bits per byte (time-capped)
- Peak GPU memory at length 2048 (batch 1): train 3476 MB, inference 60 MB
- Peak GPU memory at length 8192 (batch 1): train 14384 MB, inference 105 MB

## Tests

- check: data counts, causality (no look-ahead, outputs use the current byte), parameters
- scaling: peak GPU memory and time vs length 512-8192
- overfit: memorise 4 x 256 bytes below 0.5 nats
- LM-02: enwik8 next-byte prediction (bits per byte)
- LM-05: long-horizon key-value recall with key-removed control (v3: distinct queries)
- LM-06: memory update - latest value vs stale value after a key is reassigned

## Run

```bash
pip install torch==2.5.1 einops numpy
python run.py check feedback
python run.py lm02 feedback      # enwik8 is downloaded to data/ on first use
```
