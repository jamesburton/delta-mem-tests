# Smoke test — delta-mem chat

- Scan implementation: `torch` (locked via `DELTA_MEM_SCAN_IMPL=torch`; Triton unavailable — see `report/kernels-gate.md`)
- Memory tensors that changed across the 3-turn chat: **36**
- Final-turn recall of 'teal': **PASS**
- State-change gate: **PASS**
- Peak VRAM: **7.54 GB**

## Transcript

**USER:** My favourite colour is teal. Remember that for later.

**ASSISTANT:** Got it! Teal is your favorite color — I'll keep that in mind for future conversations. 🌿💙

**USER:** Quick aside: what is 7 times 8?

**ASSISTANT:** 7 times 8 is 56. 😊

**USER:** What did I tell you my favourite colour was?

**ASSISTANT:** You told me your favorite color is **teal**! 🌿💙 I remember — it's a beautiful, calming shade that blends the cool of blue with the freshness of green. Great choice! 😊

## Memory-state changes (per-module signature sums)

- `model.layers.0.self_attn.delta_state`: 0.0000 → 0.3711
- `model.layers.1.self_attn.delta_state`: 0.0000 → 0.4121
- `model.layers.2.self_attn.delta_state`: 0.0000 → 0.6289
- `model.layers.3.self_attn.delta_state`: 0.0000 → 1.1562
- `model.layers.4.self_attn.delta_state`: 0.0000 → 0.8320
- `model.layers.5.self_attn.delta_state`: 0.0000 → 0.8164
- `model.layers.6.self_attn.delta_state`: 0.0000 → 1.4766
- `model.layers.7.self_attn.delta_state`: 0.0000 → 2.9375
- `model.layers.8.self_attn.delta_state`: 0.0000 → 1.9453
- `model.layers.9.self_attn.delta_state`: 0.0000 → 2.8750
- `model.layers.10.self_attn.delta_state`: 0.0000 → 3.7969
- `model.layers.11.self_attn.delta_state`: 0.0000 → 2.9531
- `model.layers.12.self_attn.delta_state`: 0.0000 → 2.6406
- `model.layers.13.self_attn.delta_state`: 0.0000 → 2.7344
- `model.layers.14.self_attn.delta_state`: 0.0000 → 3.0938
- `model.layers.15.self_attn.delta_state`: 0.0000 → 2.8906
- `model.layers.16.self_attn.delta_state`: 0.0000 → 3.0938
- `model.layers.17.self_attn.delta_state`: 0.0000 → 2.8750
- `model.layers.18.self_attn.delta_state`: 0.0000 → 3.8594
- `model.layers.19.self_attn.delta_state`: 0.0000 → 3.8281
- `model.layers.20.self_attn.delta_state`: 0.0000 → 4.6875
- `model.layers.21.self_attn.delta_state`: 0.0000 → 5.7500
- `model.layers.22.self_attn.delta_state`: 0.0000 → 7.7812
- `model.layers.23.self_attn.delta_state`: 0.0000 → 9.7500
- `model.layers.24.self_attn.delta_state`: 0.0000 → 4.0312
- `model.layers.25.self_attn.delta_state`: 0.0000 → 7.9062
- `model.layers.26.self_attn.delta_state`: 0.0000 → 11.6875
- `model.layers.27.self_attn.delta_state`: 0.0000 → 11.5625
- `model.layers.28.self_attn.delta_state`: 0.0000 → 14.3750
- `model.layers.29.self_attn.delta_state`: 0.0000 → 24.1250
- ... and 6 more
