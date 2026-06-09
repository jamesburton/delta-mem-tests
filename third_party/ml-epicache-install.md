# Installing EpiCache into our `.venv`

**Submodule:** `third_party/ml-epicache`
**Pinned commit:** `b742661a1b763d0a57f0a1c6b82acbdbe5ed578c` (public release, 2025-10-02)
**Upstream:** https://github.com/apple/ml-epicache
**License:** APPLE-SAMPLE-CODE / non-OSI (see `third_party/ml-epicache/LICENSE`).
Internally MIT-derived from KVzip but the wrapping LICENSE is Apple Sample
Code, which is more restrictive than what the alternatives doc described as
"MIT".

## Why this doc exists

EpiCache ships as a research repo, not a pip package. The repo's `pyproject.toml`
is for an unrelated `kvzip` package and exists only to make `pip install -e .`
register the *root* directory on `sys.path` so that `import attention`,
`import model`, `import utils`, etc. resolve. There is no `epicache` namespace.

## Install procedure (CPU-only steps first)

1. **Acquire the submodule** (already done if you cloned this repo with
   `--recursive`; otherwise):

   ```powershell
   git submodule update --init --recursive third_party/ml-epicache
   ```

2. **Install Python deps that don't need a GPU.** From repo root, into the
   existing `.venv`:

   ```powershell
   .venv\Scripts\python.exe -m pip install `
       "transformers==4.51.0" `
       "datasets" `
       "accelerate" `
       "sentence_transformers" `
       "fuzzywuzzy" `
       "rouge-score" `
       "rouge"
   ```

   Most of these are already in our env; pinning `transformers` is the risky
   one — see "Dependency conflicts" below.

3. **Register the repo on PYTHONPATH** (do NOT run `pip install -e .` from
   `third_party/ml-epicache`; it would install a package named `kvzip` with no
   modules that ever get imported under that name). Two options:

   - **Option A (preferred): set PYTHONPATH per-invocation** in the runner
     wrapper. See the `EPICACHE_PYTHONPATH` handling we will add to
     `run/_chunked_eval_runner.py` once the integration lands.

   - **Option B: drop a `.pth` file** into
     `.venv\Lib\site-packages\epicache.pth` containing the absolute path
     `E:\Development\delta-mem-tests\third_party\ml-epicache`. Loaded on every
     interpreter start. Avoid this for now — pollutes the whole venv with
     unscoped top-level packages (`attention`, `model`, `utils`, `data`,
     `csrc`) that will collide with anything else.

4. **GPU-only steps (do NOT run in Day 1; document for Day 2):**

   ```powershell
   # flash-attn 2.7.4.post1 — requires CUDA toolkit + nvcc on PATH. On
   # native Windows + RTX 3060 this usually fails to build; either install
   # WSL2 + CUDA, use a prebuilt wheel, or fall back to an SDPA replacement
   # patch on attention/attn.py (~50 LOC; see alternatives doc Phase 3).
   .venv\Scripts\python.exe -m pip install flash-attn==2.7.4.post1 --no-build-isolation

   # Custom CUDA kernel for the flattened-cache update path. Requires
   # CUDA_HOME set and an sm_80+ GPU (RTX 3060 is sm_86, OK). The kernel
   # source pins -gencode arch=compute_80,code=sm_80; on Ampere we can
   # build with the default arch flags by editing csrc/build.py if needed.
   cd third_party\ml-epicache\csrc && make
   ```

## Dependency conflicts with our existing env

| Pkg | EpiCache pin | Our current | Resolution |
|---|---|---|---|
| `transformers` | 4.51.0 (req.txt) / 4.51.3 (pyproject) | per OSCAR + delta-mem (~4.50+) | Probably compatible. If our delta-mem path needs newer, pin to 4.51.x for both — Qwen3Attention class is stable from 4.51 onward. |
| `torch` | 2.3.0 (pyproject) | 2.5+ in our `.venv` | Their pin is *advisory*; flash-attn 2.7.4.post1 requires torch ≥2.3. Keep our newer torch and verify flash-attn build picks up the right cuda compute capability. |
| `numpy` | 1.26.4 | ≥2.0 in our env (we have the `np.trapz` shim for turboquant) | Risk: `score.py` and clustering use numpy heavily; if 2.0 breaks them, downgrade in this venv or apply per-call shims (mirror the turboquant `np.trapz = np.trapezoid` pattern in `_chunked_eval_runner.py:76`). |
| `accelerate` | latest | already present | OK. |
| `flash-attn` | 2.7.4.post1 | **not installed** on this host | HARD requirement of EpiCache's `attention/attn.py:13`. Day 2 install. |
| `tiny_api_cuda` (custom) | built via `csrc/make` | not installed | HARD requirement for cache flattening in `attention/kvcache.py:8`. Day 2 build. |

## What this means for Day 1

Today (Day 1, no GPU touch) we can:

- Have the submodule on disk at the right commit.
- Author the Qwen3 monkeypatch port (see `run/epicache_qwen3_adapter.py`).
- Wire the CLI flags and runner branch.
- Smoke-test the *Python wiring* — i.e. imports of EpiCache's pure-Python
  modules and our adapter's `install_epicache_on_qwen3` function — without
  touching the GPU-only `attention.attn` / `attention.kvcache` / `tiny_api_cuda`
  imports. The smoke runs against an `AutoConfig`-only Qwen3-4B stub so it
  completes in <30 s on CPU.

Day 2 (GPU available) does:

1. `pip install flash-attn==2.7.4.post1 --no-build-isolation` (or apply the
   SDPA-fallback patch on `attention/attn.py`).
2. `cd third_party/ml-epicache/csrc && make` (build `tiny_api_cuda`).
3. Run BookSum layer-sensitivity calibration:
   `python data/layer_scores/layer_profile.py --model_path Qwen/Qwen3-4B-Instruct-2507 --input_file <booksum_preproc>`
4. Run conv-41 baseline + EpiCache-only comparison per the alternatives doc.

## Integration-shape caveat (important)

The alternatives doc described EpiCache as "a tiny monkey-patch + custom Cache
subclass". After reading the code:

- The Cache subclass (`EvictCache`) is real, but it does NOT slot into
  `model.generate(past_key_values=...)` like a `DynamicCache` would.
  EpiCache's flow is:
  1. Cluster the conversation history into E=4 episodes (offline, CPU).
  2. For each episode, run `model.prefill_memory_constrained(ctx_ids, ...)`
     which fills an `EvictCache` then triggers eviction down to budget M.
  3. Move each episode's cache to CPU (`kv.to_cpu()`).
  4. Per query: embed query → match to closest episode → restore that
     episode's cache to GPU → call `model.generate(query_ids, kv=...)`.
- This is **a different inference pipeline** from our
  `DeltaMemChatSession._ingest_full_ids → _decode_generate` loop. We can't
  drop EpiCache in as just another `KV_CACHE_BACKEND` value the way we did
  OSCAR. The wiring will need to construct the episode caches at
  history-prefill time and dispatch the right one per question.
- The runner branch we add in Day 1 is therefore a **placeholder** that
  raises NotImplementedError when actually invoked at eval time, but provides
  the import surface so the CLI accepts the new value and the smoke test
  can validate the Python wiring. Day 2 will replace it with the real
  episode pipeline.
