# GPU day prep: 24 hours on one RTX 4090 with nothing set up on the clock

Owner, 3 Oct 2026: one Vast.ai RTX 4090 24 GB (offer #44585860, CUDA <= 13.1 driver, ~880 Mbps, 499 GB NVMe, 45 GB RAM, $0.343/h)
for 24 hours, about $8.23. Every block runs from a job list the pulse runner executes (creator/gpupulse.py, branch h41/gpu-pulse,
merged as h45/batch5). This branch (h44/gpu-prep) adds the parts the runner calls. The owner's own language model is not part of this day.

| Piece | File | Where it runs |
|---|---|---|
| data export, privacy + frozen filters, splits, harness, embedding index, day plan, runner job list | `creator/gpuday.py` (loaded on demand only) | PC |
| export CLI | `scripts/gpuday/export_data.py` | PC (IDLE priority) |
| fine-tune pipeline: SFT / DPO / ORPO -> merge -> GGUF -> sha256 | `scripts/gpuday/finetune.py` | pod (Unsloth) and CPU test (transformers + peft) |
| training-stack setup step | `scripts/gpuday/pod_setup.sh` | pod (runner `setup_steps`) |
| coder harness CLI | `scripts/gpuday/coder_harness.py` | PC (IDLE), model on the pod through the tunnel |
| embedding index build + CPU query | `scripts/gpuday/embed_pod.py` | pod (GPU) and PC (CPU) |
| RL proof of concept (GRPO, unit-test reward) | `scripts/gpuday/rl_grpo.py` | PC (`make-tasks`), pod (training) |
| job list / plan | `scripts/gpuday/jobs.py`, `creator.gpuday:day_jobs` | PC |

## 1. Template and training stack: one choice

**Rent the 4090 with the Vast "Unsloth Studio" image, `vastai/unsloth-studio:2026.9.14-cuda-12.9-py312`, not the Llama.cpp image.
llama.cpp gets installed on it by the runner's existing fallback.**

- The image already has torch (CUDA 12.9 build), Unsloth, TRL, PEFT and transformers (9.3 GB image, pushed 2 Oct 2026; Docker Hub tag list
  read 3 Oct). The CUDA 12.9 tag runs natively on the host's 13.1 driver.
- llama.cpp on that image: the runner's `setup` finds no `llama-server`, so it runs `fallback_install_script`. That script downloads the
  ggml-org prebuilt CUDA 12.8 release (`b11379`, a tarball of tens of MB) plus cudart. At ~880 Mbps that takes **about 1 minute**, with no
  compiling. `pod_setup.sh` (a runner `setup_step`) then shallow-clones the llama.cpp source at the same tag to get
  `convert_hf_to_gguf.py` / `convert_lora_to_gguf.py` and pip-installs its `gguf-py` (**about 0.5-1 minute**). It also finds
  `llama-quantize` from that release; if the release lacks it, it builds just that one tool on the CPU (1-3 minutes). It prints the time of
  each phase (`@@t_*`).
- The alternative was keeping `vastai/llama-cpp:...-cuda-12.9` and pip-installing Unsloth at pod start. That means torch, triton,
  bitsandbytes, xformers and unsloth_zoo: **about 5-15 minutes**, over 4 GB of wheels, and risks resolver or CUDA-wheel mismatches that are
  then debugged on the paid clock. `pod_setup.sh` still has this as a slow fallback, but it is not the plan.
- Total setup on the clock: about **2-3 minutes** of installs, plus the image pull (9.3 GB). The pull happens while the instance is
  "loading", before the GPU rental starts. Model downloads go into the same 0:00-0:30 block either way.
- **Why Unsloth and not Axolotl.** This is one GPU and one day. Unsloth's loader halves VRAM and roughly doubles speed for LoRA and QLoRA. It
  supports Qwen3 dense and MoE, and DPO / ORPO / GRPO through the same TRL trainers. Axolotl's YAML adds a layer we would debug on the pod.
  The exported data still loads in Axolotl unchanged (`type: chat_template`, `field_messages: messages`).
- **VRAM, from the sources.** Unsloth's Qwen3 guide: "Qwen3 (14B) fits comfortably in a Google Colab 16GB VRAM Tesla T4 GPU" (QLoRA), and
  "Qwen3-30B-A3B works on just 17.5GB VRAM with Unsloth". For MoE QLoRA "the full 16-bit model must be downloaded and converted to 4-bit on
  the fly" (about 61 GB for 30B-A3B: ~10 min at 880 Mbps; the disk is ample). Qwen3-1.7B and Qwen3-4B with bf16 LoRA need about 4 GB and
  about 9-10 GB of weights plus activations, so both fit 24 GB at 4-8k tokens with Unsloth's gradient checkpointing. These are vendor
  claims. The first training job's `result.json` records the real time.
- **Owner checklist change.** In section 3 of `gpu_pulse.py checklist`, choose template "Unsloth Studio" (image tag above) instead of
  'Llama.cpp'. Set disk to **150 GB** (needed by the 30B option: 61 GB download plus merge; 80 GB is enough for the 14B plan). Keep SSH on.
  The Studio web UI may stay off.

## 2. Training data: what exists (honest counts, export of 3 Oct 2026 ~14:05 local)

`python scripts/gpuday/export_data.py` writes to `~/creator_runtime/gpuday/export/` (refused inside the repo):

| File | Rows | What it is |
|---|---|---|
| `handoff_train` / `handoff_eval` | **6 / 1** | GOLD: task text (.creator_task.md) -> the change the kernel ADOPTED, as search/replace edits |
| `pref_train` / `pref_eval` | **2 / 0** | same requirement: adopted answer vs a rejected/failed one (TRL conversational prompt/chosen/rejected) |
| `commit_train` / `commit_eval` | **110 / 27** | Nupen's own public history: commit message -> its diff (creator/, tests/, scripts/), a lower tier |
| `handoff_unjudged` | **44** | teacher answers the kernel never judged (interrupted runs); kept apart, not in the default mix |
| `worked_train` / `worked_eval` | **0** today | correct worked examples from the runner's bank (P4 traces, block 2:30-6:30) |
| `coder_sft_mix` | **116** | handoff_train + commit_train: the FINE-TUNE 2 SFT set |
| `embed_docs` | ~1,500 | commit messages, lesson objectives+reasoning, plan explanations |
| `rl_tasks` | 37 | pure functions of creator/*.py with oracle asserts (RL proof of concept) |
| `splits.json` | judgment 222/55, reasoning 7/2 | held-out ids, time-ordered |

Dropped in that export: 94 rows too long for 12k tokens, 95 with a private marker (mostly commits that mention the owner's journal or
code that names the live-sim state), and 53 frozen-benchmark rows.

**Verdict.** The kernel's own adoption history is tiny: 7 adopted handoffs and 2 clean preference pairs. That is not a fine-tuning
dataset. It is an evaluation set. FINE-TUNE 2 therefore trains on the 116-row mix, which is mostly public commits, and the DPO step is
skipped automatically below 20 pairs (`gpuday_pref_rows`). The honest expectation for FINE-TUNE 2 is a format/house-style gain, not new
competence. The held-out harness is what says whether it helped. FINE-TUNE 1 depends entirely on the bank the runner produces in block 3
(it needs at least 200 correct rows, or the job skips itself).

**Formats.** SFT: `{"messages": [system, user, assistant]}`; the training script converts to TRL prompt/completion so the loss falls on
the answer only. Preference: `{"prompt": [...], "chosen": [...], "rejected": [...]}`. Ids and metadata live in `*.ids.jsonl` sidecars.
The coder reply format is `creator.generator.EDIT_RE` (REASONING line + FILE / SEARCH / REPLACE blocks), so a tuned coder plugs into the
model student and the harness unchanged.

**Privacy (tested).** Every row passes `private_reason`: the owner's journal / Masterstock, live-sim data, the ~/oldpc projects, private
keys, API tokens and the button token. The only allowed mention is the protected-path list's `state/livesim/*`, which is removed. Rows that
fail are DROPPED, never redacted. `scrub_obj` then removes author names, e-mail addresses and home paths. Author identity is never read
from git. `audit_export` re-reads every file before any upload, and `upload_bundle` refuses the whole upload if it finds anything.

**Frozen thinkbench (tested).** `Frozen.load` refuses any items.json whose hash is not `1f520b202ac4`. Excluded from every export: the
verdict subjects (CP0065...CP0152, whose outcome IS the label), the git subjects (commit prefixes), all item ids/subjects (including
journal ids and qids), and the exact text of every question prompt and planning context. With `--strict`, the planning candidates
(CP0001-CP0060) are excluded too. Their label is an ordering, not an outcome, so they stay in the default export.

**Splits.** `time_split`: every training item is strictly earlier than every eval item. A group (requirement / package / question) never
has members on both sides: a group that started before the cut keeps its later members out of eval.

## 3. Fine-tune pipeline (`scripts/gpuday/finetune.py`)

`pipeline` = `sft` (LoRA; `--qlora` for 14B/30B) -> optional `pref` (`--method dpo|orpo`) -> `merge` (always into the 16-bit base,
`--merge-base`) -> `gguf` (convert_hf_to_gguf.py f16 -> llama-quantize Q4_K_M) -> sha256. Every step appends to `<out>/result.json`
(seconds, losses, file bytes + sha256) and prints `@@result=`. `lora-gguf` writes the adapter alone as GGUF (tens of MB) for
`llama-server --lora`.

CPU proof (3 Oct): `trl-internal-testing/tiny-Qwen3ForCausalLM` (real Qwen3 architecture and tokenizer, random weights) ran the whole
pipeline on this PC at IDLE priority, through the transformers + peft path: SFT on 4 exported rows -> DPO -> merge -> GGUF f16 ->
llama-quantize Q4_K_M -> sha256. Results are in section 8. The Unsloth loader branch is the only part that cannot run on a CPU. It is the
documented `FastLanguageModel.from_pretrained` / `get_peft_model` call, and the trainer code after it is the same code the CPU test ran.

## 4. Coder harness (`creator.gpuday.replay_case` / `run_harness`)

Each held-out case (commit_eval: 27, handoff_eval: 1) is replayed in its own fresh sandbox (`creator.sandbox.Sandbox.open`) at the
recorded base commit of a SEPARATE clone (`~/creator_runtime/gpuday/harness_repo`), so the main repository's refs are never touched. The
prompt is the same `coder_messages` the training rows carry. The model is served by llama-server (through the tunnel, or locally). Its
answer is parsed and applied with `creator.generator.parse_edits` / `apply_edits`. Judging uses the kernel's sandbox gates that need no
ledger: build OK, affected tests CLEAN against the base (`Sandbox.evaluate`), no weakened tests (`check_test_weakening`), and every
declared output present. The sandbox is then discarded (`S.discard`); nothing is adopted or merged.

"would pass" is necessary but not sufficient for an adoption: the ledger claim verdict, the post-merge re-assessment and the start-load
guard are not run. Best-of-N: N generations per case, and the tests pick the first passing one. The report gives pass@1 and pass@N.
Home-side load follows the teacher's rule: the process sets itself to IDLE priority, generations run 4 cases ahead (GPU busy), and
evaluations run one at a time.

## 5. Embedding index

`embed_pod.py build` starts `llama-server --embedding --pooling <model's>` and embeds `embed_docs.jsonl`. It writes `vectors.npy`
(float16, L2-normalised), `docs.jsonl` (id, kind, time, 600-char snippet) and `meta.json` (model + sha256). For ~1,500 docs x 1024 dims
that is about 3 MB. The same GGUF is used on both sides, always through llama.cpp, so pod vectors and PC query vectors are comparable.
Day model: `Qwen3-Embedding-0.6B-Q8_0.gguf` (639 MB, sha256 pinned in `EMBED_MODELS`). The PC query path (CPU, IDLE) is
`embed_pod.py query --index ... --model <same gguf> --cpu "text"`, and numpy is the only Python dependency. It is tested on the CPU with
`bge-small-en-v1.5-q8_0.gguf` (37 MB) and the local llama-server.

## 6. RL proof of concept: smallest honest version

`rl_grpo.py make-tasks` turns pure top-level functions of creator/*.py (1-3 simple annotated parameters, deterministic, non-constant) into
tasks. The original function is the oracle for sampled inputs. The prompt shows the signature, the docstring and two examples. Tasks are
held out by module. Reward: 0.1 for a parsable `def`, plus 0.9 x the fraction of asserts that pass when the completion runs in a separate
`python -I` process with a 5 s timeout. Training is TRL `GRPOTrainer` (num_generations 8, beta 0) with the same loader as finetune.py.
Success means the held-out reward goes from before to after, never the train reward.

Readiness: the code path is tested on the CPU with the tiny model for 2 steps (section 8). That proves the data, reward and trainer
wiring, not learning. The task bank is small (37 tasks) and partly trivial (identity-like functions). A real RL run needs a few hundred
tasks. The next source is Nupen's own unit tests (run a test file against a model-written module) instead of oracle asserts. Verdict:
**runnable on the day as a proof of concept; a negative held-out result is the most likely honest outcome.**

## 7. Runner hook points (what the runner calls; no runner file edited here)

| Hook | Interface |
|---|---|
| H1 job list | `gpu_pulse.py run --jobs-from creator.gpuday:day_jobs` (or `scripts/gpuday/jobs.py write` -> `--jobs-file`). Built-in strings plus `ext_job` dicts: `gpuday_upload` (remote), `coder_trial_*` / `best_of_n_goals` (call `creator.gpuday:harness_job`), `embed_index` / `ft1_*` / `ft2_coder` / `rl_poc` (remote, `free_gpu`, `outputs`), `gpuday_reexport` (call). |
| H2 setup | config `"setup_steps": [{"name": "gpuday_stack", "script_file": "scripts/gpuday/pod_setup.sh", "timeout_s": 1800}]` |
| H3 serve a tuned model | each FT job copies its Q4_K_M GGUF to `remote_dir/models/<name>` and writes `<name>.ok` = sha256 (the runner's own "verified copy" marker). The runner still needs a catalog entry to serve a name that is not on Hugging Face. **Ask of the runner:** accept config `extra_models: {name: {bytes, sha256}}` (no download, served when present), or let a job's result register one. Until then the `thinkbench:<tuned>` and `coder_trial_tuned` jobs fail fast and say so. |
| H4 upload | there is none in the runner by design. `gpuday_upload` is a remote job whose script carries an audited base64 tarball (scripts + this module as `gpuday_lib.py` + the export's training files; sha256 checked on the pod), so it passes `outbound_ok`. |
| H5 outputs | small files only: `result.json`, `adapter/` (LoRA safetensors 35-260 MB), `index/`. Big GGUFs stay on the pod; bring one home only for a winner (an `outputs` entry naming the GGUF, 1.1 / 2.5 / 9 / 18.6 GB). |
| H6 env | `harness_job` reads `GPUDAY_N` (best-of-N, default 1); the export dir is `~/creator_runtime/gpuday/export` (cfg `gpuday_export`). |

## 8. Plan, cost, and readiness per block

`python scripts/gpuday/jobs.py plan` prints this table (from `creator.gpuday.DAY`):

| Hours | $ | Block | Comes home | Ready? |
|---|---|---|---|---|
| 0:00-0:30 | 0.17 | setup + probe (runner) + `gpuday_stack` (~2-3 min) | probe json | READY |
| 0:30-2:30 | 0.69 | thinkbench 1.7B/4B/8B/14B + coder trial (harness, 14B, n=1) | results, harness json | READY |
| 2:30-6:30 | 1.37 | worked bank (runner `traces`) + embedding index | bank (5-30 MB), index (~3 MB) | READY |
| 6:30-10:00 | 1.20 | judgment / reasoning rounds (exist) | state records | READY |
| 10:00-13:00 | 1.03 | FINE-TUNE 1: worked bank -> 1.7B, 4B LoRA -> GGUF -> frozen thinkbench | adapters (35-70 MB) | READY, NOT YET RUN ON GPU; needs H3 to evaluate |
| 13:00-18:00 | 1.72 | FINE-TUNE 2: coder SFT (14B QLoRA) [+ DPO if >= 20 pairs] -> harness on held-out | adapter (~130-260 MB) | THIN DATA (7 gold rows) |
| 18:00-21:00 | 1.03 | best-of-N (n=8) on held-out / open goals; home kernel judges | patches + json | READY, NOT YET RUN ON GPU |
| 21:00-23:00 | 0.69 | RL proof of concept (GRPO) | result.json | READY, NOT YET RUN ON GPU (tiny CPU run only) |
| 23:00-24:00 | 0.34 | results home + teardown --destroy | | READY |

The total is 24 h, about $8.23 plus bandwidth. If FINE-TUNE 1 skips itself because the bank is too thin, its 3 h go to more traces or
judgment rounds.
