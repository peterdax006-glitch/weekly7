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

## 1. Host, image and training stack (provider- and GPU-agnostic)

The original offer is gone. The owner rents wherever value is best at the time: Vast.ai, RunPod or any SSH-able Linux NVIDIA host. All
choices follow from the card the pod reports, not from one template.

**Image, best first.** `pod_setup.sh` handles all three. It is a runner `setup_step`, every phase is timed, and it prints `@@gpu=` and
writes `gpuday/gpu.json`.

| Provider | Image / template | Torch + Unsloth on the clock |
|---|---|---|
| Vast.ai | "Unsloth Studio" `vastai/unsloth-studio:2026.9.14-cuda-12.9-py312` (9.3 GB, 2 Oct) | none: checked only (~1 min) |
| Vast.ai | "PyTorch" `vastai/pytorch:cuda-12.8.1-auto` (or `cuda-12.9.2-auto`) | torch kept; pip Unsloth/TRL/PEFT (~3-6 min) |
| RunPod | "PyTorch" `runpod/pytorch:1.4.1-rc.171-cu1290-torch291-ubuntu2204` (1 Oct) | torch kept; pip Unsloth/TRL/PEFT (~3-6 min) |
| Vast.ai | "Llama.cpp" `vastai/llama-cpp:<b>-cuda-12.9` | torch wheel matched to the DRIVER, then Unsloth (~5-15 min, last resort) |

Torch wheels are matched to the driver. If an image's torch cannot run a 64x64 matmul kernel on the card, `pod_setup.sh` installs from
the PyTorch index that fits the driver's CUDA: cu128 for drivers at 12.8 or later, else cu126 or cu124. A Blackwell card (compute
capability 12.x, e.g. RTX 5090) on a driver older than CUDA 12.8 stops setup at once with "rent another host". After installing Unsloth
it re-checks torch, in case Unsloth replaced it with a wheel that lacks the card's architecture.

llama.cpp: the runner's own fallback installs the ggml-org prebuilt CUDA 12.8 release (~1 min) wherever the image has no `llama-server`.
`pod_setup.sh` adds the source at the same tag (convert scripts and gguf-py) and `llama-quantize`, building that one tool on the CPU if
needed. Open point: confirm that the prebuilt 12.8 release includes sm_120 kernels before renting a 5090. If it does not, the runner's
`allow_build` (cmake, `CMAKE_CUDA_ARCHITECTURES`) is the fallback, and it currently pins 89 (H-ask in section 7).

**Card -> choices** (`creator.gpuday.gpu_profile`, from `@@gpu=` = name, memory, compute capability; tested):

| Card | VRAM | Coder fine-tune | 4B / 1.7B batch | Precision | Train / gen speed vs 4090 |
|---|---|---|---|---|---|
| RTX 3090 (sm_86) | 24 GB | Qwen3-14B QLoRA, batch 1, 12k tokens | 4 / 8 | bf16, no FP8 | 0.55 / 0.93 |
| RTX 4090 (sm_89) | 24 GB | Qwen3-14B QLoRA, batch 1, 12k | 4 / 8 | bf16 | 1.0 / 1.0 |
| RTX 5090 (sm_120) | 32 GB | Qwen3-14B QLoRA, batch 2, 16k | 8 / 16 | bf16; CUDA >= 12.8 | 1.35 / 1.75 |
| 48 GB 4090 / A6000 / L40S | 48 GB | Qwen3-Coder-30B-A3B QLoRA, batch 2, 16k | 16 / 32 | bf16 | per card |
| <= 16 GB | 16 GB | Qwen3-8B QLoRA | 2 / 4 | bf16 or fp16 | per card |

`finetune.py` also decides precision on the pod (`torch.cuda.is_bf16_supported()`: bf16, else fp16). No FP8 path exists anywhere.
QLoRA always merges into the 16-bit base (`--merge-base`).

**Time scales with the card.** Each block has a fixed slot. `day_plan(gpu)` scales the 4090-sized work by training speed for the
fine-tune and RL blocks, and by memory bandwidth for generation blocks. Each job's `minutes` / `max_minutes` are scaled the same way
in `day_jobs`. Example: `jobs.py plan --gpu "NVIDIA GeForce RTX 3090, 24576 MiB, 8.6" --rate 0.16` shows FINE-TUNE 2 needing ~9.1 h of
work in a 5 h slot. On a 3090, either accept fewer epochs (the job is cut at max_minutes, and adapters are only saved at the end, so
lower `--epochs`), or rent longer. The cost is 24 h x $0.16 = $3.84.

**Why Unsloth and not Axolotl.** This is one GPU and one day. Unsloth's loader halves VRAM and roughly doubles speed for LoRA and QLoRA.
It supports Qwen3 dense and MoE, and DPO / ORPO / GRPO run through the same TRL trainers. Axolotl's YAML adds a layer we would debug on
the pod. The exported data loads in Axolotl unchanged (`type: chat_template`, `field_messages: messages`).

**VRAM, from the sources.** Unsloth's Qwen3 guide: "Qwen3 (14B) fits comfortably in a Google Colab 16GB VRAM Tesla T4 GPU" (QLoRA), and
"Qwen3-30B-A3B works on just 17.5GB VRAM with Unsloth". For MoE QLoRA "the full 16-bit model must be downloaded and converted to 4-bit on
the fly" (~61 GB for 30B-A3B: ~10 min at ~900 Mbps). Disk: 150 GB for the 30B option, 80 GB otherwise. These are vendor claims. The first
training job's `result.json` records the real numbers.

**Owner checklist, per provider.** Pick the image from the table. Disk: 80 GB, or 150 GB for a 48 GB card (30B coder). SSH on. Add the
`gpuday_stack` setup step to `pulse.json`. After `setup`, paste the `@@gpu=` line as `"gpuday_gpu"` in `pulse.json` (or pass `--gpu` to
`jobs.py`). Set `usd_per_hr` to the rented rate.

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
| H3 serve a tuned model | DONE. Each FT job hard-links its Q4_K_M GGUF to `remote_dir/models/<name>` (+ `<name>.ok`). The next job, `register_<ft job>` (`creator.gpuday:register_tuned_job`, args `{source, serve_as}`), reads that FT job's row of THIS pulse from `<state>/thinking/gpu_pulse_runs.jsonl` and writes `{name: {bytes, sha256}}` into the extra_models file (config `extra_models_file`, else env `NUPEN_GPU_EXTRA_MODELS`, else `<runtime>/gpu/extra_models.json`; pulse.json is never rewritten). The runner lays that file over config `extra_models` on every read, re-hashes the file ON THE POD before serving (size + sha256; mismatch refused), and never downloads it. A job whose model cannot be served (FT skipped, thin data, short disk) is recorded as skipped and the day goes on. |
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

## 9. Pod facts that shape the scripts (live 5090 pod, 3 Oct)

- **Python.** Ubuntu 24 system Python is PEP 668 externally-managed. `pod_setup.sh` picks `$GPUDAY_PY`, else `/venv/main/bin/python`
  (Vast images), else `gpuday/venv`, else system python3 only if it is not externally managed (otherwise it makes
  `gpuday/venv --system-site-packages`). It never uses `--break-system-packages`. The choice is written to `gpuday/python`, and every pod job
  (ft, embed, rl) reads it (`creator.gpuday.POD_PY`). `PIP_NO_CACHE_DIR=1` is set.
- **Torch.** If the driver is CUDA >= 13.0, setup tries the cu130 index first, then cu128, and keeps whichever runs a kernel on the card.
  A torch that already works is kept.
- **llama.cpp.** The source (convert scripts, gguf-py) is cloned at the pod's `llama-server --version` build (`b<N>`), so the converter matches
  the server. `$GPUDAY_LLAMA_TAG` overrides this. `llama-quantize` is also looked for under `/opt/llama.cpp`.
- **Disk.** Each fine-tune checks free space in the working directory and in `$HOME` first. If either is short, it skips itself with
  `disk: N GB free, needs ~M GB`. It needs (`FT_DISK_GB`, GB) 1.7B 12, 4B 27, 8B 60, 14B QLoRA 110, 30B 210. With
  `"gpuday_ft_gguf": false` the jobs train adapters only, with no merge, GGUF or serving, and need the base download alone (`FT_DL_GB`:
  5 / 10 / 8 / 12 / 64). The adapters come home through `outputs`. `"gpuday_skip_disk_check": true` turns the check off.
