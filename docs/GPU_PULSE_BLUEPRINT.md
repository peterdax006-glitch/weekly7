# GPU pulse blueprint: getting the most thinking out of ~$8

Owner, 3 Oct 2026: rent a GPU in short pulses (about $8 total), not 24/7; "make sure we have the blueprint on how we fill fully utilize it".
Target: Vast.ai offer #44585860 - 1x RTX 4090 24 GB, $0.343/h (+ bandwidth, + storage while it exists), 'Llama.cpp' template
(image vastai/llama-cpp, CUDA 13). $8 = about 23 GPU-hours. Runner: scripts/gpu_pulse.py (branch h41/gpu-pulse).

Everything below is chosen from today's measurements on the home PC (CPU-only, shared with Nupen's own work):
Qwen3-1.7B ~1-5 tok/s per sequence under load, Qwen3-4B ~0.3-0.5 tok/s, judgment answers 30-120 s each, a 400-subject judgment
round x ~12 strategies (~2,500 calls) = 6-7 h. A 4090 runs these models one to two orders of magnitude faster, so the scarce
resource on a pulse is not GPU speed - it is idle GPU time. The whole design is about never letting the GPU wait.

## 1. Rules that decide whether the money is well spent

1. **Zero setup on the clock.** Every job list, prompt set, case set and public-repo text cache is prepared on the PC before
   renting. Models download on the pod at datacenter speed (seconds to minutes) and are sha256-checked. No compiling: the
   template's llama-server is used as installed.
2. **Keep the GPU saturated.** One llama-server per model with many parallel slots (`-np 8..16`, continuous batching, `-ngl 99`),
   and the PC side keeps at least as many requests in flight as there are slots. Watch `nvidia-smi` utilisation: below ~70% busy
   for more than a few minutes means the client side is the bottleneck - raise concurrency, never just wait.
3. **Bank lasting value, not just scores.** Each pulse must leave something that keeps improving Nupen on the CPU afterwards
   (decisions, labelled data, worked examples) - see section 3.
4. **Stop the meter.** The budget guard stops jobs at the cap; the pod stops itself when the job list is done
   (`vastai stop/destroy instance $CONTAINER_ID` from inside). Destroy (not just stop) when the models need not be kept:
   storage is billed while a stopped instance exists.
5. **Nothing private leaves the PC.** Only public data (public repos' text caches, the frozen benchmark inputs, public-commit
   judgment cases, reasoning questions from public repos), model files and Nupen's code. Never state/livesim, the ~/oldpc private
   projects (they exist only privacy-stripped and are not uploaded), Masterstock, or any secret. Results come back over the tunnel.
6. **Same rules as at home.** Every record is tagged with its model and `gpu_pulse`, CPU and GPU timings never mix in throughput
   statistics, frozen benchmark items never become training data, and the trust gate is never touched.

## 2. Pulse plan (~23 GPU-hours)

| Pulse | Length | Cost | Jobs | Decision it buys |
|---|---|---|---|---|
| 1 Smoke + bake-off | ~1.5 h | ~$0.55 | runner self-test (1 min); frozen thinkbench `--compare --model thinker` for Qwen3-1.7B, 4B, 8B, 14B (Q4_K_M; 14B fits in 24 GB); throughput probe per model at 1/4/8/16 slots | Which thinker is best per job (judgment vs reasoning) and its real cost per answer on GPU - with paired CIs on the frozen items |
| 2 Judgment rounds | ~5 h | ~$1.70 | public-commit judgment (pub_git_fixed, pub_git_churn) rounds of 400 fresh subjects with racing (h36), the anchor-on-statistical-p strategy, the best model from pulse 1; plus the 48 verdict cases | Which judgment strategy really beats the free statistical predictor (out-of-sample re-test on the next round); today 7.5 CPU-hours bought a 'winner' that was worse than statistics |
| 3 Reasoning epochs | ~4 h | ~$1.40 | reasoning MC drills (~60,000 questions from public repos) for several epochs per strategy (plain, cot, eliminate, retrieve, vote) with the online allocation | Which reasoning method helps on fresh questions, with hundreds of paired items per arm instead of 16 |
| 4 Teacher traces (distillation data) | ~6 h | ~$2.10 | the strongest model that fits (8B or 14B, chosen in pulse 1) answers tens of thousands of reasoning questions and judgment cases with short worked reasoning; every trace is scored against the known answer; only correct, outcome-blind traces are kept | A large bank of correct worked examples that the CPU's 1.7B uses through retrieval (its 'learn from earlier solved questions' strategies) long after the pulse - lasting value |
| 5 Reserve | ~6 h | ~$2.10 | re-run the best configuration on newly acquired public data; or (owner's call only) run the language-model trainer UNCHANGED with `torch_device=cuda` | Re-test of earlier winners on new data; the LM option stays the owner's decision (hands-off rule) |
| Overheads | - | ~$0.15 | bandwidth for model downloads (~2.5 / 5 / 9 GB) and results; storage while the instance exists | - |

Pulse 1 decides the rest: if the 8B or 14B is clearly better and fast enough, pulses 2-4 use it; if not, the 4B.

## 3. Lasting value after the GPU is gone

- **Decisions** (pulse 1-3): which model and which strategies Nupen should use on the CPU, decided on enough paired data that
  they are not noise.
- **Worked-example bank** (pulse 4): correct, outcome-blind reasoning traces keyed by question type and repo, stored in
  state/creator/thinking/ and used by the CPU model's retrieval strategies (strictly earlier / other items only, as today).
- **Calibration data**: thousands of scored probabilities per strategy, so the trust gate on the CPU starts with real evidence.
- Everything is recorded in progress.jsonl so the learn loop shows the before/after curve.

## 4. During a pulse (runner checklist)

1. `setup`: SSH in, confirm `nvidia-smi` and the template's llama-server, download + sha256-check the models, start one server
   per model with N slots on distinct internal ports, open the SSH tunnel.
2. `run <pulse>`: keep requests in flight = slots; log tok/s and GPU utilisation every minute; abort a job whose utilisation stays
   low and report why.
3. Copy results into Nupen's state as each job finishes (a cut-off pulse loses at most the job in flight).
4. `teardown`: stop servers; stop or destroy the instance; record spend in the local pulse ledger (outside the repo).

## 5. What the owner does

1. Generate the pulse SSH key and paste its public half into the Vast.ai console (one-time; the runner's checklist has the exact
   commands).
2. Rent offer #44585860 (or any verified RTX 4090 at about $0.35/h, reliability > 98%) with the 'Llama.cpp' template, ~80 GB
   disk, SSH on.
3. Give the teacher the SSH host and port; the runner does the rest and reports spend after each pulse.

## 6. Interface for other modules' GPU work (runner: creator/gpupulse.py)

- **Jobs.** A job list may mix the built-in strings (`smoke|probe|thinkbench|judgment|drills|traces:<model>[:<n>[:<max min>]]`) and
  external-job dicts (`creator.gpupulse.ext_job`): `name`, `minutes` (the plan's estimate), exactly one of `call`
  (`"package.module:function"`, run in a child process on the PC with the pulse on; gets `{state, repo, owner_dir, workers, deadline,
  pulse, tunnel_file, model, endpoints}`, returns a JSON dict), `command` (local argv, same environment) or `remote` (bash run on the pod
  in `remote_dir`; print `@@result=<json>`); optional `model` (served first), `free_gpu` (stop all servers: training needs the VRAM),
  `max_minutes`, `outputs` (pod paths copied home to `<runtime>/gpu/outputs/<pulse>/<name>/`), `low_util_abort_minutes`.
  Run them with `gpu_pulse.py run|plan|prepare --jobs-file list.json` or `--jobs-from package.module:function` (called with the config).
  Budget cap, GPU monitor, ledger and `gpu_pulse_runs.jsonl` records apply to every job.
- **Tunnel.** `creator.gpupulse.endpoints()` -> `{"pulse", "models": {model: {"urls": ["http://127.0.0.1:<port>/v1"], "slots"}},
  "extra": {name: url}}` (file: `<runtime>/gpu/tunnel.json`). Local port of model k of the config's `models` list = `local_port_base` (18100)
  + 10 k; config `extra_forwards` `{name: pod port}` adds more forwards (e.g. a training dashboard). Loopback only; prompts must pass
  `outbound_ok()`.
- **Setup.** Config `setup_steps` `[{"name", "script" | "script_file", "always", "timeout_s"}]` run after the models are verified and before the
  servers start, once each (marker `run/step_<name>.done`); in-process hooks: append to `creator.gpupulse.SETUP_HOOKS`
  (`hook(cfg, shell, say)`). This is where a training stack (Unsloth/Axolotl) is installed if the Llama.cpp image is kept.
