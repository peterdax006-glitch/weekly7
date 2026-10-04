# GPU guardian (h64)

The owner's target for the rented GPU is an average of 90% or more, with useful work. The guardian lives on the pod: cron and a respawn
loop keep it running without the PC, SSH or an agent. When the GPU is under 90%, it fills the spare capacity with the highest-priority
unfinished work that fits in free VRAM. It gets out of the way the moment the runner needs the card.

## Files

### Pod side

These live in `scripts/util3/guardian/` and are installed to `/root/guardian/`.

| file | role |
|---|---|
| `install.sh` | Installs the cron entries (`* * * * *` and `@reboot`, `flock -n -o run.lock`) and starts the guardian. |
| `run.sh` | Respawns `guard.sh` 3 s after any exit. A `STOP` file keeps everything down. |
| `guard.sh` | The control loop, run once per second. Details below. |
| `feed.py` | Drives every port tagged `traces`. It answers the staged public reasoning questions into `out/<model>.jsonl`, then retries the unsolved ones. |
| `coder.py` | Drives every port tagged `coding`. Public coding tasks only, with `TEACHER_BRIEF.md` as the system prompt. Details below. |

What `guard.sh` does each second:
- Reads GPU utilisation, free VRAM and the runner's GPU processes.
- Turns every filler off within about 1.5 s when a runner 8B+ llama-server starts, or when any new runner GPU process appears.
- Holds the refill for 130 s after a new fine-tune starts. While training runs, it reserves that run kind's measured peak + 2 GB (`ftpeaks.txt`).
- In a runner gap (fetch, download, upload), keeps its own servers running and scales up to 4 servers while utilisation is under 90%.
- Fetches missing queue models in the background, sha256-checked.
- Disk guard: keeps `/` at 6 GB or more free, deleting only stale ladder variants, `.part` files and regenerable chunks. It never touches HF caches, base GGUFs, files a process has open, or queued models.
- Writes one log line per minute to `/root/guardian/log`. Events go to `events.log`.

What `coder.py` writes for each task:
- Role-tagged rows: SPEC, PLAN, CODE, REVIEW, DEBUG, VALIDATE and LESSON, plus one `trajectory` row.
- Every row carries `verified`, decided by real runs on the pod CPU: python -I, nice 10, a 2 GB limit, at most 4 at once.
- Failures are kept.
- Rows go to `out_roles/<model>.jsonl`.

### PC side

These live in `scripts/util3/`.

| file | role |
|---|---|
| `guardian_sync.py stage` | Stages the public reasoning questions on the pod. Nupen's own repository is never staged. |
| `guardian_sync.py stage-coding` | Stages the public coding tasks (rl_tasks_hf + rl_tasks_more, train split) and the teacher brief. |
| `guardian_sync.py loop` | Pulls the pod's answers. Only answers re-checked against this PC's own question and passing `trace_ok` reach the trace bank. Role rows go to `~/creator_runtime/gpuday/trajectories/roles/`. Run it at idle priority. |
| `guardian_queue.py` | Writes `queue.txt` and drives the effladder jobs (`creator.effladder.run`) against the port that serves them. Sends a heartbeat each minute; without it the pod serves no PC-driven job and falls back to the pod-driven items. |

## The work queue

`/root/guardian/queue.txt` holds one item per line: `<prio> <model file> <tag> [<bytes> <sha256> <url>]`. A lower number means a higher
priority. Owner order: efficiency first, then coding, then language, with traces last.

| tag | driven by | instances | server shape |
|---|---|---|---|
| `effladder` | the PC (needs the heartbeat) | 1 per line; the line is removed when done | 8 slots × 4096 tokens, f16 KV (the runner's ladder conditions) |
| `coding` | `coder.py` on the pod | any number; the line is removed when every task is done | 8 slots × 4096 tokens, q8 KV |
| `traces` | `feed.py` on the pod | any number | 16 slots × 1024 tokens, q8 KV |

A higher-priority item that would fit in the newest instance's VRAM replaces that instance. Each port's tag is written to
`/root/guardian/port_<port>.tag`, so the drivers know which ports are theirs.

## Install on a fresh pod

1. Copy the guardian files to the pod and strip carriage returns:

   ```
   scp scripts/util3/guardian/{guard.sh,run.sh,install.sh,feed.py,coder.py} root@POD:/root/guardian/
   ssh root@POD 'sed -i "s/\r$//" /root/guardian/*; bash /root/guardian/install.sh'
   ```

2. Stage the work from the PC, in the main checkout, at low priority:

   ```
   python scripts/util3/guardian_sync.py stage --host H --sshport P [--relay host:port]
   python scripts/util3/guardian_sync.py stage-coding --host H --sshport P
   ```

3. Start the two PC loops:

   ```
   python scripts/util3/guardian_sync.py loop --host H --sshport P
   python scripts/util3/guardian_queue.py --host H --sshport P
   ```

The pod needs `/opt/llama.cpp/llama-server`, the base GGUFs `Qwen3-4B-Q4_K_M.gguf` and `Qwen3-1.7B-Q4_K_M.gguf` in
`/workspace/nupen/models`, a system `python3` with `pytest`, and cron. Without a `queue.txt`, the guardian falls back to traces on its
own.

## Stop

- `touch /root/guardian/STOP` stops everything and keeps it down. Delete the file to resume.
- `touch /root/guardian/STOP_FEED` stops only the traces feeder; `touch /root/guardian/STOP_CODER` stops only the coding item.
- On the PC, a `STOP` or `STOP_QUEUE` file in `creator_runtime/util3/guardian/` stops the sync loop or the queue driver.

## Measured on 3-4 Oct

- One llama-server is limited by its single server thread: 36 slots of a 4B reached only 44% GPU. Three to four small servers reach 98-99%.
- During fine-tunes, VRAM limits the filler to one or two servers, and the GPU ran at 71-77% before the reserve was set from the measured peak.
- Effladder rows produced at 1024 tokens per slot were invalid: think@2048 was capped at about 957 tokens. They were purged from `effladder.jsonl` and re-run at 4096 per slot.
