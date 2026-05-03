# Phase 4: Data Augmentation and GPU Training via ClearML Agent - Research

**Researched:** 2026-05-03
**Domain:** PyTorch augmentation (no torchvision), ClearML agent deployment, WSL2+CUDA for Blackwell GPU
**Confidence:** MEDIUM — SDK patterns verified against official docs; RTX 5060 specifics verified via PyTorch forums; agent config verified via official clearml.conf example

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Apply three transforms: rotation ±5-10°, brightness/contrast jitter, Gaussian noise/blur. No horizontal flip (reverses RTL text), no elastic distortion.
- **D-02:** Conservative parameters. Expose all augmentation parameters as CLI flags (rotation_max, brightness_delta, noise_sigma).
- **D-03:** Online augmentation — transforms inside `CropDataset.__getitem__`. No new files written.
- **D-04:** Augmentation applies to training split only. Val split always receives clean crops.
- **D-05:** `--aug_copies` flag (default: Claude's discretion). When 0 or absent, augmentation is off.
- **D-06:** ClearML agent runs in WSL2 with CUDA on the Windows RTX 5060 host.
- **D-07:** `train_ctc.py` gets `--enqueue` flag. When set, creates task, enqueues to named queue, exits.
- **D-08:** Task name `train_baseline_ctc` unchanged. Differentiated by tags (`"gpu"`) and hyperparams.
- **D-09:** Add `--dataset_id` arg. When provided, use `Dataset.get(dataset_id=...).get_local_copy()`.
- **D-10:** Remap crop_path/page_path in-memory: `dataset_root / "crops" / Path(original).name`.
- **D-11:** ClearML SDK caches datasets by dataset ID; second call with same ID uses cached copy.

### Claude's Discretion

- Default `--aug_copies` value (suggested: 2)
- Exact conservative parameter defaults for rotation, brightness, noise (suggested: ±7°, ±10%, sigma=5)
- Whether augmentation is a standalone `AugmentTransform` class or inline in `__getitem__`
- ClearML queue name (suggested: `"gpu"`)

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.

</user_constraints>

---

## Summary

Phase 4 has two independent workstreams. The augmentation workstream is purely local Python code: three transforms (rotation, brightness jitter, Gaussian noise) implemented with pure PyTorch and NumPy — no torchvision required. The GPU training workstream requires installing `clearml-agent` inside WSL2, pointing it at a `"gpu"` queue, and adding `--enqueue` / `--dataset_id` flags to `train_ctc.py`.

The most significant finding is the **RTX 5060 (Blackwell sm_120) PyTorch requirement**: PyTorch 2.7+ with `--index-url https://download.pytorch.org/whl/cu128` is mandatory. The project currently pins `torch==2.11.0` (CPU wheel), which already satisfies this since 2.11 > 2.7, but the agent's venv must use the CUDA wheel index, not the CPU wheel index.

The ClearML agent's `execute_remotely()` pattern is well-understood: call `Task.init()`, `task.connect(args)`, then `task.execute_remotely(queue_name="gpu")` when `--enqueue` is set. The call exits the local process; the agent re-runs the full script but treats `execute_remotely` as a no-op.

**Primary recommendation:** Implement augmentation as a standalone `AugmentTransform` class (not inline in `__getitem__`) for testability. Use pure PyTorch (`F.affine_grid`, `torch.randn`) — no new dependencies.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch (CPU) | 2.11.0 (pinned) | Dev/local training | Already in pyproject.toml |
| torch (CUDA) | 2.11.0 cu128 | Agent-side GPU training | RTX 5060 requires CUDA 12.8 wheel; 2.11 >= 2.7 threshold |
| clearml-agent | 3.0.0 | Queue-based remote execution | Official ClearML orchestration tool |
| clearml (SDK) | 2.1.5 (pinned) | Dataset.get(), execute_remotely | Already in pyproject.toml |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | 2.4.4 (pinned) | Augmentation helper arrays | Already available; used in load_crop path |
| cv2 | 4.13.0 (pinned) | Gaussian blur on numpy arrays | Already available; cv2.GaussianBlur is clean |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Pure PyTorch affine_grid rotation | cv2.warpAffine | cv2 is already imported in ctc_utils.py; either works. Pure PyTorch stays in tensor domain without numpy round-trip |
| clearml-agent venv mode | clearml-agent Docker mode | Docker mode requires Docker Desktop on Windows; venv mode works natively in WSL2 |

**Installation (agent-side WSL2 only — NOT added to pyproject.toml):**
```bash
pip install clearml-agent==3.0.0
```

**PyTorch CUDA wheel (agent venv, configured in clearml.conf extra_index_url):**
```
https://download.pytorch.org/whl/cu128
```

**Version verification:** clearml-agent 3.0.0 verified from PyPI 2026-05-03. torch 2.11.0 verified from PyPI 2026-05-03.

---

## Architecture Patterns

### Recommended Project Structure (changes only)
```
src/
├── ctc_utils.py         # Add AugmentTransform class here
└── train_ctc.py         # Add --enqueue, --dataset_id, --aug_copies flags

.planning/phases/04-.../
└── 04-RESEARCH.md       # this file
```

No new files required. All augmentation lives in `ctc_utils.py`. All ClearML agent SDK calls live in `train_ctc.py::main()`.

---

### Pattern 1: AugmentTransform class in ctc_utils.py

**What:** Standalone callable that takes a `(1, H, W)` float32 tensor and returns an augmented tensor of the same shape. Seeded per (dataset_index, epoch_step) for reproducibility.

**When to use:** Applied inside `CropDataset.__getitem__` when `self._augment` is set.

```python
# Source: PyTorch docs + search-verified pure-PyTorch patterns
import torch
import torch.nn.functional as F
import cv2
import numpy as np

class AugmentTransform:
    def __init__(
        self,
        rotation_max: float = 7.0,
        brightness_delta: float = 0.10,
        noise_sigma: float = 0.02,
    ) -> None:
        self.rotation_max = rotation_max
        self.brightness_delta = brightness_delta
        self.noise_sigma = noise_sigma

    def __call__(self, tensor: torch.Tensor, seed: int) -> torch.Tensor:
        rng = torch.Generator()
        rng.manual_seed(seed)

        # 1. Rotation via affine_grid (pure PyTorch, no import needed beyond F)
        angle_deg = torch.empty(1).uniform_(-self.rotation_max, self.rotation_max).item()
        angle_rad = angle_deg * math.pi / 180.0
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        theta = torch.tensor(
            [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]], dtype=torch.float32
        ).unsqueeze(0)
        grid = F.affine_grid(theta, tensor.unsqueeze(0).shape, align_corners=False)
        tensor = F.grid_sample(
            tensor.unsqueeze(0), grid, align_corners=False, padding_mode="border"
        ).squeeze(0)

        # 2. Brightness/contrast jitter (multiplicative)
        factor = 1.0 + torch.empty(1, generator=rng).uniform_(
            -self.brightness_delta, self.brightness_delta
        ).item()
        tensor = torch.clamp(tensor * factor, 0.0, 1.0)

        # 3. Gaussian noise
        noise = torch.randn(tensor.shape, generator=rng) * self.noise_sigma
        tensor = torch.clamp(tensor + noise, 0.0, 1.0)

        return tensor
```

**Notes on brightness vs contrast:** Multiplicative scaling approximates both brightness and contrast shift simultaneously. A separate contrast term (subtract mean, scale, add mean) can be added but is not needed given D-02 conservative scope.

**Notes on Gaussian blur:** `cv2.GaussianBlur` operates on numpy uint8 or float32 arrays. Apply before converting to tensor: insert in `load_crop()` or apply via numpy conversion inside `AugmentTransform`. Recommended approach: apply blur on the numpy array inside `load_crop()` when a blur sigma parameter is supplied, to keep the blur in the image domain and avoid tensor-to-numpy round-trips.

---

### Pattern 2: CropDataset aug_copies integration

**What:** When `aug_copies > 0`, `__len__` is multiplied and `__getitem__` maps logical index to (physical_index, copy_index). Only train dataset gets the transform; val dataset has `augment=None`.

```python
class CropDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        charset: list[str],
        augment: AugmentTransform | None = None,
        aug_copies: int = 0,
    ) -> None:
        self._df = df
        self._charset = charset
        self._augment = augment
        self._copies = aug_copies if augment is not None else 0

    def __len__(self) -> int:
        return len(self._df) * (1 + self._copies)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, list[int]]:
        real_idx = index % len(self._df)
        copy_idx = index // len(self._df)
        row = self._df.iloc[real_idx]
        image = load_crop(str(row["crop_path"]))
        if copy_idx > 0 and self._augment is not None:
            seed = index  # deterministic per logical index
            image = self._augment(image, seed)
        label_ids = encode_label(str(row["label"]), self._charset)
        return image, label_ids
```

---

### Pattern 3: --enqueue flow in train_ctc.py

**What:** After `init_task()` and `task.connect(args)`, if `--enqueue` is set, call `task.execute_remotely()` which exits the local process. The agent re-runs the full script and skips the `execute_remotely` call as a no-op.

**Critical ordering:** `task.connect(args)` MUST happen before `task.execute_remotely()`. This ensures hyperparameters are recorded in the task before it is handed to the agent. The agent will read those recorded hyperparameters back and override args at execution time.

```python
# Source: ClearML official execute_remotely example + docs
task = init_task("handwriting-hebrew-ocr", "train_baseline_ctc", tags=["phase-4"])

# CRITICAL: connect args BEFORE execute_remotely
task.connect(vars(args), name="hyperparams")

if args.enqueue:
    task.execute_remotely(queue_name=args.queue_name)
    # local process exits here; agent re-runs from top

# ... rest of training code runs only locally or on the agent
```

---

### Pattern 4: --dataset_id path remapping

**What:** `Dataset.get(dataset_id=...).get_local_copy()` returns an absolute string path to the cached dataset root. Manifest crop_path/page_path columns are remapped in-memory.

```python
# Source: ClearML Dataset SDK reference
from clearml import Dataset

def remap_dataset_paths(df: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    ds = Dataset.get(dataset_id=dataset_id)
    root = Path(ds.get_local_copy())  # e.g., ~/.clearml/cache/datasets/abc123/
    df = df.copy()
    df["crop_path"] = df["crop_path"].apply(
        lambda p: str(root / "crops" / Path(p).name)
    )
    df["page_path"] = df["page_path"].apply(
        lambda p: str(root / "pages" / Path(p).name)
    )
    return df
```

`get_local_copy()` is documented as read-only/immutable. Re-calling with same dataset_id uses the cached copy (confirmed by search results: "if the same `dataset_id` is requested again, ClearML will reuse the existing cached copy rather than re-downloading").

**Cache location:** `~/.clearml/cache/` by default; overridable with `CLEARML_CACHE_DIR` env var.

---

### Anti-Patterns to Avoid

- **Applying augmentation to val set:** D-04 is explicit. Val CropDataset must receive `augment=None`.
- **Calling execute_remotely before task.connect:** The agent won't see hyperparameters — they'll be blank on the remote side.
- **Using torchvision transforms:** The project CLAUDE.md prohibits additional heavy dependencies. Pure PyTorch + cv2 (already available) is sufficient.
- **Installing CUDA toolkit inside WSL2:** The Windows host NVIDIA driver stubs `libcuda.so` into WSL2. Running `apt install cuda-toolkit` can overwrite this stub and break CUDA access.
- **Horizontal flip augmentation:** D-01 prohibits it — reverses Hebrew RTL text.
- **Writing augmented images to disk:** D-03 requires online-only augmentation.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Affine rotation | Custom pixel loop | `F.affine_grid` + `F.grid_sample` | GPU-compatible, handles interpolation and boundary correctly |
| Gaussian blur | Convolution kernel | `cv2.GaussianBlur` on numpy array | Handles kernel size/sigma correctly, already available |
| Dataset download + caching | Custom downloader | `Dataset.get(...).get_local_copy()` | ClearML cache handles dedup, partial downloads, locking |
| Task enqueue | Custom HTTP call to ClearML API | `task.execute_remotely()` | SDK handles auth, serialization, queue routing |
| PyTorch CUDA wheel resolution | Hard-code wheel URL | ClearML agent's automatic torch resolver | Detects CUDA version, picks correct index URL automatically |

---

## Common Pitfalls

### Pitfall 1: RTX 5060 requires PyTorch CUDA 12.8 wheel (sm_120)
**What goes wrong:** Installing any torch wheel before PyTorch 2.7 (or a cu121/cu118 wheel) fails with `no kernel image is available for execution on the device` at runtime. The local `pyproject.toml` pins `torch==2.11.0` from the CPU index — this is correct for dev but the agent venv needs the CUDA variant.
**Why it happens:** RTX 5060 is Blackwell architecture (compute capability sm_120). Support for sm_120 was added in PyTorch 2.7 stable with CUDA 12.8 build.
**How to avoid:** In `clearml.conf` on the WSL2 host, set `package_manager.extra_index_url: ["https://download.pytorch.org/whl/cu128"]`. The agent's torch resolver will use this index when installing torch for the task venv.
**Warning signs:** `torch.cuda.is_available()` returns `False` on the agent even when `nvidia-smi` works. Or explicit `RuntimeError: no kernel image`.

### Pitfall 2: CUDA toolkit install clobbers WSL2 driver stub
**What goes wrong:** Running `apt install nvidia-cuda-toolkit` inside WSL2 replaces the driver stub injected by Windows NVIDIA driver with a full toolkit that may not match the host driver version. CUDA becomes unavailable.
**Why it happens:** WSL2 injects `libcuda.so` via Windows driver; the apt toolkit brings its own stub.
**How to avoid:** Install CUDA toolkit from NVIDIA's WSL-specific repository (`cuda-toolkit-X-Y` from `developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/...`), not from Ubuntu's default apt repos. Or skip the toolkit entirely — the Windows driver provides what's needed for PyTorch.
**Warning signs:** `nvidia-smi` works but Python `torch.cuda.is_available()` returns False.

### Pitfall 3: execute_remotely and the --enqueue flag
**What goes wrong:** If `parse_known_args` pre-flight check (established Phase 2 pattern) runs before `Task.init()`, and `--enqueue` triggers early, the task may not have been fully initialized before `execute_remotely` is called.
**Why it happens:** The pre-flight pattern exists to catch missing manifest path before spawning a ClearML task. The `--enqueue` flag should not trigger before `init_task()`.
**How to avoid:** Pre-flight check only validates `--manifest` existence (or skip if `--dataset_id` provided). `--enqueue` handling must happen AFTER `task.connect(vars(args))`.

### Pitfall 4: aug_copies doubles epoch time silently
**What goes wrong:** With `--aug_copies 2`, the DataLoader sees 3x the rows. Training time triples. With large `--epochs`, this compounds.
**Why it happens:** `__len__` returns `len(df) * (1 + copies)`. The shuffle in DataLoader treats augmented copies as independent samples.
**How to avoid:** Print a warning when `aug_copies > 0`: `"augmentation: {aug_copies} copies, effective dataset size: {effective_n}"`. Log effective size as a ClearML hyperparam.

### Pitfall 5: Rotation padding mode affects blank-padded images
**What goes wrong:** When rotating a right-padded image (padded to batch width with zeros), the rotation can drag blank regions into the actual content area, making more zeros appear on the left of a short crop.
**Why it happens:** Default `padding_mode="zeros"` fills rotated-off-canvas with 0 which is indistinguishable from actual padding.
**How to avoid:** Use `padding_mode="border"` in `F.grid_sample` — replicates edge pixels instead. Or apply augmentation before the collate step (which is correct — `__getitem__` runs before collate).

### Pitfall 6: ClearML agent requires code to be committed to git
**What goes wrong:** The agent clones the git repo at the commit captured when `Task.init()` ran. Uncommitted changes are captured as a diff and re-applied. If the diff is large or contains binary files, this can fail silently.
**Why it happens:** ClearML captures git diff at task init time and stores it as task metadata. The agent replays it.
**How to avoid:** Commit Phase 4 changes before running `--enqueue` for the first time. This is the standard workflow.

---

## Code Examples

### ClearML Agent installation and config (WSL2)

```bash
# In WSL2 Ubuntu shell — install agent as system package (not in venv)
pip install clearml-agent==3.0.0

# Run setup wizard — will prompt for API credentials, git credentials, extra index URL
clearml-agent init
```

Minimal additions to `~/clearml.conf` agent section after `clearml-agent init`:
```hocon
agent {
    git_user: ""          # leave blank to use SSH
    git_pass: ""
    worker_id: "wsl2-gpu-worker"

    package_manager {
        type: pip
        # GPU torch wheel index for CUDA 12.8 (RTX 5060 / Blackwell)
        extra_index_url: ["https://download.pytorch.org/whl/cu128"]
        system_site_packages: false
    }
}
```

Start the agent (venv mode, no Docker required):
```bash
clearml-agent daemon --queue gpu --gpus 0 --foreground
```

For detached background operation:
```bash
clearml-agent daemon --detached --queue gpu --gpus 0
```

---

### train_ctc.py: new flags skeleton

```python
# Add to _build_parser():
p.add_argument("--enqueue", action="store_true", default=False,
               help="Enqueue task to ClearML queue instead of running locally")
p.add_argument("--queue_name", type=str, default="gpu",
               help="ClearML queue name for --enqueue mode")
p.add_argument("--dataset_id", type=str, default=None,
               help="ClearML dataset ID; downloads and caches dataset locally")
p.add_argument("--aug_copies", type=int, default=0,
               help="Augmented copies per original crop (0 = disabled)")
p.add_argument("--rotation_max", type=float, default=7.0)
p.add_argument("--brightness_delta", type=float, default=0.10)
p.add_argument("--noise_sigma", type=float, default=0.02)

# In main(), after task.connect(vars(args)):
if args.enqueue:
    task.execute_remotely(queue_name=args.queue_name)
    # process exits here when running locally

# Dataset loading with --dataset_id:
if args.dataset_id:
    df = remap_dataset_paths(df, args.dataset_id)
```

---

### Queue creation (one-time, via ClearML Web UI or SDK)

Queues are created on the ClearML server side (app.clear.ml). No SDK call needed to create a queue before the agent starts — the agent implicitly creates the queue on first `clearml-agent daemon --queue gpu` if it doesn't exist. Alternatively create via UI: Settings > Workers & Queues > New Queue.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| torchvision transforms | Pure PyTorch F.affine_grid + torch.randn | Always possible, just under-documented | No new deps needed |
| PyTorch pre-2.7 (sm_90 max) | PyTorch 2.7+ cu128 (sm_120 Blackwell) | PyTorch 2.7 release, 2025 | RTX 5060 requires this |
| clearml-agent 1.x/2.x | clearml-agent 3.0.0 | 2025 | Latest stable; verify no breaking API changes vs 2.x |

**Deprecated/outdated:**
- PyTorch CPU-only wheels on the agent host: must use CUDA wheel for RTX 5060. pyproject.toml keeps CPU wheel for dev — two separate wheel sources is the correct pattern.

---

## Open Questions

1. **clearml-agent 3.0.0 breaking changes vs 2.x**
   - What we know: clearml-agent 3.0.0 is current PyPI release
   - What's unclear: Whether `clearml==2.1.5` (pinned in pyproject.toml) is compatible with `clearml-agent==3.0.0`. These are separate packages; the SDK version and agent version don't need to match exactly.
   - Recommendation: During agent setup, verify `clearml-agent daemon --version` and test with a trivial enqueued task before running real training. If incompatible, pin agent to the version that matches clearml SDK 2.x.

2. **PyTorch on Windows (non-WSL) sm_120 status**
   - What we know: PyTorch 2.7 stable supports sm_120 on Linux x86. Forum posts indicate Windows support was "ongoing concern" as of early 2025.
   - What's unclear: Whether PyTorch 2.11 cu128 Windows wheel supports sm_120.
   - Recommendation: The plan targets WSL2 (Linux), not native Windows. WSL2 support is confirmed. No action needed.

3. **Dataset structure inside get_local_copy() root**
   - What we know: `get_local_copy()` returns a base folder path. The path remapping in D-10 assumes `root/crops/` and `root/pages/` sub-directories.
   - What's unclear: The exact directory layout inside the cached copy depends on how `maybe_create_dataset` added files — specifically whether folder structure is preserved.
   - Recommendation: The existing `maybe_create_dataset` calls `ds.add_files(str(folder))` for each folder. ClearML preserves the relative folder structure. If the dataset was created with `add_files("data/crops")`, the local copy will have `crops/` at the root. The remapping pattern `root / "crops" / Path(p).name` should be correct, but must be verified against the actual dataset layout during Wave 0 or a smoke test.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.13 (local dev) | pyproject.toml | ✓ | 3.13 (inferred from .python-version) | — |
| torch (CPU) | local training | ✓ | 2.11.0 | — |
| clearml SDK | Dataset.get(), Task | ✓ | 2.1.5 | — |
| cv2 | load_crop, Gaussian blur | ✓ | 4.13.0 | — |
| numpy | tensor conversions | ✓ | 2.4.4 | — |
| WSL2 (Windows host) | GPU agent | Unknown | — | Must be set up by user |
| NVIDIA driver (Windows) | WSL2 CUDA | Unknown | — | Must be installed (GeForce driver >= 570 for sm_120) |
| clearml-agent | GPU queue | ✗ (not on dev machine) | 3.0.0 on PyPI | Install in WSL2 |
| Python 3.13 (WSL2) | Agent venv | Unknown | — | Install via uv or pyenv in WSL2 |
| torch cu128 wheel | RTX 5060 training | ✗ (only CPU wheel) | 2.11.0 cu128 on PyPI | Not applicable — required |

**Missing dependencies with no fallback:**
- Windows NVIDIA GeForce driver (>= 576.x / Game Ready supporting CUDA 12.8) on the host machine — required for WSL2 CUDA.
- clearml-agent in WSL2 — core requirement for D-07.

**Missing dependencies with fallback:**
- WSL2 Python 3.13: if unavailable, clearml-agent can use Python 3.11/3.12 for its own venv (agent Python != task Python). The task venv is created fresh.

---

## Sources

### Primary (HIGH confidence)
- [ClearML Dataset SDK Reference](https://clear.ml/docs/latest/docs/references/sdk/dataset/) — `Dataset.get()` and `get_local_copy()` signatures
- [ClearML Remote Execution Guide](https://clear.ml/docs/latest/docs/guides/advanced/execute_remotely/) — `execute_remotely()` pattern
- [clearml/clearml-agent clearml.conf example](https://github.com/clearml/clearml-agent/blob/master/docs/clearml.conf) — package_manager configuration
- [ClearML Execution Environments](https://clear.ml/docs/latest/docs/clearml_agent/clearml_agent_execution_env/) — venv mode execution flow
- [PyTorch Forums: sm_120 RTX 5060 support](https://discuss.pytorch.org/t/pytorch-support-for-sm_120-nvidia-geforce-rtx-5060/220941) — confirmed PyTorch 2.7+ cu128 required
- [PyTorch 2.7 Release Blog](https://pytorch.org/blog/pytorch-2-7/) — official Blackwell sm_120 support announcement

### Secondary (MEDIUM confidence)
- [ClearML Workers and Queues](https://clear.ml/docs/latest/docs/fundamentals/agents_and_queues/) — queue configuration and agent service modes
- [NVIDIA WSL2 CUDA Guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html) — don't install separate Linux CUDA driver inside WSL2
- [ClearML Agent bare metal deployment](https://clear.ml/docs/latest/docs/clearml_agent/clearml_agent_deployment_bare_metal/) — venv mode steps

### Tertiary (LOW confidence)
- [GitHub: RTX 5060-wsl2-workspace](https://github.com/kennito2035/rtx5060-wsl2-workspace) — community verified CUDA 12.6/PyTorch 2.7 stack on RTX 5060 WSL2; useful sanity check but not official

---

## Metadata

**Confidence breakdown:**
- Augmentation patterns: HIGH — pure PyTorch F.affine_grid is standard; code verified against official PyTorch docs
- ClearML execute_remotely flow: HIGH — official docs and GitHub example confirm ordering requirement
- Dataset caching behavior: MEDIUM — documented as read-only/cached but caching skip-on-hit behavior inferred from search results, not direct API doc statement
- clearml-agent config format: MEDIUM — verified via clearml.conf example in GitHub repo
- RTX 5060 / sm_120 requirement: HIGH — confirmed by PyTorch 2.7 release blog and forum threads
- clearml-agent 3.0.0 / SDK 2.1.5 compatibility: LOW — not directly verified; recommend smoke-test before relying on it

**Research date:** 2026-05-03
**Valid until:** 2026-08-03 (stable ecosystem; ClearML SDK versions change slowly; PyTorch CUDA support won't regress)
