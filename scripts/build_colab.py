"""Build the self-contained Colab notebook from this checkout; no GPU required."""

import ast
import sys
import base64
import hashlib
import io
from pathlib import Path
import zipfile
import nbformat

ROOT = Path(__file__).resolve().parents[1]
EXTRAS_ONLY = "--extras-only" in sys.argv
EXCLUDED = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    "runs",
    "notebooks",
    ".venv",
    ".ipynb_checkpoints",
}


def source_files():
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(
            part in EXCLUDED or part.endswith(".egg-info") for part in relative.parts
        ):
            continue
        yield path, relative.as_posix()


buffer = io.BytesIO()
with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for path, relative in source_files():
        entry = zipfile.ZipInfo(relative, date_time=(2026, 10, 13, 0, 0, 0))
        entry.compress_type = zipfile.ZIP_DEFLATED
        entry.external_attr = (0o100644 << 16)
        archive.writestr(entry, path.read_bytes(), compresslevel=9)
payload = buffer.getvalue()
digest = hashlib.sha256(payload).hexdigest()
encoded = base64.b64encode(payload).decode()
encoded_lines = "\n".join(
    f"    {encoded[index : index + 120]!r}" for index in range(0, len(encoded), 120)
)

setup = (
    """#@title Check the A100 runtime and extract the complete project
import base64, hashlib, importlib.util, io, os, shutil, subprocess, sys, zipfile
from pathlib import Path
import torch
if not torch.cuda.is_available():
    raise RuntimeError("Attach the A100 GPU: Runtime > Change runtime type > A100 > Run all.")
major, minor = torch.cuda.get_device_capability()
if major < 8:
    raise RuntimeError("The BF16 full engine needs an Ampere-or-newer GPU. Use your A100 runtime.")
os.environ.setdefault("MAX_JOBS", "2")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
nvcc = shutil.which("nvcc")
if not nvcc:
    nvcc = next((str(path) for path in sorted(Path('/usr/local').glob('cuda*/bin/nvcc'), reverse=True)
                 if path.is_file()), None)
if nvcc:
    os.environ.setdefault("CUDA_HOME", str(Path(nvcc).parent.parent))
    os.environ['PATH'] = str(Path(nvcc).parent) + os.pathsep + os.environ['PATH']
missing = []
for module, requirement in [('ninja', 'ninja>=1.11'), ('pytest', 'pytest>=8'),
                            ('triton', 'triton==3.6.0')]:
    if importlib.util.find_spec(module) is None:
        missing.append(requirement)
if missing:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', *missing])
print('GPU:', torch.cuda.get_device_name())
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
import triton
print('Triton:', triton.__version__)
PAYLOAD = (
"""
    + encoded_lines
    + """
)
data = base64.b64decode(PAYLOAD)
assert hashlib.sha256(data).hexdigest() == """
    + repr(digest)
    + """
BASE = Path('/content') if Path('/content').is_dir() else Path.cwd()
PROJECT = BASE / 'cuda-moe-dispatch-v02'
if PROJECT.exists():
    # Preserve an earlier run if Run all is clicked again.
    import datetime
    PROJECT = BASE / ('cuda-moe-dispatch-v02-' + datetime.datetime.now().strftime('%H%M%S'))
PROJECT.mkdir(parents=True, exist_ok=False)
with zipfile.ZipFile(io.BytesIO(data)) as archive:
    for item in archive.infolist():
        destination = (PROJECT / item.filename).resolve()
        if not destination.is_relative_to(PROJECT.resolve()):
            raise RuntimeError('Invalid embedded archive path')
    archive.extractall(PROJECT)
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(PROJECT), '--no-deps'])
print('Complete repository extracted:', PROJECT)
print('Setup complete. Keep this PyTorch install; the next cell compiles and validates the engine.')
"""
)

run = """#@title Compile, test, benchmark, show the report, and download everything
# Defaults finish the complete small/medium comparison first. Extras are switches.
RUN_CORE = True            # Reproduce CPU/GPU checks and smoke/quick eager results.
RUN_MODEL_SHAPED = False  # H=4096, I=14336, E=8; synthetic weights use several GB.
RUN_CUDA_GRAPHS = False   # Optional replay experiment; reported separately from eager forward.
RUN_PROFILE = True
EXPERT_BACKEND = 'triton'
from IPython.display import HTML, FileLink, display
import json, statistics, traceback
RUNS = PROJECT / 'runs'
RUNS.mkdir(exist_ok=True)
LOG = RUNS / 'run.log'


def run_command(*arguments):
    command = [sys.executable, '-u', *arguments]
    print('Running:', ' '.join(command), flush=True)
    with LOG.open('a') as log:
        log.write('\\nCOMMAND: ' + ' '.join(command) + '\\n')
        process = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        raise RuntimeError(f'Command failed with exit {code}. Complete output is in runs/run.log.')


failure = None
try:
    run_command('-m', 'moe_dispatch', 'check-env')
    if RUN_CORE:
        run_command('-m', 'pytest', '-q', '-m', 'not gpu')
    if RUN_CORE:
        run_command('-m', 'pytest', '-q', '-m', 'gpu')
    if RUN_CORE:
        run_command('-m', 'moe_dispatch', 'validate')
    for suite in (('smoke', 'quick') if RUN_CORE else ()):
        run_command('-m', 'moe_dispatch', 'block-benchmark', '--suite', suite,
                    '--expert-backend', EXPERT_BACKEND, '--output', f'runs/block-{suite}')
    for enabled, suite, destination, flags in (
        (RUN_CUDA_GRAPHS, 'quick', 'runs/block-graphs', ('--cuda-graphs',)),
        (RUN_MODEL_SHAPED, 'model', 'runs/model-shaped', ()),
    ):
        if enabled:
            try:
                run_command('-m', 'moe_dispatch', 'block-benchmark', '--suite', suite,
                            '--expert-backend', EXPERT_BACKEND, '--output', destination, *flags)
                display(HTML((PROJECT / destination / 'block_report.html').read_text()))
            except RuntimeError as error:
                print('Optional experiment failed:', destination, error)
                (RUNS / (Path(destination).name + '_error.txt')).write_text(str(error))
    if RUN_PROFILE:
        try:
            run_command('-m', 'moe_dispatch', 'profile-block', '--expert-backend', EXPERT_BACKEND,
                        '--output', 'runs/moe_block_trace.json')
        except RuntimeError as error:
            print('Optional profiler unavailable:', error)
            (RUNS / 'profiler_error.txt').write_text(str(error))
    result_path = RUNS / 'block-quick/block_results.json'
    if not result_path.exists():
        result_path = PROJECT / 'benchmarks/a100-block/block_results.json'
    result = json.loads(result_path.read_text())
    forwards = {(row['case_id'], row['implementation']): row['median_us']
                for row in result['benchmarks'] if row['stage'] == 'forward'}
    ratios = [forwards[case['case_id'], 'torch'] / forwards[case['case_id'], 'cuda_fused']
              for case in result['configurations']]
    print('Validated full-forward configurations:', len(ratios))
    print('Actual median full-forward ratio:', round(statistics.median(ratios), 3), 'x')
    print('All case ratios:', [round(value, 3) for value in ratios])
    print('Includes router, dispatch, grouped experts, SwiGLU, and combine.')
    display(HTML(result_path.with_name('block_report.html').read_text()))
except Exception as error:
    failure = error
    details = traceback.format_exc()
    print(details)
    (RUNS / 'failure.txt').write_text(details)
finally:
    # A failure still exports source, environment output, and complete diagnostic logs.
    archive_path = PROJECT.parent / 'cuda_moe_engine_gpu_bundle.zip'
    excluded = {'__pycache__', '.pytest_cache', '.ruff_cache', '.git', 'build', 'dist'}
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(PROJECT.rglob('*')):
            relative = path.relative_to(PROJECT)
            if path.is_file() and not any(part in excluded or part.endswith('.egg-info')
                                          for part in relative.parts):
                archive.write(path, Path('cuda-moe-dispatch') / relative)
    print('Download this ZIP and send it back for review:', archive_path)
    display(FileLink(str(archive_path)))
    try:
        from google.colab import files
        files.download(str(archive_path))
    except ImportError:
        pass
if failure:
    raise RuntimeError('GPU validation stopped. Downloaded ZIP includes the complete error; send it back.') from failure
"""

intro = """# CUDA MoE Engine — Full Forward Upgrade

Use the **same A100 Colab runtime** as your successful first run. Upload this notebook,
select **Runtime → Run all**, and download `cuda_moe_engine_gpu_bundle.zip` when it finishes.
No repository, model, or dataset download is required. The notebook embeds the complete project.

This expands the transport experiment into a forward-only, single-GPU MoE block:
**FP32 router → fused bias/top-k/softmax → aligned CUDA dispatch → grouped BF16 gate/up
matmul → fused SwiGLU → grouped BF16 down matmul → FP32 weighted combine**.

The three variants share weights, precision, and the grouped-matmul backend. The generated
HTML report compares complete forward latency and isolated stages across eight configurations,
including balanced and skewed expert loads. Optional larger synthetic model shapes and CUDA
Graph replay are off until the core run succeeds; a profiler trace is enabled.

**Recorded A100 checkpoint:** 81 CPU tests, 39 GPU tests, and 36 transport checks passed.
All 24 quick-suite full-forward comparisons passed; the median eager forward ratio was 3.17×
against the included PyTorch variant with the same Triton expert backend.
The archived 6.36× transport result is included separately and is not a full-block result.

Keep the existing PyTorch install. The setup installs only missing Ninja, pytest, and Triton.
Use **Save a copy in Drive** to keep the executed notebook as well as downloading the ZIP.
"""

if EXTRAS_ONLY:
    run = run.replace("RUN_CORE = True", "RUN_CORE = False")
    run = run.replace("RUN_MODEL_SHAPED = False", "RUN_MODEL_SHAPED = True")
    run = run.replace("RUN_CUDA_GRAPHS = False", "RUN_CUDA_GRAPHS = True")
    run = run.replace("RUN_PROFILE = True", "RUN_PROFILE = False")
    intro = """# CUDA MoE — Optional A100 Experiments

Run all in your A100 runtime. This notebook embeds the validated v0.2.0 source and evidence.
It runs **quick-suite CUDA Graph comparisons first**, then the **larger synthetic model-shaped
preset** (H=4096, I=14336, E=8), and exports a ZIP containing every result and diagnostic.
The core 81 CPU / 39 GPU tests and eager quick suite were already passed in the recorded run
and are skipped here. Optional failures are recorded independently so one does not hide the other.
Graph capture may be unsupported for some variants. Only successful replay measurements count.
Recorded results: 24 graph replay checks passed (1.74× median graph-to-graph ratio),
and 6 larger-preset oracle comparisons passed (1.06× median eager ratio).
The original 3.17× eager result remains the primary small/medium-block claim.
"""

notebook = nbformat.v4.new_notebook(
    cells=[
        nbformat.v4.new_markdown_cell(intro),
        nbformat.v4.new_code_cell(
            setup, metadata={"cellView": "form", "jupyter": {"source_hidden": True}}
        ),
        nbformat.v4.new_markdown_cell("""## Run the full engine

The first build compiles CUDA sources and the grouped-matmul kernel. Compilation and validation
are excluded from benchmark timings. Correctness failures stop the comparison and export an error
bundle. Every speed ratio reports the measured included baseline; values below 1 mean a slowdown.
"""),
        nbformat.v4.new_code_cell(run),
        nbformat.v4.new_markdown_cell("""## After the run

- Open `runs/block-quick/block_report.html` locally for the portable demo.
- Raw samples, hardware, validation errors, and source hashes are in `block_results.json`.
- Open `runs/moe_block_trace.json` in a Chrome trace viewer for stage and kernel inspection.
- Send back the downloaded ZIP so the new GPU checks and full-forward measurements can be reviewed.

The project is synthetic inference engineering, with no training, multi-GPU communication,
pretrained-model evaluation, or production-library speed claim. Explicit persistent buffers are
reported separately from weights and backend temporary allocations. CUDA-event eager timings can
include CPU launch gaps. Stage medians are independent measurements and do not add up to forward.

[Grouped GEMM tutorial](https://triton-lang.org/main/getting-started/tutorials/08-grouped-gemm.html) ·
[PyTorch CUDA extensions](https://docs.pytorch.org/docs/stable/cpp_extension.html)
"""),
    ]
)
notebook.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "accelerator": "GPU",
    "colab": {"name": "CUDA_MoE_Optional_A100.ipynb" if EXTRAS_ONLY else "Colab_Upgrade.ipynb", "provenance": []},
}
for index, cell in enumerate(notebook.cells):
    cell.id = hashlib.sha256((str(index) + cell.source).encode()).hexdigest()[:16]
nbformat.validate(notebook)
for cell in notebook.cells:
    if cell.cell_type == "code":
        ast.parse(cell.source)
(ROOT / "notebooks").mkdir(exist_ok=True)
filename = "CUDA_MoE_Optional_A100.ipynb" if EXTRAS_ONLY else "Colab_Upgrade.ipynb"
nbformat.write(notebook, ROOT / "notebooks" / filename)
nbformat.write(notebook, ROOT.parent / filename)
print(
    "Notebook built; embedded source files:",
    len(list(source_files())),
    "payload bytes:",
    len(payload),
    "sha256:",
    digest,
)
print(ROOT.parent / filename)
