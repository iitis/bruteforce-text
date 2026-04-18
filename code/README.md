# Running The Bruteforce Sampler

This folder now contains a single benchmark entrypoint: `bf.py`.

## 1) Create and activate the Conda environment

From repository root (`bruteforce-text`):

```bash
conda env create -f code/environment.yml
conda activate omnisolver-bruteforce-bench
```

## 2) Install local Omnisolver packages

This project expects sibling repositories:
- `../omnisolver`
- `../omnisolver-bruteforce`

Install them in editable mode:

```bash
python -m pip install -e ../omnisolver
python -m pip install -e ../omnisolver-bruteforce
```

## 3) (Distributed mode only) start Ray on local node

```bash
ray stop --force
ray start --head --disable-usage-stats
```

## 4) Run benchmark script

### Distributed sampler (default)

```bash
python code/bf.py --start 60 --stop 40 --step -2 --sampler-mode distributed --skip-existing
```

### Single-GPU sampler (no Ray required)

```bash
python code/bf.py --start 60 --stop 40 --step -2 --sampler-mode single-gpu --skip-existing
```

## Output layout

- Generated instances: `instances/<N>.txt`
- Solver results: `results/<N>.json`

Where `<N>` is the problem size from your `start/stop/step` sweep.
