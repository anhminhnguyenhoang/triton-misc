#!/usr/bin/env python3
"""Inject MHC tuner winners into the production per-C JSON config files.

Reads the latest `best_configs_mhc_fused_mhc_<hres>_M*_n*_C*.json` files emitted
by `tune_mhc.py` (one file per C bucket) and merges each `(M, C)` winner into
`aiter/ops/triton/configs/{arch}-MHC_FUSED_<HRES>-C={C}.json` under the matching
`M_LEQ_<M>` bucket. The catch-all `"any"` entry is also refreshed with the
largest-M winner.

Usage (from /home/anguyenh/aiter):
    python tuning_results/refresh_mhc_configs.py                 # default: gfx942 + gfx950, sinkhorn
    python tuning_results/refresh_mhc_configs.py --arches gfx950
    python tuning_results/refresh_mhc_configs.py --hres-mode lite
    python tuning_results/refresh_mhc_configs.py --tuner-glob 'best_configs_mhc_*_C*.json'
    python tuning_results/refresh_mhc_configs.py --dry-run

Notes:
- M buckets below the smallest tuned M (e.g., M_LEQ_1, M_LEQ_64) are preserved
  unchanged - the tuner only sweeps M >= 1024.
- `BLOCK_N` is dropped from the merged config because the wrapper always derives
  it from `n_squared` at launch time.
- Split-C block sizes (`BLOCK_M_SPLITC`, `BLOCK_C_SPLITC`) are kept only when
  `USE_REDUCE_SPLITC=True`; the tuner already emits `None` for the inline path
  and we strip those.
"""
import argparse
import glob
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "aiter" / "ops" / "triton" / "configs"
DEFAULT_TUNER_GLOB = "best_configs_mhc_fused_mhc_{hres}_M*_n*_C*.json"


def detect_active_arch() -> str | None:
    """Return the active GPU arch (e.g., 'gfx950'), or None if it can't be determined.

    The default is intentionally limited to the arch the tuner ran on; writing
    cross-arch is opt-in via `--arches` to avoid silently clobbering configs
    tuned on other hardware.
    """
    try:
        from aiter.ops.triton.utils._triton import arch_info
        return arch_info.get_arch()
    except Exception:
        return None


def collect_winners(tuner_files):
    """Return {C: {M: (config, time_ms, src_path)}} from tuner outputs."""
    by_c = {}
    for path in tuner_files:
        data = json.loads(Path(path).read_text())
        for key, entry in data.items():
            m = re.match(r"M(\d+)_n\d+_C(\d+)_\w+", key)
            if not m:
                continue
            M = int(m.group(1))
            C = int(m.group(2))
            cfg = dict(entry["config"])
            time_ms = float(entry["time_ms"])
            prev = by_c.setdefault(C, {}).get(M)
            # If multiple files target the same (C, M), keep the fastest.
            if prev is None or time_ms < prev[1]:
                by_c[C][M] = (cfg, time_ms, path)
    return by_c


def clean_config(cfg: dict) -> dict:
    """Drop `BLOCK_N` and any `None`-valued split-C block entries."""
    out = {k: v for k, v in cfg.items() if v is not None}
    out.pop("BLOCK_N", None)
    return out


def merge_into_json(c: int, winners_by_m: dict, arch: str, hres: str, dry_run: bool) -> bool:
    """Update `{arch}-MHC_FUSED_{HRES}-C={c}.json`. Return True if anything changed."""
    fname = f"{arch}-MHC_FUSED_{hres.upper()}-C={c}.json"
    path = CONFIG_DIR / fname
    if not path.exists():
        print(f"  [skip] {fname} (does not exist)")
        return False

    existing = json.loads(path.read_text())
    changes = []
    for M in sorted(winners_by_m):
        cfg, time_ms, _ = winners_by_m[M]
        bucket = f"M_LEQ_{M}"
        new_cfg = clean_config(cfg)
        if existing.get(bucket) != new_cfg:
            existing[bucket] = new_cfg
            changes.append(
                f"      {bucket}: {time_ms*1000:>7.1f}us  USE_REDUCE_SPLITC={new_cfg.get('USE_REDUCE_SPLITC')}"
            )

    # Refresh "any" with the largest-M tuned config.
    if winners_by_m:
        max_m = max(winners_by_m)
        new_any = clean_config(winners_by_m[max_m][0])
        if existing.get("any") != new_any:
            existing["any"] = new_any
            changes.append(f"      any: derived from M_LEQ_{max_m}")

    if not changes:
        print(f"  [unchanged] {fname}")
        return False

    print(f"  [update] {fname}")
    for line in changes:
        print(line)
    if not dry_run:
        path.write_text(json.dumps(existing, indent=4) + "\n")
    return True


def main():
    p = argparse.ArgumentParser(
        description="Merge MHC tuner winners into production config JSONs."
    )
    p.add_argument(
        "--arches",
        nargs="+",
        default=None,
        help=(
            "Architectures to update. Default: only the active GPU arch detected "
            "via aiter.arch_info (e.g., 'gfx950'). Pass multiple to mirror across "
            "arches, but only do this if you've verified the configs transfer "
            "cleanly - cross-arch writing can silently clobber known-good configs."
        ),
    )
    p.add_argument(
        "--hres-mode",
        default="sinkhorn",
        choices=["sinkhorn", "lite"],
        help="H_res mode the tuner ran in (default: sinkhorn).",
    )
    p.add_argument(
        "--tuner-glob",
        default=None,
        help=(
            "Glob (relative to repo root) for tuner output files. "
            f"Default: '{DEFAULT_TUNER_GLOB}'."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would change without writing files.",
    )
    args = p.parse_args()

    if args.arches is None:
        active = detect_active_arch()
        if not active:
            print(
                "[error] could not detect active GPU arch; pass --arches explicitly.",
                file=sys.stderr,
            )
            return 3
        args.arches = [active]
        print(f"[info] defaulting --arches to active arch: {active}")

    tuner_glob = args.tuner_glob or DEFAULT_TUNER_GLOB.format(hres=args.hres_mode)
    files = sorted(glob.glob(str(REPO_ROOT / tuner_glob)))
    if not files:
        print(f"[error] no tuner outputs found matching {tuner_glob!r}", file=sys.stderr)
        return 1

    print(f"Found {len(files)} tuner file(s):")
    for f in files:
        print(f"  - {Path(f).relative_to(REPO_ROOT)}")
    print()

    winners = collect_winners(files)
    if not winners:
        print("[error] no winners parsed from tuner files", file=sys.stderr)
        return 2

    any_changes = False
    for c in sorted(winners):
        for arch in args.arches:
            any_changes |= merge_into_json(
                c, winners[c], arch, args.hres_mode, args.dry_run
            )
        print()

    if args.dry_run:
        print("[dry-run] no files written")
    elif not any_changes:
        print("All target JSONs already match the tuner winners.")
    else:
        print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
