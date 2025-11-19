"""
Helper utilities for optional Weights & Biases logging.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional


def maybe_init_wandb(enable: bool, project: str, run_name: Optional[str], config: Dict[str, Any]):
    if not enable:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is not installed. Install it or disable --wandb.") from exc
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    return wandb.init(project=project, name=run_name, config=config)


def wandb_log(run, metrics: Dict[str, Any]):
    if run is None:
        return
    run.log(metrics)


def wandb_finish(run):
    if run is None:
        return
    run.finish()


