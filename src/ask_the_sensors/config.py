from __future__ import annotations
import os
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict
import yaml

def find_repo_root(start: Path | None = None) -> Path:
    here = start or Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "configs" / "config.yaml").is_file():
            return parent
    return Path(__file__).resolve().parents[2]

REPO_ROOT = find_repo_root()
CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"

@dataclass
class LLMConfig:
    host: str
    model: str
    request_timeout_seconds: int
    temperature: float
    max_retries: int
    max_evidence_rows_shown: int

@dataclass
class Config:
    raw: Dict[str, Any]
    repo_root: Path

    def path(self, *keys: str) -> Path:
        node = self.raw["paths"]
        for k in keys:
            node = node[k]
        return (self.repo_root / node).resolve()

    @property
    def downloads_dir(self) -> Path:
        return self.path("downloads_dir")

    @property
    def primary_data_dir(self) -> Path:
        return self.path("primary_data_dir")

    @property
    def cv_folds_dir(self) -> Path:
        return self.path("cv_folds_dir")

    @property
    def processed_dir(self) -> Path:
        return self.path("processed_dir")

    @property
    def outputs_dir(self) -> Path:
        return self.path("outputs_dir")

    @property
    def models_dir(self) -> Path:
        return self.path("models_dir")

    @property
    def figures_dir(self) -> Path:
        return self.path("figures_dir")

    @property
    def logs_dir(self) -> Path:
        return self.path("logs_dir")

    @property
    def answers_dir(self) -> Path:
        return self.path("answers_dir")

    def ensure_dirs(self) -> None:
        for p in [
            self.downloads_dir,
            self.primary_data_dir,
            self.cv_folds_dir,
            self.processed_dir,
            self.outputs_dir,
            self.models_dir,
            self.figures_dir,
            self.logs_dir,
            self.answers_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)

    @property
    def activity_classes(self) -> list[str]:
        return list(self.raw["activity_classes"])

    @property
    def activity_display_names(self) -> Dict[str, str]:
        return dict(self.raw["activity_display_names"])

    @property
    def feature_prefixes(self) -> Dict[str, str]:
        return dict(self.raw["feature_prefixes"])

    @property
    def canonical_sample_rate_hz(self) -> int:
        return int(self.raw["canonical_sample_rate_hz"])

    @property
    def example_duration_seconds(self) -> int:
        return int(self.raw["example_duration_seconds"])

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    @property
    def llm(self) -> LLMConfig:
        llm_cfg = self.raw["llm"]
        host = os.getenv(llm_cfg["host_env_var"], llm_cfg["default_host"])
        model = os.getenv(llm_cfg["model_env_var"], llm_cfg["default_model"])
        return LLMConfig(
            host=host,
            model=model,
            request_timeout_seconds=int(llm_cfg["request_timeout_seconds"]),
            temperature=float(llm_cfg["temperature"]),
            max_retries=int(llm_cfg["max_retries"]),
            max_evidence_rows_shown=int(llm_cfg["max_evidence_rows_shown"]),
        )

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

_CONFIG_CACHE: Config | None = None

def load_config(config_path: Path | None = None, force_reload: bool = False) -> Config:
    """Load (and cache) the project configuration."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and not force_reload and config_path is None:
        return _CONFIG_CACHE

    cp = config_path or CONFIG_PATH
    with open(cp, "r") as f:
        raw = yaml.safe_load(f)

    cfg = Config(raw=copy.deepcopy(raw), repo_root=REPO_ROOT)
    if config_path is None:
        _CONFIG_CACHE = cfg
    return cfg

if __name__ == "__main__":
    cfg = load_config()
    print("Repo root:", cfg.repo_root)
    print("Primary data dir:", cfg.primary_data_dir)
    print("CV folds dir:", cfg.cv_folds_dir)
    print("LLM config:", cfg.llm)