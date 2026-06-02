from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class VideoConfig:
    fps: float = 10.0
    width: int | None = None
    height: int | None = None
    format: str = "mp4"
    codec: str = "h264"
    resize_mode: str = "preserve"


@dataclass
class GazeConfig:
    frequency_hz: float = 10.0
    coordinates: str = "normalized"
    file_format: str = "parquet"
    interpolation: str = "linear"


@dataclass
class AnnotationConfig:
    frequency_hz: float = 10.0
    file_format: str = "parquet"
    sample_mode: str = "active_interval"


@dataclass
class DepthConfig:
    enabled: bool = True
    frequency_hz: float = 10.0
    units: str = "meters"
    value_mapping: str = "gaze_aligned_scalar"
    file_format: str = "parquet"
    interpolation: str = "linear"


@dataclass
class ValidationConfig:
    time_tolerance_s: float = 0.000001
    duration_tolerance_s: float = 0.05
    numeric_tolerance: float = 0.00001
    frame_time_tolerance_s: float = 0.05
    image_difference_tolerance: float = 0.03


@dataclass
class RectifyConfig:
    profile_name: str = "default-10hz"
    target_hz: float = 10.0
    video: VideoConfig = field(default_factory=VideoConfig)
    gaze: GazeConfig = field(default_factory=GazeConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_config() -> RectifyConfig:
    return RectifyConfig()


def _deep_update(target: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


def _coerce_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        if "." not in value:
            return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _set_nested(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = target
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _from_dict(data: dict[str, Any]) -> RectifyConfig:
    return RectifyConfig(
        profile_name=data.get("profile_name", "default-10hz"),
        target_hz=float(data.get("target_hz", 10.0)),
        video=VideoConfig(**data.get("video", {})),
        gaze=GazeConfig(**data.get("gaze", {})),
        annotation=AnnotationConfig(**data.get("annotation", {})),
        depth=DepthConfig(**data.get("depth", {})),
        validation=ValidationConfig(**data.get("validation", {})),
    )


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> RectifyConfig:
    data = default_config().to_dict()
    if path:
        loaded = json.loads(Path(path).read_text())
        if not isinstance(loaded, dict):
            raise ValueError("Config file must contain a JSON object")
        _deep_update(data, loaded)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got {override!r}")
        key, value = override.split("=", 1)
        _set_nested(data, key, _coerce_value(value))
    cfg = _from_dict(data)
    _validate_config(cfg)
    return cfg


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _validate_config(cfg: RectifyConfig) -> None:
    _validate_positive("target_hz", cfg.target_hz)
    _validate_positive("video.fps", cfg.video.fps)
    _validate_positive("gaze.frequency_hz", cfg.gaze.frequency_hz)
    _validate_positive("annotation.frequency_hz", cfg.annotation.frequency_hz)
    _validate_positive("depth.frequency_hz", cfg.depth.frequency_hz)
    if cfg.video.resize_mode not in {"preserve", "stretch", "pad", "crop"}:
        raise ValueError("video.resize_mode must be preserve, stretch, pad, or crop")
    if cfg.gaze.coordinates not in {"normalized", "pixel", "both"}:
        raise ValueError("gaze.coordinates must be normalized, pixel, or both")
    if cfg.depth.units != "meters":
        raise ValueError("depth.units must be meters")
