"""Video frame dataset utilities for PyTorch."""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple

import imageio.v3 as iio
import numpy as np
import torch
import kornia
from kornia.filters import GaussianBlur2d
from kornia.geometry.transform import resize as kornia_resize
from kornia.geometry.transform import translate as kornia_translate
from PIL import Image
from torch.utils.data import IterableDataset

Resampling = getattr(Image, "Resampling", Image)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoBatch:
    """Container returned by :class:`VideoFrameDataset`.

    Attributes
    ----------
    degraded:
        A tensor of degraded frames with shape ``(batch, channels, height, width)``.
    clean:
        A tensor containing the clean (non-degraded) frames with the same shape as
        :attr:`degraded`.
    metadata:
        Optional information about the degradation that has been applied.
    """

    degraded: torch.Tensor
    clean: torch.Tensor
    metadata: Optional[dict]


class VideoFrameDataset(IterableDataset):
    """Stream frames from a directory of videos with dynamic degradations.

    The dataset discovers all compatible video files below ``dataset_path`` and
    iterates over them in random order. For each video, its frames are shuffled and
    emitted as batches of ``batch_size`` frames. Each batch is degraded using a
    single randomly-sampled degradation pipeline that is applied consistently across
    every frame in the batch. Both degraded and pristine frames are returned to
    facilitate supervised training setups.

    Parameters
    ----------
    dataset_path:
        Root directory containing video files in any nested structure.
    batch_size:
        Number of frames in each yielded batch.
    input_size:
        Desired frame size as ``(height, width)``. Frames are resized using
        high-quality resampling prior to degradation.
    seed:
        Optional seed for deterministic shuffling and degradation selection. When
        using multi-processing data loaders, the seed is offset by the worker id so
        that each worker traverses the dataset independently.
    extensions:
        Sequence of video file extensions (case-insensitive) that should be treated
        as valid videos.
    drop_last:
        Whether to drop incomplete batches of frames at the end of a video.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        batch_size: int,
        input_size: Tuple[int, int],
        *,
        seed: Optional[int] = None,
        extensions: Sequence[str] = (".mp4", ".mkv", ".webm", ".mov", ".avi"),
        drop_last: bool = False,
    ) -> None:
        super().__init__()
        self.dataset_path = Path(dataset_path)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = int(batch_size)
        if len(input_size) != 2:
            raise ValueError("input_size must be a (height, width) tuple")
        self.input_size = tuple(int(v) for v in input_size)
        self.seed = seed if seed is not None else random.randrange(1 << 30)
        self.extensions = tuple(e.lower() for e in extensions)
        self.drop_last = drop_last

        self._degradation_factories: Tuple[
            Callable[[random.Random], Tuple[Callable[[torch.Tensor], torch.Tensor], dict]],
            ...,
        ] = (
            self._down_upscale_factory,
            self._jpeg_factory,
            self._banding_factory,
            self._ghosting_factory,
            self._blur_noise_factory,
        )

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"dataset_path '{self.dataset_path}' does not exist")

    # ------------------------------------------------------------------
    # IterableDataset API
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator[VideoBatch]:  # type: ignore[override]
        worker_info = torch.utils.data.get_worker_info()
        worker_seed = self.seed
        if worker_info is not None:
            worker_seed += worker_info.id
        rng = random.Random(worker_seed)

        video_files = self._discover_videos()
        rng.shuffle(video_files)

        for video_path in video_files:
            frames = self._read_video_frames(video_path)
            if not frames:
                LOGGER.warning("Video %s did not yield any frames", video_path)
                continue

            rng.shuffle(frames)
            for batch in self._batch_frames(frames):
                degradation_fn, metadata = self._choose_degradation(rng)
                clean_tensor = torch.stack([self._prepare_frame(img.copy()) for img in batch])
                degraded_tensor = degradation_fn(clean_tensor)
                yield VideoBatch(degraded=degraded_tensor, clean=clean_tensor, metadata=metadata)

    # ------------------------------------------------------------------
    # Video discovery and batching helpers
    # ------------------------------------------------------------------
    def _discover_videos(self) -> List[Path]:
        candidates = [
            path
            for path in self.dataset_path.rglob("*")
            if path.is_file() and path.suffix.lower() in self.extensions
        ]
        if not candidates:
            LOGGER.warning("No video files found in %s", self.dataset_path)
        return candidates

    def _read_video_frames(self, video_path: Path) -> List[Image.Image]:
        frames: List[Image.Image] = []
        try:
            try:
                iterator = iio.imiter(video_path, plugin="pyav")
            except Exception:
                iterator = iio.imiter(video_path)
            for frame in iterator:
                image = Image.fromarray(frame).convert("RGB")
                image = image.resize(self.input_size[::-1], Resampling.LANCZOS)
                frames.append(image)
        except Exception as exc:  # pragma: no cover - log for visibility
            LOGGER.error("Failed to read %s: %s", video_path, exc)
        return frames

    def _batch_frames(self, frames: Sequence[Image.Image]) -> Iterable[Sequence[Image.Image]]:
        if self.drop_last:
            total_batches = len(frames) // self.batch_size
            for idx in range(total_batches):
                start = idx * self.batch_size
                end = start + self.batch_size
                yield frames[start:end]
        else:
            for idx in range(0, len(frames), self.batch_size):
                yield frames[idx : idx + self.batch_size]

    # ------------------------------------------------------------------
    # Degradation pipelines
    # ------------------------------------------------------------------
    def _choose_degradation(
        self, rng: random.Random
    ) -> Tuple[Callable[[torch.Tensor], torch.Tensor], dict]:
        factory = rng.choice(self._degradation_factories)
        return factory(rng)

    def _down_upscale_factory(
        self, rng: random.Random
    ) -> Tuple[Callable[[torch.Tensor], torch.Tensor], dict]:
        scale = rng.uniform(0.35, 0.7)
        down_method = rng.choice(("nearest", "bilinear", "bicubic"))
        up_method = rng.choice(("bilinear", "bicubic"))

        def apply(batch: torch.Tensor) -> torch.Tensor:
            if batch.ndim != 4:
                raise ValueError("Expected a batched tensor of shape (B, C, H, W)")
            height, width = batch.shape[-2:]
            down_size = (
                max(1, int(height * scale)),
                max(1, int(width * scale)),
            )
            down_align_corners = None if down_method == "nearest" else False
            downscaled = kornia_resize(
                batch,
                down_size,
                interpolation=down_method,
                align_corners=down_align_corners,
                antialias=True,
            )
            restored = kornia_resize(
                downscaled,
                (height, width),
                interpolation=up_method,
                align_corners=False,
                antialias=True,
            )
            return torch.clamp(restored, 0.0, 1.0)

        metadata = {
            "type": "down_upscale",
            "scale": scale,
            "down_method": down_method,
            "up_method": up_method,
        }
        return apply, metadata

    def _jpeg_factory(self, rng: random.Random) -> Tuple[Callable[[torch.Tensor], torch.Tensor], dict]:
        quality = float(rng.uniform(5.0, 40.0))

        def apply(batch: torch.Tensor) -> torch.Tensor:
            if batch.ndim != 4:
                raise ValueError("Expected a batched tensor of shape (B, C, H, W)")
            quality_tensor = torch.full(
                (batch.shape[0],), quality, dtype=batch.dtype, device=batch.device
            )
            degraded = kornia.enhance.jpeg_codec_differentiable(batch, quality_tensor)
            return torch.clamp(degraded, 0.0, 1.0)

        metadata = {
            "type": "jpeg",
            "quality": quality,
        }
        return apply, metadata

    def _banding_factory(self, rng: random.Random) -> Tuple[Callable[[torch.Tensor], torch.Tensor], dict]:
        bits = rng.randint(3, 5)
        apply_dither = rng.random() < 0.5
        dither_strength = rng.uniform(0.002, 0.01) if apply_dither else 0.0
        dither_seed = rng.randrange(1 << 30)

        def apply(batch: torch.Tensor) -> torch.Tensor:
            if batch.ndim != 4:
                raise ValueError("Expected a batched tensor of shape (B, C, H, W)")
            posterized = kornia.enhance.posterize(batch, bits)
            if not apply_dither:
                return posterized
            generator = torch.Generator(device=batch.device)
            generator.manual_seed(dither_seed)
            noise = torch.rand(
                (1, *batch.shape[1:]), generator=generator, device=batch.device, dtype=batch.dtype
            )
            noise = (noise - 0.5) * 2.0 * dither_strength
            return torch.clamp(posterized + noise.expand_as(batch), 0.0, 1.0)

        metadata = {
            "type": "banding",
            "bits": bits,
            "dither_strength": dither_strength,
        }
        return apply, metadata

    def _ghosting_factory(
        self, rng: random.Random
    ) -> Tuple[Callable[[torch.Tensor], torch.Tensor], dict]:
        dx = float(rng.randint(-4, 4))
        dy = float(rng.randint(-4, 4))
        blend = float(rng.uniform(0.15, 0.4))
        blur_sigma = float(rng.uniform(0.5, 1.5))
        kernel_size = self._kernel_size_for_sigma(blur_sigma)
        blur_module = GaussianBlur2d(
            kernel_size=(kernel_size, kernel_size),
            sigma=(blur_sigma, blur_sigma),
            border_type="reflect",
        )

        def apply(batch: torch.Tensor) -> torch.Tensor:
            if batch.ndim != 4:
                raise ValueError("Expected a batched tensor of shape (B, C, H, W)")
            module = blur_module.to(device=batch.device, dtype=batch.dtype)
            translation = torch.tensor(
                [[dx, dy]], dtype=batch.dtype, device=batch.device
            ).expand(batch.shape[0], -1)
            shifted = kornia_translate(
                batch,
                translation,
                mode="bilinear",
                padding_mode="reflection",
                align_corners=False,
            )
            blurred = module(shifted)
            return torch.clamp(torch.lerp(batch, blurred, blend), 0.0, 1.0)

        metadata = {
            "type": "ghosting",
            "dx": dx,
            "dy": dy,
            "blend": blend,
            "blur_sigma": blur_sigma,
        }
        return apply, metadata

    def _blur_noise_factory(
        self, rng: random.Random
    ) -> Tuple[Callable[[torch.Tensor], torch.Tensor], dict]:
        blur_sigma = float(rng.uniform(0.8, 1.6))
        kernel_size = self._kernel_size_for_sigma(blur_sigma)
        noise_sigma = float(rng.uniform(3.0, 10.0) / 255.0)
        noise_seed = rng.randrange(1 << 30)
        blur_module = GaussianBlur2d(
            kernel_size=(kernel_size, kernel_size),
            sigma=(blur_sigma, blur_sigma),
            border_type="reflect",
        )

        def apply(batch: torch.Tensor) -> torch.Tensor:
            if batch.ndim != 4:
                raise ValueError("Expected a batched tensor of shape (B, C, H, W)")
            module = blur_module.to(device=batch.device, dtype=batch.dtype)
            blurred = module(batch)
            generator = torch.Generator(device=batch.device)
            generator.manual_seed(noise_seed)
            noise = torch.randn(
                (1, *batch.shape[1:]), generator=generator, device=batch.device, dtype=batch.dtype
            )
            degraded = blurred + noise.expand_as(batch) * noise_sigma
            return torch.clamp(degraded, 0.0, 1.0)

        metadata = {
            "type": "blur_noise",
            "blur_sigma": blur_sigma,
            "kernel_size": kernel_size,
            "noise_sigma": noise_sigma,
        }
        return apply, metadata

    def _kernel_size_for_sigma(self, sigma: float) -> int:
        size = max(3, int(math.ceil(sigma * 6.0)))
        if size % 2 == 0:
            size += 1
        return size

    # ------------------------------------------------------------------
    # Frame preparation
    # ------------------------------------------------------------------
    def _prepare_frame(self, image: Image.Image) -> torch.Tensor:
        array = np.asarray(image).astype(np.float32) / 255.0
        if array.ndim == 2:  # grayscale fallback
            array = np.stack([array] * 3, axis=-1)
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor

    # ------------------------------------------------------------------
    # Convenience utilities
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        video_files = self._discover_videos()
        total_frames = sum(self._count_video_frames(path) for path in video_files)
        if self.drop_last:
            return total_frames // self.batch_size
        return math.ceil(total_frames / self.batch_size)

    def _count_video_frames(self, path: Path) -> int:
        try:
            props = iio.improps(path, plugin="pyav")
        except Exception:
            try:
                props = iio.improps(path)
            except Exception:
                return 0
        shape = getattr(props, "shape", None)
        if not shape:
            return 0
        return int(shape[0])
