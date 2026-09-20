"""Concrete MLX cognitive backend for the persistent organism."""

from __future__ import annotations

import re
from typing import Mapping, Sequence

import numpy as np

from ..mlx_steering import extract_last_token_hiddens, unwrap, wrap_steering
from ..probes import ProbeBundle
from .backend import CognitiveResponse
from .mlx_bridge import build_mlx_offsets

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)


class MLXCognitiveBackend:
    """Frozen MLX LLM with arbitrary calibrated activation axes."""

    def __init__(
        self,
        model,
        tokenizer,
        probes: ProbeBundle,
        *,
        max_tokens: int = 80,
        temperature: float = 0.7,
        repetition_penalty: float = 1.2,
        repetition_context_size: int = 24,
    ) -> None:
        if not probes.selected_layers:
            raise ValueError("ProbeBundle has no selected_layers")
        self.model = model
        self.tokenizer = tokenizer
        self.probes = probes
        self.model_id = probes.model_id
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.repetition_penalty = float(repetition_penalty)
        self.repetition_context_size = int(repetition_context_size)

    def generate(
        self,
        messages: Sequence[Mapping[str, str]],
        steering_alphas: Mapping[str, float],
    ) -> CognitiveResponse:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        prompt = self.tokenizer.apply_chat_template(
            list(messages),
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        offsets = build_mlx_offsets(self.probes, steering_alphas)
        processors = make_logits_processors(
            repetition_penalty=self.repetition_penalty,
            repetition_context_size=self.repetition_context_size,
        )
        sampler = make_sampler(temp=self.temperature, top_p=0.9)

        originals = wrap_steering(self.model, offsets)
        try:
            raw = generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=self.max_tokens,
                verbose=False,
                sampler=sampler,
                logits_processors=processors,
            )
        finally:
            unwrap(self.model, originals)

        text = _scrub_output(raw)
        readback, raw_projection = self._readback(prompt, text, tuple(steering_alphas))
        return CognitiveResponse(
            text=text,
            readback=readback,
            raw_projection=raw_projection,
            metadata={
                "model_id": self.model_id,
                "steering_alphas": dict(steering_alphas),
            },
        )

    def _readback(
        self,
        prompt: str,
        output_text: str,
        axes: tuple[str, ...],
    ) -> tuple[dict[str, float], dict[str, float]]:
        if not axes:
            return {}, {}
        missing = sorted(axis for axis in axes if axis not in self.probes.selected_layers)
        if missing:
            raise KeyError(f"Readback axes missing selected layers: {missing}")

        layers = sorted({self.probes.selected_layers[axis] for axis in axes})
        hiddens = extract_last_token_hiddens(
            self.model,
            self.tokenizer,
            prompt + output_text,
            layers,
        )
        normalised: dict[str, float] = {}
        raw: dict[str, float] = {}
        for axis in axes:
            layer = self.probes.selected_layers[axis]
            vec = self.probes.vec(axis, layer).astype(np.float32)
            hidden = hiddens[layer].astype(np.float32)
            projection = float(np.dot(hidden, vec))
            raw[axis] = projection
            diag = self.probes.diagnostics.get(axis, {}).get(layer, {})
            calibration = max(float(diag.get("norm_unnormalised", 2.0)) / 2.0, 1.0)
            normalised[axis] = float(np.clip(projection / calibration, -1.5, 1.5))
        return normalised, raw


def _scrub_output(text: str) -> str:
    text = _THINK_BLOCK_RE.sub("", text)
    text = _OPEN_THINK_RE.sub("", text)
    return text.strip()
