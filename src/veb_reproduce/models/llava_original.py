"""Original LLaVA image-text generation adapter."""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any, Mapping

from veb_reproduce.models.types import GenerationResult, TokenScore
from veb_reproduce.utils.config import PROJECT_ROOT
from veb_reproduce.veb.token_alignment import token_spans_from_ids


class OriginalLlavaGenerator:
    """LLaVA-1.5 adapter built on the original `llava` package."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        try:
            import numpy as np
            import torch
            import torch.backends.cudnn as cudnn
            from PIL import Image
            from llava.constants import (
                DEFAULT_IMAGE_TOKEN,
                DEFAULT_IM_END_TOKEN,
                DEFAULT_IM_START_TOKEN,
                IMAGE_TOKEN_INDEX,
            )
            from llava.conversation import conv_templates
            from llava.mm_utils import (
                get_model_name_from_path,
                process_images,
                tokenizer_image_token,
            )
            from llava.model.builder import load_pretrained_model
        except ImportError as exc:  # pragma: no cover - optional model deps
            raise RuntimeError(
                "Original LLaVA generation requires the official LLaVA package. "
                "Install the model stack from requirements-models-cu12.txt or install "
                "`git+https://github.com/haotian-liu/LLaVA.git`."
            ) from exc

        self._np = np
        self._torch = torch
        self._cudnn = cudnn
        self._image_cls = Image
        self._default_image_token = DEFAULT_IMAGE_TOKEN
        self._default_im_start_token = DEFAULT_IM_START_TOKEN
        self._default_im_end_token = DEFAULT_IM_END_TOKEN
        self._image_token_index = IMAGE_TOKEN_INDEX
        self._conv_templates = conv_templates
        self._process_images = process_images
        self._tokenizer_image_token = tokenizer_image_token

        generation_config = config.get("generation", {})
        runtime_config = config.get("runtime", {})
        self.model_name_or_path = _resolve_model_reference(
            generation_config.get("model_name_or_path", "models/llava-v1.5-7b")
        )
        self.model_base = _optional_model_reference(generation_config.get("model_base"))
        self.device = runtime_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = _resolve_dtype(torch, runtime_config.get("dtype", "float16"), self.device)
        self.seed = int(runtime_config.get("seed", 42))
        self.conv_mode = str(generation_config.get("conv_mode", "llava_v1"))
        self.temperature = float(generation_config.get("temperature", 0.0))
        self.top_p = float(generation_config.get("top_p", 1.0))
        self.do_sample = bool(generation_config.get("do_sample", False))
        self.num_beams = int(generation_config.get("num_beams", 1))
        self.return_token_scores = bool(generation_config.get("return_token_scores", True))
        self._setup_seed()

        if self.conv_mode not in self._conv_templates:
            raise ValueError(f"Unsupported LLaVA conversation mode: {self.conv_mode}")

        model_name = str(
            generation_config.get("model_name")
            or get_model_name_from_path(self.model_name_or_path)
        )
        load_kwargs = {
            "load_8bit": bool(generation_config.get("load_8bit", False)),
            "load_4bit": bool(generation_config.get("load_4bit", False)),
            "device": self.device,
        }
        device_map = generation_config.get("device_map")
        if device_map is not None:
            load_kwargs["device_map"] = device_map

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            self.model_name_or_path,
            self.model_base,
            model_name,
            **load_kwargs,
        )
        self.model.eval()
        if self.device == "cpu" and hasattr(self.model, "to"):
            self.model.to(self.device)

    def _setup_seed(self) -> None:
        random.seed(self.seed)
        self._np.random.seed(self.seed)
        self._torch.manual_seed(self.seed)
        self._cudnn.benchmark = False
        self._cudnn.deterministic = True

    def generate(
        self,
        image_path: str | Path,
        prompt: str,
        *,
        sample_id: str | None = None,
        max_new_tokens: int | None = None,
    ) -> GenerationResult:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        image = self._image_cls.open(path).convert("RGB")
        formatted_prompt = self._format_prompt(prompt)
        input_ids = self._tokenizer_image_token(
            formatted_prompt,
            self.tokenizer,
            self._image_token_index,
            return_tensors="pt",
        ).unsqueeze(0)
        input_ids = input_ids.to(self.device)
        input_token_count = int(input_ids.shape[-1])
        image_tensor = self._process_images([image], self.image_processor, self.model.config)
        if isinstance(image_tensor, list):
            image_tensor = self._torch.stack(image_tensor, dim=0)
        image_tensor = image_tensor.to(self.device, dtype=self.dtype)

        generation_kwargs: dict[str, Any] = {
            "images": image_tensor,
            "image_sizes": [image.size],
            "max_new_tokens": int(max_new_tokens or 64),
            "do_sample": self.do_sample,
            "top_p": self.top_p,
            "num_beams": self.num_beams,
            "use_cache": True,
        }
        if self.do_sample:
            generation_kwargs["temperature"] = self.temperature
        if self.return_token_scores:
            generation_kwargs.update({"output_scores": True, "return_dict_in_generate": True})

        start = time.perf_counter()
        with self._torch.inference_mode():
            outputs = self.model.generate(input_ids, **generation_kwargs)
        latency_sec = time.perf_counter() - start

        sequence = outputs.sequences[0] if self.return_token_scores else outputs[0]
        generated_ids_tensor = sequence[input_token_count:]
        generated_ids = [int(item) for item in generated_ids_tensor.tolist()]
        text, spans = token_spans_from_ids(
            generated_ids,
            lambda ids: self.tokenizer.decode(ids, skip_special_tokens=True),
        )
        token_scores = (
            self._build_token_scores(outputs, generated_ids, spans)
            if self.return_token_scores
            else None
        )
        return GenerationResult(
            text=text,
            latency_sec=latency_sec,
            token_scores=token_scores,
            metadata={
                "backend": "llava_original",
                "model_name_or_path": self.model_name_or_path,
                "sample_id": sample_id,
            },
        )

    def _format_prompt(self, prompt: str) -> str:
        if _looks_preformatted_prompt(prompt):
            return (
                prompt.replace("<IMAGE>", self._default_image_token)
                .replace("<ImageHere>", self._default_image_token)
                .replace("<IMAGEHERE>", self._default_image_token)
            )
        prompt_text = (
            prompt.replace("<image>", "")
            .replace("<IMAGE>", "")
            .replace("<ImageHere>", "")
            .strip()
        )
        image_token = self._default_image_token
        if getattr(self.model.config, "mm_use_im_start_end", False):
            image_token = f"{self._default_im_start_token}{image_token}{self._default_im_end_token}"
        conv = self._conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], f"{image_token}\n{prompt_text}")
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def _build_token_scores(
        self, outputs: Any, generated_ids: list[int], spans: list[tuple[int, int]]
    ) -> list[TokenScore]:
        records: list[TokenScore] = []
        if not getattr(outputs, "scores", None):
            return records
        vocab_size = int(outputs.scores[0].shape[-1]) if outputs.scores else 1
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences,
            outputs.scores,
            getattr(outputs, "beam_indices", None),
            normalize_logits=True,
        )[0]
        for index, (token_id, logits) in enumerate(zip(generated_ids, outputs.scores), start=1):
            row = logits[0].float()
            probs = self._torch.softmax(row, dim=-1)
            top_probs = self._torch.topk(probs, k=2).values.tolist()
            entropy = float(-(probs * self._torch.log(probs.clamp_min(1e-12))).sum().item())
            entropy = entropy / math.log(max(vocab_size, 2))
            margin = float(top_probs[0] - top_probs[1])
            logprob = float(transition_scores[index - 1].item())
            span = spans[index - 1] if index - 1 < len(spans) else (0, 0)
            records.append(
                TokenScore(
                    token_id=token_id,
                    token_text=self.tokenizer.decode([token_id], skip_special_tokens=True),
                    token_char_span=span,
                    logprob=logprob,
                    top1_prob=float(top_probs[0]),
                    top2_prob=float(top_probs[1]),
                    entropy=entropy,
                    margin=margin,
                    position=index,
                )
            )
        return records


def _resolve_model_reference(value: Any) -> str:
    raw = str(value)
    if "llava-hf/" in raw.lower() or raw.lower().endswith("-hf"):
        raise ValueError(
            "HF-converted LLaVA checkpoints are disabled. Use the original merged "
            "LLaVA checkpoint, e.g. `models/llava-v1.5-7b` or `liuhaotian/llava-v1.5-7b`."
        )
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    local_path = PROJECT_ROOT / path
    if raw.startswith(("models/", "models\\", "./", ".\\")) or local_path.exists():
        return str(local_path)
    return raw


def _optional_model_reference(value: Any) -> str | None:
    if value in {None, ""}:
        return None
    return _resolve_model_reference(value)


def _looks_preformatted_prompt(prompt: str) -> bool:
    normalized = prompt.upper()
    has_image = "<IMAGE>" in normalized or "<IMAGEHERE>" in normalized
    return has_image and ("USER:" in normalized or "ASSISTANT:" in normalized)


def _resolve_dtype(torch: Any, dtype_name: str, device: str) -> Any:
    if device == "cpu":
        return torch.float32
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported runtime.dtype: {dtype_name}")
    return mapping[dtype_name]
