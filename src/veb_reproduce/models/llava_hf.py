"""Hugging Face LLaVA image-text generation adapter."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Mapping

from veb_reproduce.models.types import GenerationResult, TokenScore
from veb_reproduce.veb.token_alignment import token_spans_from_ids


class LlavaHfGenerator:
    """LLaVA-1.5 adapter built on `transformers`."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        try:
            import torch
            from PIL import Image
            from transformers import AutoProcessor, LlavaForConditionalGeneration
        except ImportError as exc:  # pragma: no cover - optional model deps
            raise RuntimeError(
                "LLaVA generation requires the model stack. Install requirements-models-cu12.txt."
            ) from exc

        self._torch = torch
        self._image_cls = Image
        generation_config = config.get("generation", {})
        runtime_config = config.get("runtime", {})
        self.model_name_or_path = generation_config.get(
            "model_name_or_path", "llava-hf/llava-1.5-7b-hf"
        )
        self.device = runtime_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = _resolve_dtype(torch, runtime_config.get("dtype", "float16"), self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_name_or_path)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

        self.temperature = float(generation_config.get("temperature", 0.0))
        self.top_p = float(generation_config.get("top_p", 1.0))
        self.do_sample = bool(generation_config.get("do_sample", False))
        self.return_token_scores = bool(generation_config.get("return_token_scores", True))

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
        formatted_prompt = _format_llava_prompt(self.processor, prompt)
        inputs = self.processor(text=formatted_prompt, images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        input_token_count = int(inputs["input_ids"].shape[-1])

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": int(max_new_tokens or 64),
            "do_sample": self.do_sample,
            "top_p": self.top_p,
        }
        if self.do_sample:
            generation_kwargs["temperature"] = self.temperature
        if self.return_token_scores:
            generation_kwargs.update({"output_scores": True, "return_dict_in_generate": True})

        start = time.perf_counter()
        with self._torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        latency_sec = time.perf_counter() - start

        sequence = outputs.sequences[0] if self.return_token_scores else outputs[0]
        generated_ids_tensor = sequence[input_token_count:]
        generated_ids = [int(item) for item in generated_ids_tensor.tolist()]
        text, spans = token_spans_from_ids(
            generated_ids,
            lambda ids: self.processor.decode(ids, skip_special_tokens=True),
        )
        token_scores = self._build_token_scores(outputs, generated_ids, spans) if self.return_token_scores else None
        return GenerationResult(
            text=text,
            latency_sec=latency_sec,
            token_scores=token_scores,
            metadata={"backend": "llava_hf", "sample_id": sample_id},
        )

    def _build_token_scores(
        self, outputs: Any, generated_ids: list[int], spans: list[tuple[int, int]]
    ) -> list[TokenScore]:
        records: list[TokenScore] = []
        vocab_size = int(outputs.scores[0].shape[-1]) if outputs.scores else 1
        transition_scores = self.model.compute_transition_scores(
            outputs.sequences, outputs.scores, normalize_logits=True
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
                    token_text=self.processor.decode([token_id], skip_special_tokens=True),
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


def _format_llava_prompt(processor: Any, prompt: str) -> str:
    if _looks_preformatted_prompt(prompt):
        return prompt
    if hasattr(processor, "apply_chat_template"):
        prompt_text = prompt.replace("<image>", "").replace("<IMAGE>", "").strip()
        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}
        ]
        try:
            return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
    return f"USER: <image>\n{prompt}\nASSISTANT:"


def _looks_preformatted_prompt(prompt: str) -> bool:
    normalized = prompt.upper()
    return "<IMAGE>" in normalized and ("USER:" in normalized or "ASSISTANT:" in normalized)


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
