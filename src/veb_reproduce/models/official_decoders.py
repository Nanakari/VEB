"""Adapters for official HALC and OPERA decoder repositories."""

from __future__ import annotations

import importlib
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from veb_reproduce.models.types import GenerationResult
from veb_reproduce.utils.config import PROJECT_ROOT


class OfficialDecoderGenerator:
    """Run official HALC/OPERA LLaVA decoding code from a local source checkout."""

    def __init__(self, config: Mapping[str, Any], method: str) -> None:
        if method not in {"halc", "opera"}:
            raise ValueError(f"Unsupported official decoder method: {method}")
        self.method = method
        self.config = config
        generation_config = config.get("generation", {})
        method_config = generation_config.get(method, {})
        if not isinstance(method_config, Mapping):
            method_config = {}
        runtime_config = config.get("runtime", {})
        dtype_name = str(runtime_config.get("dtype", "float16")).lower()
        if dtype_name not in {"float16", "fp16"}:
            raise ValueError(
                f"Official {method.upper()} decoding uses fp16 LLaVA code. "
                "Set runtime.dtype to float16 so all methods share the same precision."
            )
        self.repo_path = _resolve_repo_path(
            method_config.get("repo_path")
            or config.get("external_methods", {}).get(method, {}).get("repo_path")
            or f"external/{method.upper()}"
        )
        if not self.repo_path.exists():
            raise RuntimeError(
                f"Official {method.upper()} source not found: {self.repo_path}. "
                f"Clone the official repo there or set generation.{method}.repo_path."
            )

        try:
            self._prepare_imports()
            import numpy as np
            import torch
            import torch.backends.cudnn as cudnn
            from PIL import Image
            from torchvision import transforms
            from minigpt4.common.config import Config
            from minigpt4.common.registry import registry
            from minigpt4.models import load_preprocess
        except ImportError as exc:  # pragma: no cover - optional official deps
            raise RuntimeError(
                f"Official {method.upper()} decoding requires the dependencies from "
                f"{self.repo_path / 'environment.yml'} or {self.repo_path / 'requirements.txt'}."
            ) from exc

        self._np = np
        self._torch = torch
        self._cudnn = cudnn
        self._image_cls = Image
        self._transforms = transforms
        self._registry = registry
        self._load_preprocess = load_preprocess

        self.device = runtime_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.seed = int(runtime_config.get("seed", 42))
        self._setup_seed()

        model_path = _resolve_model_reference(
            generation_config.get("model_name_or_path", "models/llava-v1.5-7b")
        )
        cfg_path = Path(method_config.get("eval_config", "eval_configs/llava-1.5_eval.yaml"))
        if not cfg_path.is_absolute():
            cfg_path = self.repo_path / cfg_path
        options = ["model.merged_ckpt", model_path]
        if "low_resource" in method_config:
            options.extend(["model.low_resource", str(bool(method_config["low_resource"])).lower()])
        args = SimpleNamespace(
            cfg_path=str(cfg_path),
            options=options,
        )
        cfg = Config(args)
        model_config = cfg.model_cfg
        model_config.device_8bit = _device_index(self.device)
        model_cls = registry.get_model_class(model_config.arch)
        self.model = model_cls.from_config(model_config).to(self.device)
        self.model.eval()

        processor_cfg = cfg.get_config().preprocess
        try:
            processor_cfg.vis_processor.eval.do_normalize = False
        except (AttributeError, KeyError):
            pass
        self.vis_processors, _ = load_preprocess(processor_cfg)
        self.vis_processor = self.vis_processors["eval"]
        self.halc_vis_processor = None
        if method == "halc":
            vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
            processor_cls = registry.get_processor_class(vis_processor_cfg.name)
            self.halc_vis_processor = processor_cls.from_config(vis_processor_cfg)

        self.norm = transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
        self.method_config = method_config
        self.temperature = float(generation_config.get("temperature", 0.0))
        self.top_p = float(generation_config.get("top_p", 1.0))
        self.do_sample = bool(generation_config.get("do_sample", False))

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
        raw_image = self._image_cls.open(path).convert("RGB")
        image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        prompt_text = _format_official_llava_prompt(prompt)
        start = time.perf_counter()
        with self._torch.inference_mode():
            if self.method == "halc":
                output = self._generate_halc(path, image, prompt_text, max_new_tokens)
            else:
                output = self._generate_opera(image, prompt_text, max_new_tokens)
        latency_sec = time.perf_counter() - start
        return GenerationResult(
            text=str(output[0]).strip() if output else "",
            latency_sec=latency_sec,
            token_scores=None,
            metadata={
                "backend": f"official_{self.method}",
                "method": self.method,
                "repo_path": str(self.repo_path),
                "sample_id": sample_id,
            },
        )

    def _generate_halc(
        self, image_path: Path, image: Any, prompt: str, max_new_tokens: int | None
    ) -> list[str]:
        try:
            from decoder_zoo.HALC.context_density.halc import halc_assistant
        except ImportError as exc:  # pragma: no cover - optional official deps
            raise RuntimeError(
                f"HALC assistant could not be imported from {self.repo_path}. "
                "Install the HALC repository dependencies, including GroundingDINO."
            ) from exc

        halc_config = self.method_config
        num_beams = int(halc_config.get("num_beams", 3))
        halc_params = {
            "context_domain": halc_config.get("context_domain", "upper"),
            "contrast_weight": float(halc_config.get("contrast_weight", 0.05)),
            "context_window": int(halc_config.get("context_window", 3)),
            "expand_ratio": float(halc_config.get("expand_ratio", 0.6)),
            "beam_size": num_beams,
            "k_candidate_num": int(halc_config.get("k_candidate_num", 4)),
            "LVLM_backbone": "llava-1.5",
            "detector": halc_config.get("detector", "dino"),
            "score_type": halc_config.get("score_type", "BLIP"),
            "debugger": bool(halc_config.get("debugger", False)),
            "box_threshold": float(halc_config.get("box_threshold", 0.45)),
        }
        assistant = halc_assistant(
            self.model,
            vis_processor=self.halc_vis_processor,
            device=self.device,
            halc_params=halc_params,
        )
        assistant.update_input(img_path=str(image_path), input_prompt=prompt)
        early_exit_layers = list(halc_config.get("early_exit_layers", list(range(0, 33, 2))))
        mature_layer = int(halc_config.get("mature_layer", early_exit_layers[-1]))
        candidate_layers = list(
            halc_config.get("candidate_premature_layers", early_exit_layers[:-1])
        )
        return self.model.generate(
            {"image": self.norm(image), "prompt": prompt, "img_path": str(image_path)},
            use_nucleus_sampling=self.do_sample,
            num_beams=num_beams,
            max_new_tokens=int(max_new_tokens or 64),
            top_p=self.top_p,
            temperature=max(self.temperature, 1e-6),
            output_attentions=True,
            premature_layer=None,
            candidate_premature_layers=candidate_layers,
            mature_layer=mature_layer,
            beam_search=True,
            dola_decoding=True,
            halc_decoding=True,
            opera_decoding=False,
            vcd_decoding=False,
            halc_assistant=assistant,
        )

    def _generate_opera(self, image: Any, prompt: str, max_new_tokens: int | None) -> list[str]:
        opera_config = self.method_config
        return self.model.generate(
            {"image": self.norm(image), "prompt": prompt},
            use_nucleus_sampling=self.do_sample,
            num_beams=int(opera_config.get("num_beams", 5)),
            max_new_tokens=int(max_new_tokens or 64),
            top_p=self.top_p,
            temperature=max(self.temperature, 1e-6),
            output_attentions=True,
            opera_decoding=True,
            scale_factor=float(opera_config.get("scale_factor", 50)),
            threshold=int(opera_config.get("threshold", 15)),
            num_attn_candidates=int(opera_config.get("num_attn_candidates", 5)),
            penalty_weights=float(opera_config.get("penalty_weights", 1.0)),
        )

    def _prepare_imports(self) -> None:
        _insert_sys_path(self.repo_path)
        transformers_src = _embedded_transformers_src(self.repo_path)
        if transformers_src is not None:
            _insert_sys_path(transformers_src)
        mplug_owl2_src = self.repo_path / "mPLUG-Owl" / "mPLUG-Owl2"
        if mplug_owl2_src.exists():
            _insert_sys_path(mplug_owl2_src)
        for module_name in [
            "minigpt4.datasets.builders",
            "minigpt4.models",
            "minigpt4.processors",
            "minigpt4.runners",
            "minigpt4.tasks",
        ]:
            importlib.import_module(module_name)

    def _setup_seed(self) -> None:
        random.seed(self.seed)
        self._np.random.seed(self.seed)
        self._torch.manual_seed(self.seed)
        self._cudnn.benchmark = False
        self._cudnn.deterministic = True


def _insert_sys_path(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def _embedded_transformers_src(repo_path: Path) -> Path | None:
    candidates = sorted(repo_path.glob("transformers-*/src"), reverse=True)
    return candidates[0] if candidates else None


def _resolve_repo_path(value: Any) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _resolve_model_reference(value: Any) -> str:
    raw = str(value)
    if "llava-hf/" in raw.lower() or raw.lower().endswith("-hf"):
        raise ValueError(
            "HF-converted LLaVA checkpoints are disabled for official decoders. "
            "Use the original merged LLaVA checkpoint."
        )
    path = Path(raw)
    if path.is_absolute():
        return str(path)
    local_path = PROJECT_ROOT / path
    if raw.startswith(("models/", "models\\", "./", ".\\")) or local_path.exists():
        return str(local_path)
    return raw


def _device_index(device: str) -> int:
    if device.startswith("cuda:"):
        return int(device.split(":", maxsplit=1)[1])
    return 0


def _format_official_llava_prompt(prompt: str) -> str:
    normalized = prompt.replace("<image>", "<ImageHere>").replace("<IMAGE>", "<ImageHere>")
    if "<ImageHere>" in normalized and "USER:" in normalized.upper():
        return normalized
    prompt_text = normalized.replace("<ImageHere>", "").strip()
    return f"USER: <ImageHere> {prompt_text} ASSISTANT:"
