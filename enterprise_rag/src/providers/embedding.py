"""Embedding provider 抽象层。

支持：
- bge_local: 本地 sentence-transformers 跑 BGE 模型（M2 Pro 用 MPS）
- dashscope: 阿里云百炼 embedding API（text-embedding-v3）
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence
import os

import numpy as np

from src.config import EmbeddingConfig


class BaseEmbedder(ABC):
    dim: int

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """把一批文本编码成归一化向量，shape=(n, dim)。本地实现直接返回 float32 numpy，
        避免 .tolist() 把整批向量转成 Python float 对象造成的瞬时内存放大。"""


def _apply_cpu_thread_limit(n: int | None) -> None:
    if not n:
        return
    import torch
    torch.set_num_threads(int(n))
    # interop 线程必须在任何并行工作开始前设置，且只能设一次；放在 try 里避免重复设置抛错。
    try:
        torch.set_num_interop_threads(max(1, min(int(n), 2)))
    except RuntimeError:
        pass


class BGELocalEmbedder(BaseEmbedder):
    def __init__(self, cfg: EmbeddingConfig):
        import torch
        from sentence_transformers import SentenceTransformer

        _apply_cpu_thread_limit(cfg.max_cpu_threads)

        kwargs = {}
        # fp16 只在加速设备上开启：MPS/CUDA 上权重常驻减半且推理更快；CPU 上 fp16 算子支持差，保持 fp32。
        if cfg.fp16 and cfg.device in ("mps", "cuda"):
            kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
        self.model = SentenceTransformer(cfg.model, device=cfg.device, **kwargs)
        self.batch_size = cfg.batch_size
        self.dim = cfg.dim or self.model.get_sentence_embedding_dimension()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        embs = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        # 统一 float32 连续数组：Chroma / numpy 库 / 余弦计算都按 float32 处理。
        return np.ascontiguousarray(embs, dtype=np.float32)


class DashScopeEmbedder(BaseEmbedder):
    def __init__(self, cfg: EmbeddingConfig):
        import dashscope
        api_key = cfg.api_key or os.environ.get(cfg.api_key_env or "DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DashScope API key not found in env")
        dashscope.api_key = api_key
        self._dashscope = dashscope
        self.model = cfg.model
        self.batch_size = min(cfg.batch_size, 25)
        self.dim = cfg.dim

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = list(texts[i:i + self.batch_size])
            resp = self._dashscope.TextEmbedding.call(
                model=self.model,
                input=batch,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"DashScope embedding failed: {resp.message}")
            for item in resp.output["embeddings"]:
                out.append(item["embedding"])
        return np.asarray(out, dtype=np.float32) if out else np.empty((0, self.dim), dtype=np.float32)


def build_embedder(cfg: EmbeddingConfig) -> BaseEmbedder:
    provider = cfg.provider.lower()
    if provider == "bge_local":
        return BGELocalEmbedder(cfg)
    if provider == "dashscope":
        return DashScopeEmbedder(cfg)
    raise ValueError(f"Unknown embedding provider: {cfg.provider}")
