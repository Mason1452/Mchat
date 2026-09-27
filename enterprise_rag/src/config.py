"""统一的配置加载器：从 YAML 读取 + 环境变量覆盖 + 敏感字段从 env 拉取。

用法：
    from src.config import load_config
    cfg = load_config("configs/local.yaml")
    print(cfg.llm.provider)
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Optional
import yaml
from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    name: str
    env: str


class PathsConfig(BaseModel):
    raw_dir: str
    processed_dir: str
    index_dir: str


class ChunkingConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 80
    min_chunk_chars: int = 40


class EmbeddingConfig(BaseModel):
    provider: str
    model: str
    device: str = "cpu"
    batch_size: int = 32
    dim: int = 512
    # 限制 torch CPU 侧线程数，避免建索引/CPU 回退时把性能核吃满导致整机卡顿。None 表示不主动设置。
    max_cpu_threads: Optional[int] = None
    # MPS/CUDA 上是否用 fp16 权重：可把权重常驻内存减半，精度对检索几乎无影响。
    fp16: bool = True
    api_key_env: Optional[str] = None
    api_key: Optional[str] = None


class VectorStoreConfig(BaseModel):
    provider: str
    collection: str
    persist_dir: Optional[str] = None
    # numpy 本地库需要知道向量维度来初始化矩阵；chroma/dashvector 不使用该字段。
    dim: Optional[int] = None
    endpoint_env: Optional[str] = None
    endpoint: Optional[str] = None
    api_key_env: Optional[str] = None
    api_key: Optional[str] = None
    metric: str = "cosine"


class LLMConfig(BaseModel):
    provider: str
    model: str
    host: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout_s: int = 120
    keep_alive: Optional[str] = None
    # provider-specific runtime tuning，例如 Ollama 的 num_ctx / num_batch / num_thread / flash_attention。
    # 直接透传给底层 API 的 options 字段，键名沿用官方文档，避免这里维护一层同义词。
    runtime_options: dict[str, Any] = Field(default_factory=dict)
    api_key_env: Optional[str] = None
    api_key: Optional[str] = None


class RetrieverConfig(BaseModel):
    top_k: int = 6
    score_threshold: float = 0.3
    fetch_k: int = 0
    dedup_by_source: bool = False
    # 文档级内容聚类：同一材料被多目录重复归档时，按全文 shingle containment 归为一簇，
    # 检索按簇去重；命中簇内任一物理副本即算命中。
    cluster_duplicates: bool = False
    dup_threshold: float = 0.72


class WarmupConfig(BaseModel):
    # on_startup 是总开关；下面 embedding/llm 可分别关闭，只预热真正需要常驻的组件以降低空闲基线内存。
    on_startup: bool = True
    embedding: bool = True
    llm: bool = True
    text: str = "知识"


class CacheConfig(BaseModel):
    """查询缓存。企业场景同一问题常被反复提问，命中时跳过本地 embedding/向量检索，
    非流式答案缓存还能跳过 LLM 调用，显著降延迟和费用。

    - retrieve/answer 分别可独立开关；容量为 LRU 条目上限，超出后最久未用项被淘汰。
    - 缓存键包含归一化后的问题与实际生效的 top_k/score_threshold，避免不同参数串结果。
    - 数据/索引更新后重启服务即清空（进程内缓存，不落盘）。
    """
    retrieve: bool = True
    answer: bool = True
    retrieve_size: int = 2048
    answer_size: int = 512


class PromptConfig(BaseModel):
    system: str
    user_template: str


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


class Config(BaseModel):
    app: AppConfig
    paths: PathsConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    vectorstore: VectorStoreConfig
    llm: LLMConfig
    retriever: RetrieverConfig
    warmup: WarmupConfig = Field(default_factory=WarmupConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    prompt: PromptConfig
    server: ServerConfig


def _resolve_env(section: dict[str, Any]) -> dict[str, Any]:
    """把 xxx_env 字段解析成对应环境变量的值，写回 xxx。"""
    resolved = dict(section)
    for key in list(resolved.keys()):
        if key.endswith("_env") and isinstance(resolved[key], str):
            target = key[:-4]
            env_val = os.environ.get(resolved[key])
            if env_val:
                resolved[target] = env_val
    return resolved


def load_config(path: str | Path) -> Config:
    """从 YAML 加载配置，并把 *_env 引用的环境变量注入到对应字段。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    for key in ("embedding", "vectorstore", "llm"):
        if key in raw:
            raw[key] = _resolve_env(raw[key])

    return Config(**raw)


def default_config_path() -> str:
    """允许通过 RAG_CONFIG 环境变量覆盖，缺省为 configs/local.yaml。"""
    return os.environ.get("RAG_CONFIG", "configs/local.yaml")
