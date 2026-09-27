"""VectorStore provider 抽象层。

支持：
- chroma: 本地 ChromaDB（sqlite 存储）
- dashvector: 阿里云 DashVector（生产环境）

统一接口：upsert / query，向量都是已经归一化的 list[float]。
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence, Any, Optional
import json
import os

import numpy as np

from src.config import VectorStoreConfig


@dataclass
class Doc:
    id: str
    text: str
    embedding: Any  # np.ndarray 或 list[float]
    metadata: dict[str, Any]


@dataclass
class Hit:
    id: str
    text: str
    score: float
    metadata: dict[str, Any]


class BaseVectorStore(ABC):
    @abstractmethod
    def upsert(self, docs: Sequence[Doc]) -> None: ...

    @abstractmethod
    def query(
        self,
        embedding: list[float],
        top_k: int,
        score_threshold: float = 0.0,
    ) -> list[Hit]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def existing_ids(self) -> set[str]:
        """已存在的 id 集合，用于增量构建时去重。大集合下可以返回空集并把去重交给业务层。"""

    def get_all(self) -> list[Hit]:
        """返回库内全部记录，用于离线分析/文档聚类。默认未实现以保持向后兼容。"""
        raise NotImplementedError


class ChromaStore(BaseVectorStore):
    def __init__(self, cfg: VectorStoreConfig):
        import chromadb
        from chromadb.config import Settings
        persist_dir = cfg.persist_dir or "index/chroma_db"
        os.makedirs(persist_dir, exist_ok=True)
        # 关闭遥测：省去一个后台上报线程和出站连接，对本地隐私也更友好。
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self.collection = self.client.get_or_create_collection(
            name=cfg.collection,
            metadata={"hnsw:space": cfg.metric or "cosine"},
        )

    def upsert(self, docs):
        if not docs:
            return
        # Chroma 同时接受 list 与 numpy；这里不做任何 .tolist() 转换以省一次内存复制。
        self.collection.upsert(
            ids=[d.id for d in docs],
            embeddings=[np.asarray(d.embedding, dtype=np.float32) for d in docs],
            documents=[d.text for d in docs],
            metadatas=[d.metadata for d in docs],
        )

    def query(self, embedding, top_k, score_threshold=0.0):
        res = self.collection.query(
            query_embeddings=[np.asarray(embedding, dtype=np.float32)],
            n_results=top_k,
        )
        hits: list[Hit] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        dists = res.get("distances", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        for _id, _text, _dist, _meta in zip(ids, docs, dists, metas):
            score = 1.0 - float(_dist)
            if score < score_threshold:
                continue
            hits.append(Hit(id=_id, text=_text or "", score=score, metadata=_meta or {}))
        return hits

    def count(self):
        return self.collection.count()

    def existing_ids(self):
        try:
            all_res = self.collection.get(include=[])
            return set(all_res.get("ids", []))
        except Exception:
            return set()

    def get_all(self):
        res = self.collection.get(include=["documents", "metadatas"])
        ids = res.get("ids", [])
        docs = res.get("documents", [])
        metas = res.get("metadatas", [])
        return [
            Hit(id=ids[i], text=docs[i] or "", score=0.0, metadata=metas[i] or {})
            for i in range(len(ids))
        ]


class DashVectorStore(BaseVectorStore):
    def __init__(self, cfg: VectorStoreConfig):
        try:
            import dashvector
        except ImportError as e:
            raise RuntimeError(
                "dashvector SDK not installed. Run: pip install dashvector"
            ) from e
        endpoint = cfg.endpoint or os.environ.get(cfg.endpoint_env or "DASHVECTOR_ENDPOINT")
        api_key = cfg.api_key or os.environ.get(cfg.api_key_env or "DASHVECTOR_API_KEY")
        if not endpoint or not api_key:
            raise RuntimeError("DashVector endpoint / api_key not configured")
        self.client = dashvector.Client(api_key=api_key, endpoint=endpoint)
        self.collection_name = cfg.collection
        self.collection = self.client.get(cfg.collection)

    def upsert(self, docs):
        if not docs:
            return
        import dashvector
        batch = [
            dashvector.Doc(
                id=d.id,
                vector=np.asarray(d.embedding, dtype=np.float32).tolist(),
                fields={"text": d.text, **d.metadata},
            )
            for d in docs
        ]
        ret = self.collection.upsert(batch)
        if not ret:
            raise RuntimeError(f"DashVector upsert failed: {ret}")

    def query(self, embedding, top_k, score_threshold=0.0):
        ret = self.collection.query(
            vector=np.asarray(embedding, dtype=np.float32).tolist(),
            topk=top_k,
            output_fields=None,
        )
        if not ret:
            return []
        hits: list[Hit] = []
        for r in ret.output:
            score = float(r.score)
            if score < score_threshold:
                continue
            fields = dict(r.fields or {})
            text = fields.pop("text", "")
            hits.append(Hit(id=r.id, text=text, score=score, metadata=fields))
        return hits

    def count(self):
        stats = self.collection.stats()
        if stats:
            return int(getattr(stats.output, "total_doc_count", 0) or 0)
        return 0

    def existing_ids(self):
        return set()


class NumpyVectorStore(BaseVectorStore):
    """零第三方依赖的本地向量库：float32 矩阵 + JSONL 文档落盘，暴力余弦检索。

    适用数据量（512 维 float32）：
        1 万条约 20MB、5 万条约 100MB；单次检索是一次矩阵-向量乘（5 万 x512 < 30ms）。
    本地开发用它可以完全不导入 chromadb/onnxruntime，省下数百 MB 导入期内存。
    数据量再大或要生产时切回 chroma / dashvector 即可，接口一致、无需改业务代码。
    """

    def __init__(self, cfg: VectorStoreConfig):
        persist_dir = cfg.persist_dir or "index/numpy_db"
        os.makedirs(persist_dir, exist_ok=True)
        self._dir = persist_dir
        self._vec_path = os.path.join(persist_dir, "vectors.npy")
        self._meta_path = os.path.join(persist_dir, "docs.jsonl")
        self.collection = cfg.collection

        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metas: list[dict] = []
        self._vectors = np.empty((0, cfg.dim or 0), dtype=np.float32) if cfg.dim else np.empty((0, 0), dtype=np.float32)
        self._load()

    def _load(self) -> None:
        if os.path.exists(self._meta_path):
            with open(self._meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self._ids.append(rec["id"])
                    self._texts.append(rec.get("text", ""))
                    self._metas.append(rec.get("metadata", {}) or {})
        if os.path.exists(self._vec_path) and self._ids:
            self._vectors = np.load(self._vec_path, allow_pickle=False).astype(np.float32, copy=False)

    def persist(self) -> None:
        np.save(self._vec_path, self._vectors.astype(np.float32, copy=False), allow_pickle=False)
        with open(self._meta_path, "w", encoding="utf-8") as f:
            for i in range(len(self._ids)):
                f.write(json.dumps(
                    {"id": self._ids[i], "text": self._texts[i], "metadata": self._metas[i]},
                    ensure_ascii=False,
                ) + "\n")

    def upsert(self, docs):
        if not docs:
            return
        index = {id_: i for i, id_ in enumerate(self._ids)}
        new_rows = []
        for d in docs:
            vec = np.asarray(d.embedding, dtype=np.float32).reshape(1, -1)
            if d.id in index:
                i = index[d.id]
                self._vectors[i] = vec
                self._texts[i] = d.text
                self._metas[i] = d.metadata
            else:
                index[d.id] = len(self._ids)
                self._ids.append(d.id)
                self._texts.append(d.text)
                self._metas.append(d.metadata)
                new_rows.append(vec)
        if new_rows:
            block = np.concatenate(new_rows, axis=0)
            if self._vectors.size == 0:
                self._vectors = block
            else:
                # 维度以首个写入为准；正常流程下所有 embedding 同维。
                self._vectors = np.concatenate([self._vectors, block], axis=0)
        self.persist()

    def query(self, embedding, top_k, score_threshold=0.0):
        if not self._ids:
            return []
        q = np.asarray(embedding, dtype=np.float32).reshape(-1)
        # 向量均已 L2 归一化，点积即余弦相似度。
        scores = self._vectors @ q
        k = min(top_k, len(self._ids))
        # argpartition 取 top-k（O(n)），再在小集合内排序，避免对全量做 O(n log n)。
        cand = np.argpartition(scores, -k)[-k:]
        order = cand[np.argsort(scores[cand])[::-1]]
        hits: list[Hit] = []
        for i in order:
            s = float(scores[i])
            if s < score_threshold:
                continue
            hits.append(Hit(
                id=self._ids[i], text=self._texts[i], score=s, metadata=self._metas[i],
            ))
        return hits

    def count(self):
        return len(self._ids)

    def existing_ids(self):
        return set(self._ids)

    def get_all(self):
        return [
            Hit(id=self._ids[i], text=self._texts[i], score=0.0, metadata=self._metas[i])
            for i in range(len(self._ids))
        ]


def build_vectorstore(cfg: VectorStoreConfig) -> BaseVectorStore:
    provider = cfg.provider.lower()
    if provider == "chroma":
        return ChromaStore(cfg)
    if provider == "numpy":
        return NumpyVectorStore(cfg)
    if provider == "dashvector":
        return DashVectorStore(cfg)
    raise ValueError(f"Unknown vectorstore provider: {cfg.provider}")
