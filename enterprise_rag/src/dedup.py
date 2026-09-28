"""文档级去重原语：字符 n-gram shingle + containment 单链聚类。

企业资料里同一份文件常被复制进多个项目目录重复归档，文件名还会略改。
两文档只要较小一方的字符 3-gram 有 threshold 比例落在较大一方里，即判同源。

本模块不依赖 embedding / 向量库 / LLM，因此构建侧（离线产出簇工件）与
检索侧（加载工件或内存回退）都可安全复用。
"""
from __future__ import annotations
import logging
import re

logger = logging.getLogger("enterprise_rag.dedup")


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def char_shingles(text: str, n: int = 3) -> set[str]:
    t = compact(text)
    if len(t) < n:
        return {t} if t else set()
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def cluster_sources(texts: dict[str, str], threshold: float) -> dict[str, str]:
    """按全文 shingle containment 对 source 做单链聚类。

    texts: source_key -> 该文档全文（可由同一 source 的多个 chunk 拼出）。
    返回 source -> cluster_key（并查集根 source）映射。
    """
    sources = sorted(texts)
    sh = {s: char_shingles(texts[s]) for s in sources}

    parent = {s: s for s in sources}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    links = 0
    for i in range(len(sources)):
        A = sh[sources[i]]
        if not A:
            continue
        for j in range(i + 1, len(sources)):
            B = sh[sources[j]]
            if not B:
                continue
            inter = len(A & B)
            smaller = min(len(A), len(B))
            if smaller and inter / smaller >= threshold:
                ra, rb = find(sources[i]), find(sources[j])
                if ra != rb:
                    parent[rb] = ra
                    links += 1
    if links:
        logger.info("doc dedup: %d sources, %d duplicate links merged", len(sources), links)
    return {s: find(s) for s in sources}
