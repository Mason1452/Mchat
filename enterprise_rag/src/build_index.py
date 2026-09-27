"""切段 + embedding + 写入向量库。支持断点续传（基于本地 progress.json）。

用法：
    python -m src.build_index run
    python -m src.build_index run --limit 100      # 只处理前 100 个文件（用于快速验证）
    python -m src.build_index run --force          # 忽略 progress.json 从头开始
    RAG_CONFIG=configs/aliyun.yaml python -m src.build_index run
"""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import hashlib
import json
import re

import typer
from tqdm import tqdm

from src.config import load_config, default_config_path
from src.providers.embedding import build_embedder
from src.providers.vectorstore import build_vectorstore, Doc

app = typer.Typer(add_completion=False)


_FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n\n?", re.DOTALL)


def _parse_front_matter(text: str) -> tuple[dict, str]:
    m = _FRONT_MATTER_RE.match(text)
    if not m:
        return {}, text
    meta_raw = m.group(1)
    meta: dict = {}
    for line in meta_raw.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        v = v.strip()
        try:
            meta[k.strip()] = json.loads(v)
        except Exception:
            meta[k.strip()] = v
    return meta, text[m.end():]


def _split_text(text: str, chunk_size: int, chunk_overlap: int, min_chunk: int) -> list[str]:
    """基于中文句号/换行的启发式切分，保证 chunk_size 上限并保留 overlap。"""
    if len(text) <= chunk_size:
        return [text] if len(text) >= min_chunk else []

    sentences = re.split(r"(?<=[。！？!?\n])\s*", text)
    chunks: list[str] = []
    buf = ""
    for s in sentences:
        if not s:
            continue
        if len(buf) + len(s) <= chunk_size:
            buf += s
        else:
            if len(buf) >= min_chunk:
                chunks.append(buf)
            if chunk_overlap and len(buf) > chunk_overlap:
                buf = buf[-chunk_overlap:] + s
            else:
                buf = s
            while len(buf) > chunk_size:
                head, buf = buf[:chunk_size], buf[chunk_size - chunk_overlap:]
                if len(head) >= min_chunk:
                    chunks.append(head)
    if len(buf) >= min_chunk:
        chunks.append(buf)
    return chunks


def _chunk_id(doc_id: str, idx: int) -> str:
    return f"{doc_id}_{idx:04d}"


def _load_progress(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        return set(json.loads(path.read_text(encoding="utf-8")).get("done", []))
    except Exception:
        return set()


def _save_progress(path: Path, done: set[str]) -> None:
    path.write_text(json.dumps({"done": sorted(done)}, ensure_ascii=False), encoding="utf-8")


def _iter_files(processed_dir: Path, limit: int | None) -> list[Path]:
    files = sorted(processed_dir.rglob("*.md"))
    if limit:
        files = files[:limit]
    return files


@app.command()
def run(
    config: str = typer.Option(None, help="Config file path"),
    limit: int | None = typer.Option(None, help="Process only first N markdown files"),
    force: bool = typer.Option(False, help="Ignore progress.json and start over"),
    batch: int = typer.Option(64, help="How many chunks to upsert per batch"),
):
    cfg = load_config(config or default_config_path())
    processed = Path(cfg.paths.processed_dir)
    index_dir = Path(cfg.paths.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    progress_path = index_dir / "progress.json"

    files = _iter_files(processed, limit)
    print(f"Total markdown files: {len(files)}")
    if not files:
        return

    done = set() if force else _load_progress(progress_path)
    if done:
        print(f"Resume: {len(done)} files already indexed, skipping them")

    print(f"Loading embedder: {cfg.embedding.provider} / {cfg.embedding.model}")
    embedder = build_embedder(cfg.embedding)
    print(f"Loading vectorstore: {cfg.vectorstore.provider} / {cfg.vectorstore.collection}")
    store = build_vectorstore(cfg.vectorstore)
    print(f"Existing vector count: {store.count()}")

    pending_docs: list[Doc] = []
    pending_texts: list[str] = []
    stat_files = stat_chunks = 0

    def _flush():
        nonlocal pending_docs, pending_texts, stat_chunks
        if not pending_texts:
            return
        embs = embedder.encode(pending_texts)
        for d, e in zip(pending_docs, embs):
            d.embedding = e
        store.upsert(pending_docs)
        stat_chunks += len(pending_docs)
        pending_docs, pending_texts = [], []
        # MPS/CUDA 的缓存分配器会保留已申请显存不主动归还；批量入库后手动清空，降低统一内存占用。
        try:
            import torch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    pbar = tqdm(files, desc="indexing")
    for f in pbar:
        rel_key = str(f.relative_to(processed))
        if rel_key in done:
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        meta, body = _parse_front_matter(text)
        doc_id = meta.get("doc_id") or hashlib.md5(rel_key.encode()).hexdigest()[:16]

        chunks = _split_text(
            body,
            chunk_size=cfg.chunking.chunk_size,
            chunk_overlap=cfg.chunking.chunk_overlap,
            min_chunk=cfg.chunking.min_chunk_chars,
        )
        for i, ch in enumerate(chunks):
            pending_docs.append(Doc(
                id=_chunk_id(doc_id, i),
                text=ch,
                embedding=[],
                metadata={
                    "doc_id": doc_id,
                    "source": meta.get("source_relpath", rel_key),
                    "chunk_index": i,
                },
            ))
            pending_texts.append(ch)
            if len(pending_texts) >= batch:
                _flush()
                pbar.set_postfix(files=stat_files, chunks=stat_chunks)

        stat_files += 1
        done.add(rel_key)
        if stat_files % 50 == 0:
            _flush()
            _save_progress(progress_path, done)

    _flush()
    _save_progress(progress_path, done)
    print(f"Index build done. files={stat_files}, new_chunks={stat_chunks}, total_vectors={store.count()}")


if __name__ == "__main__":
    app()
