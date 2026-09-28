"""RAG 查询入口，同时提供四种 API：

1) 纯检索 CLI（无需 LLM/Key）：python -m src.query search "你的问题"
2) 编程接口：RAGPipeline.answer(question) -> AnswerResult
3) CLI 单次：python -m src.query ask "你的问题"
4) CLI 交互（流式）：python -m src.query chat
"""
from __future__ import annotations
import json
import logging
import threading
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import typer

from src.config import Config, load_config, default_config_path, source_cluster_path
from src.dedup import cluster_sources
from src.providers.embedding import build_embedder, BaseEmbedder
from src.providers.vectorstore import build_vectorstore, BaseVectorStore, Hit
from src.providers.llm import build_llm, BaseLLM

app = typer.Typer(add_completion=False)
logger = logging.getLogger("enterprise_rag.query")


@dataclass
class AnswerResult:
    question: str
    answer: str
    hits: list[Hit] = field(default_factory=list)


class _LRU:
    """线程安全的有界 LRU：get 命中刷新为最近使用，put 超容量时淘汰最久未用项。"""

    def __init__(self, capacity: int):
        self._cap = max(1, int(capacity))
        self._data: "OrderedDict[object, object]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key, value) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._cap:
                self._data.popitem(last=False)


def _norm_question(question: str) -> str:
    return " ".join(question.split())


class RAGPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.embedder: BaseEmbedder = build_embedder(cfg.embedding)
        self.store: BaseVectorStore = build_vectorstore(cfg.vectorstore)
        self.llm: BaseLLM = build_llm(cfg.llm)
        self._warmed = False
        self._retrieve_cache = _LRU(cfg.cache.retrieve_size) if cfg.cache.retrieve else None
        self._answer_cache = _LRU(cfg.cache.answer_size) if cfg.cache.answer else None
        self._source_cluster: Optional[dict[str, str]] = None

    def warmup(self) -> None:
        """预热。按 warmup.embedding / warmup.llm 子开关分别执行，只拉起需要常驻的组件。
        两者互不依赖，并行触发以缩短启动等待。"""
        if self._warmed:
            return

        tasks = []
        if self.cfg.warmup.embedding:
            tasks.append(lambda: self.embedder.encode([self.cfg.warmup.text]))
        if self.cfg.warmup.llm:
            tasks.append(self.llm.warmup)

        if tasks:
            with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
                futs = [pool.submit(t) for t in tasks]
                for f in futs:
                    f.result()
        self._warmed = True

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> list[Hit]:
        eff_top_k = top_k if top_k is not None else self.cfg.retriever.top_k
        eff_thr = (
            score_threshold if score_threshold is not None
            else self.cfg.retriever.score_threshold
        )
        cluster_on = self.cfg.retriever.cluster_duplicates
        if cluster_on:
            self._ensure_source_cluster()
        cluster_map = self._source_cluster or {}

        eff_fetch_k = self.cfg.retriever.fetch_k or (
            eff_top_k * 3 if (self.cfg.retriever.dedup_by_source or cluster_on) else eff_top_k
        )

        key = (
            _norm_question(question),
            eff_top_k,
            round(float(eff_thr), 6),
            self.cfg.retriever.dedup_by_source,
            eff_fetch_k,
            cluster_on,
        )
        if self._retrieve_cache is not None:
            cached = self._retrieve_cache.get(key)
            if cached is not None:
                return list(cached)

        emb = self.embedder.encode([question])[0]

        if cluster_on and cluster_map:
            # 重复归档多时，固定 fetch_k 折叠后可能不足 top_k 个簇，按需扩大候选池。
            hits: list[Hit] = []
            need = eff_fetch_k
            total_n = self.store.count()
            while True:
                raw_hits = self.store.query(
                    embedding=emb, top_k=need, score_threshold=eff_thr,
                )
                seen: set = set()
                hits = []
                for h in raw_hits:
                    s = cluster_map.get(h.metadata.get("source", h.id), h.id)
                    if s in seen:
                        continue
                    seen.add(s)
                    hits.append(h)
                if len(hits) >= eff_top_k or need >= total_n or not raw_hits:
                    break
                need = min(total_n, max(need * 2, eff_top_k * 6))
            hits = hits[:eff_top_k]
        else:
            raw_hits = self.store.query(
                embedding=emb,
                top_k=eff_fetch_k,
                score_threshold=eff_thr,
            )
            if self.cfg.retriever.dedup_by_source:
                seen = set()
                hits = []
                for h in raw_hits:
                    s = h.metadata.get("source", h.id)
                    if s in seen:
                        continue
                    seen.add(s)
                    hits.append(h)
                    if len(hits) >= eff_top_k:
                        break
            else:
                hits = raw_hits[:eff_top_k]

        if self._retrieve_cache is not None:
            self._retrieve_cache.put(key, list(hits))
        return hits

    def _ensure_source_cluster(self) -> None:
        if self._source_cluster is not None:
            return

        artifact = source_cluster_path(self.cfg)
        if artifact.exists():
            try:
                data = json.loads(artifact.read_text(encoding="utf-8"))
                mapping = data.get("clusters", {})
                if mapping:
                    logger.info("loaded source cluster artifact: %s (%d sources)", artifact, len(mapping))
                    self._source_cluster = mapping
                    return
            except Exception:
                logger.warning("failed to parse cluster artifact %s; falling back", artifact)

        try:
            records = self.store.get_all()
        except NotImplementedError:
            logger.warning("no cluster artifact and vectorstore cannot scan; clustering skipped")
            self._source_cluster = {}
            return

        texts: dict[str, str] = defaultdict(str)
        for r in records:
            texts[r.metadata.get("source", r.id)] += r.text
        self._source_cluster = cluster_sources(texts, self.cfg.retriever.dup_threshold)

    def source_aliases(self) -> dict[str, list[str]]:
        """cluster_key -> 簇内全部物理 source 路径，供评测/展示用。未开聚类时返回空。"""
        if self.cfg.retriever.cluster_duplicates:
            self._ensure_source_cluster()
        if not self._source_cluster:
            return {}
        out: dict[str, list[str]] = defaultdict(list)
        for src, key in self._source_cluster.items():
            out[key].append(src)
        return dict(out)

    def build_prompt(self, question: str, hits: list[Hit]) -> tuple[str, str]:
        if hits:
            context_blocks = []
            for i, h in enumerate(hits, 1):
                src = h.metadata.get("source", h.id)
                context_blocks.append(f"【片段 {i}｜来源: {src}｜相关度: {h.score:.2f}】\n{h.text}")
            context = "\n\n".join(context_blocks)
        else:
            context = "（无相关资料检索到）"

        user_prompt = self.cfg.prompt.user_template.format(
            context=context, question=question,
        )
        return self.cfg.prompt.system, user_prompt

    def answer(self, question: str, top_k: Optional[int] = None) -> AnswerResult:
        eff_top_k = top_k if top_k is not None else self.cfg.retriever.top_k
        akey = (_norm_question(question), eff_top_k)
        if self._answer_cache is not None:
            cached = self._answer_cache.get(akey)
            if cached is not None:
                return AnswerResult(question=question, answer=cached[0], hits=list(cached[1]))
        hits = self.retrieve(question, top_k=top_k)
        system, user = self.build_prompt(question, hits)
        text = self.llm.generate(system, user)
        if self._answer_cache is not None:
            self._answer_cache.put(akey, (text, list(hits)))
        return AnswerResult(question=question, answer=text, hits=hits)

    def stream_answer(self, question: str, top_k: Optional[int] = None):
        # 流式答案本身不缓存（生成过程要实时下发），但 retrieve 内部会复用检索缓存。
        hits = self.retrieve(question, top_k=top_k)
        system, user = self.build_prompt(question, hits)
        for piece in self.llm.stream(system, user):
            yield piece, hits


@app.command()
def search(
    question: str,
    config: Optional[str] = typer.Option(None, help="Config file path"),
    top_k: Optional[int] = typer.Option(None, help="返回片段数，默认用配置 retriever.top_k"),
    score_threshold: Optional[float] = typer.Option(None, help="相关度下限，默认用配置阈值"),
    snippet: int = typer.Option(160, help="文本模式下每片段打印字符数"),
    as_json: bool = typer.Option(False, "--json", help="以 JSON 输出，方便脚本/CI 解析"),
):
    """纯检索：只跑本地 embedding + 向量库，不初始化 LLM、无需任何 API Key。"""
    cfg = load_config(config or default_config_path())
    embedder = build_embedder(cfg.embedding)
    store = build_vectorstore(cfg.vectorstore)
    emb = embedder.encode([question])[0]
    hits = store.query(
        embedding=emb,
        top_k=top_k if top_k is not None else cfg.retriever.top_k,
        score_threshold=(
            score_threshold if score_threshold is not None
            else cfg.retriever.score_threshold
        ),
    )

    if as_json:
        typer.echo(json.dumps({
            "question": question,
            "count": len(hits),
            "hits": [
                {"id": h.id, "score": h.score,
                 "source": h.metadata.get("source", h.id),
                 "metadata": h.metadata, "text": h.text}
                for h in hits
            ],
        }, ensure_ascii=False, indent=2))
        return

    typer.echo("=" * 80)
    typer.echo(f"问: {question}   (命中 {len(hits)} 个片段，无需 Key)")
    typer.echo("=" * 80)
    if not hits:
        typer.echo("（没有片段通过相关度阈值，可尝试降低 --score-threshold）")
    for i, h in enumerate(hits, 1):
        src = h.metadata.get("source", h.id)
        typer.echo(f"[{i}] score={h.score:.3f}  {src}")
        typer.echo("    " + h.text[:snippet].replace("\n", " "))
    typer.echo("=" * 80)


@app.command()
def ask(
    question: str,
    config: Optional[str] = typer.Option(None, help="Config file path"),
    show_context: bool = typer.Option(False, help="Print retrieved context"),
):
    cfg = load_config(config or default_config_path())
    pipe = RAGPipeline(cfg)
    if cfg.warmup.on_startup:
        pipe.warmup()
    res = pipe.answer(question)

    if show_context:
        print("=" * 60)
        print("[Retrieved Context]")
        for i, h in enumerate(res.hits, 1):
            src = h.metadata.get("source", h.id)
            print(f"[{i}] score={h.score:.3f}  source={src}")
            print(h.text[:300].replace("\n", " "))
            print("-" * 40)
    print("=" * 60)
    print("[Answer]")
    print(res.answer)


@app.command()
def chat(
    config: Optional[str] = typer.Option(None, help="Config file path"),
):
    """交互式对话，回答以流式逐字打印（首字延迟最低的用法）。"""
    cfg = load_config(config or default_config_path())
    pipe = RAGPipeline(cfg)
    if cfg.warmup.on_startup:
        typer.echo("[warming up models...]")
        pipe.warmup()
    typer.echo("[ready] 输入问题开始对话，输入 exit 或 Ctrl+C 退出\n")

    while True:
        try:
            question = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in {"exit", "quit", ":q"}:
            break
        print("助手> ", end="", flush=True)
        for piece, hits in pipe.stream_answer(question):
            print(piece, end="", flush=True)
        sources = sorted({h.metadata.get("source", h.id) for h in hits})
        if sources:
            print(f"\n[来源: {', '.join(sources)}]")
        print()


if __name__ == "__main__":
    app()
