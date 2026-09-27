"""基于 JSONL 测试集，跑完整 RAG 流程并输出：
- 检索命中率（expected_sources 是否被检索到）
- 关键词覆盖率（expected_keywords 是否出现在回答中）
- 每个问题的 hits 和 answer 明细

用法：
    python -m eval.run_eval run                       # 使用默认配置
    python -m eval.run_eval run --testset eval/xxx.jsonl
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import typer

from src.config import load_config, default_config_path
from src.query import RAGPipeline

app = typer.Typer(add_completion=False)


def _match_keywords(answer: str, keywords: list[str]) -> tuple[int, int]:
    if not keywords:
        return 0, 0
    hit = sum(1 for k in keywords if k and k in answer)
    return hit, len(keywords)


def _match_sources(hit_sources: list[str], expected: list[str]) -> tuple[int, int]:
    if not expected:
        return 0, 0
    got = set(hit_sources)
    hit = sum(1 for e in expected if any(e in g for g in got))
    return hit, len(expected)


def _match_sources_aliased(
    pipe,
    hit_sources: list[str],
    expected: list[str],
) -> tuple[int, int]:
    """同内容文档归簇后，命中簇内任一副本即算命中对应期望源。"""
    if not expected:
        return 0, 0
    aliases = pipe.source_aliases()
    if not aliases:
        return _match_sources(hit_sources, expected)

    def cluster_of(src: str) -> str:
        for key, members in aliases.items():
            if src in members:
                return key
        return src

    got_clusters = {cluster_of(s) for s in hit_sources}
    hit = sum(1 for e in expected if cluster_of(e) in got_clusters)
    return hit, len(expected)


@app.command()
def run(
    testset: str = typer.Option("eval/testset.jsonl", help="JSONL test file"),
    config: Optional[str] = typer.Option(None, help="Config file path"),
    output: str = typer.Option("eval/eval_report.jsonl", help="Detailed report output"),
):
    cfg = load_config(config or default_config_path())
    pipe = RAGPipeline(cfg)

    tset_path = Path(testset)
    if not tset_path.exists():
        raise typer.BadParameter(f"testset not found: {tset_path}")
    cases = [json.loads(l) for l in tset_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Loaded {len(cases)} test cases from {tset_path}")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kw_hit = kw_total = 0
    src_hit = src_total = 0
    with out_path.open("w", encoding="utf-8") as f:
        for i, case in enumerate(cases, 1):
            q = case["question"]
            exp_kw = case.get("expected_keywords", [])
            exp_src = case.get("expected_sources", [])
            res = pipe.answer(q)
            hit_sources = [h.metadata.get("source", h.id) for h in res.hits]

            k_hit, k_total = _match_keywords(res.answer, exp_kw)
            s_hit, s_total = _match_sources_aliased(pipe, hit_sources, exp_src)
            kw_hit += k_hit; kw_total += k_total
            src_hit += s_hit; src_total += s_total

            rec = {
                "idx": i,
                "question": q,
                "answer": res.answer,
                "hits": [
                    {"source": h.metadata.get("source", h.id), "score": h.score,
                     "text": h.text[:200]}
                    for h in res.hits
                ],
                "keyword_hit": f"{k_hit}/{k_total}",
                "source_hit": f"{s_hit}/{s_total}",
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[{i}/{len(cases)}] kw={k_hit}/{k_total} src={s_hit}/{s_total}")

    print("=" * 50)
    print(f"Cases: {len(cases)}")
    if kw_total:
        print(f"Keyword coverage: {kw_hit}/{kw_total} = {kw_hit / kw_total:.1%}")
    if src_total:
        print(f"Source recall:    {src_hit}/{src_total} = {src_hit / src_total:.1%}")
    print(f"Detail written to: {out_path}")


if __name__ == "__main__":
    app()
