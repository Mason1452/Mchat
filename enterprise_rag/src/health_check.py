"""RAG 系统健康体检脚本（只读，不修改任何数据）。

用途：
  1. 扫 data/raw：文档数量、按扩展名分布、总大小
  2. 扫 data/processed：抽取产物是否与 raw 匹配
  3. 检 index：向量文件是否存在、向量数、维度是否与配置一致
  4. 校验当前配置：LLM/Embedding provider、API Key 是否已注入
  5. 输出可执行建议：需不需要 extract / build_index / 或已就绪

用法：
    RAG_CONFIG=configs/deepseek.yaml python -m src.health_check
    python -m src.health_check --config configs/local.yaml
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer

from src.config import Config, default_config_path, load_config


app = typer.Typer(add_completion=False, help="RAG 健康体检")


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def _scan_raw(raw_dir: Path) -> dict:
    if not raw_dir.exists():
        return {"exists": False, "path": str(raw_dir)}
    exts = {}
    total_bytes = 0
    docs = []
    supported_exts = {"pdf", "docx", "doc"}
    doc_count = 0
    for p in raw_dir.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower().lstrip(".")
        exts[ext] = exts.get(ext, 0) + 1
        total_bytes += p.stat().st_size
        docs.append(p)
        if ext in supported_exts:
            doc_count += 1
    return {
        "exists": True,
        "path": str(raw_dir),
        "file_count": len(docs),
        "doc_count": doc_count,
        "by_ext": exts,
        "total_bytes": total_bytes,
    }


def _scan_processed(processed_dir: Path) -> dict:
    if not processed_dir.exists():
        return {"exists": False, "path": str(processed_dir), "md_count": 0}
    md_files = list(processed_dir.rglob("*.md"))
    total_bytes = sum(p.stat().st_size for p in md_files if p.is_file())
    return {
        "exists": True,
        "path": str(processed_dir),
        "md_count": len(md_files),
        "total_bytes": total_bytes,
    }


def _scan_numpy_index(persist_dir: Path) -> dict:
    vec_path = persist_dir / "vectors.npy"
    meta_path = persist_dir / "docs.jsonl"
    info = {
        "path": str(persist_dir),
        "vec_exists": vec_path.exists(),
        "meta_exists": meta_path.exists(),
        "vec_count": 0,
        "vec_dim": None,
        "meta_count": 0,
        "sources": {},
        "vec_bytes": 0,
        "meta_bytes": 0,
    }
    if vec_path.exists():
        # 只读头部，避免加载整个矩阵到内存。
        try:
            import numpy as np
            arr = np.load(vec_path, mmap_mode="r", allow_pickle=False)
            info["vec_count"] = int(arr.shape[0])
            info["vec_dim"] = int(arr.shape[1]) if arr.ndim == 2 else None
            info["vec_bytes"] = vec_path.stat().st_size
        except Exception as e:
            info["vec_error"] = repr(e)
    if meta_path.exists():
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    info["meta_count"] += 1
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    src = (obj.get("metadata") or {}).get("source") or "<unknown>"
                    info["sources"][src] = info["sources"].get(src, 0) + 1
            info["meta_bytes"] = meta_path.stat().st_size
        except Exception as e:
            info["meta_error"] = repr(e)
    return info


def _check_keys(cfg: Config) -> dict:
    """检查 provider 所需的环境变量是否已注入到 cfg（不打印真值）。"""
    result = {"embedding": {}, "llm": {}, "vectorstore": {}}
    # embedding
    if cfg.embedding.api_key_env:
        result["embedding"] = {
            "env": cfg.embedding.api_key_env,
            "present": bool(cfg.embedding.api_key),
        }
    # llm
    if cfg.llm.api_key_env:
        result["llm"] = {
            "env": cfg.llm.api_key_env,
            "present": bool(cfg.llm.api_key),
        }
    # vectorstore (dashvector 等云侧)
    if cfg.vectorstore.api_key_env:
        result["vectorstore"] = {
            "env": cfg.vectorstore.api_key_env,
            "present": bool(cfg.vectorstore.api_key),
        }
    return result


def _advise(raw: dict, processed: dict, index: dict, cfg: Config, keys: dict) -> list[str]:
    tips: list[str] = []
    # 数据侧
    if not raw["exists"] or raw.get("file_count", 0) == 0:
        tips.append(
            f"[数据] {raw['path']} 为空，请先把 PDF/Word 复盘文档放进去。"
        )
    # 抽取侧：只统计管线支持的 PDF/Word，xlsx/图片等不计入
    if raw.get("doc_count", 0) > 0 and processed.get("md_count", 0) < raw["doc_count"]:
        tips.append(
            f"[抽取] processed 目录 md 数({processed.get('md_count', 0)}) < 可抽取文档({raw['doc_count']})，"
            f"请运行: python -m src.extract"
        )
    # 索引侧
    if cfg.vectorstore.provider == "numpy":
        if not index["vec_exists"] or index["vec_count"] == 0:
            tips.append(
                f"[索引] 未发现有效向量库 ({index['path']})，请运行: "
                f"RAG_CONFIG=<配置> python -m src.build_index"
            )
        else:
            expected_dim = cfg.vectorstore.dim or cfg.embedding.dim
            if index["vec_dim"] and expected_dim and index["vec_dim"] != expected_dim:
                tips.append(
                    f"[索引] 向量维度不匹配：库中={index['vec_dim']} vs 配置期望={expected_dim}。"
                    f"若最近换过 embedding 模型，需先删掉 {index['path']} 再重建。"
                )
            if index["vec_count"] != index["meta_count"]:
                tips.append(
                    f"[索引] 向量数({index['vec_count']}) 与 docs.jsonl 行数({index['meta_count']}) 不一致，"
                    f"可能上次 build_index 中途中断。建议清空 {index['path']} 后重建。"
                )
    # LLM 侧
    if cfg.llm.provider == "deepseek":
        if not keys["llm"].get("present"):
            tips.append(
                f"[LLM] DeepSeek 配置里未注入 API Key（env={keys['llm'].get('env')}），"
                f"设置后再启动：export DEEPSEEK_API_KEY=sk-xxx"
            )
    elif cfg.llm.provider == "dashscope":
        if not keys["llm"].get("present"):
            tips.append(
                f"[LLM] DashScope 未注入 API Key（env={keys['llm'].get('env')}）"
            )
    # 上云建议：如果当前是 deepseek/aliyun 云配置但 vector_store 是本地 numpy
    if cfg.app.env in {"deepseek", "aliyun"} and cfg.vectorstore.provider == "numpy":
        tips.append(
            "[部署] 当前配置面向云端 LLM 但向量库仍是本地 numpy —— 单机部署到阿里云 ECS 没问题；"
            "若需要多实例横向扩展，考虑切到 dashvector 或 chroma-server。"
        )
    if not tips:
        tips.append("[OK] 数据/索引/配置齐全，可以直接 `python -m src.query` 或 `python -m src.serve` 开跑。")
    return tips


def _print_report(cfg_path: str, cfg: Config, raw: dict, processed: dict,
                  index: dict, keys: dict, tips: list[str], as_json: bool):
    if as_json:
        typer.echo(json.dumps({
            "config": cfg_path,
            "app": cfg.app.model_dump(),
            "raw": raw,
            "processed": processed,
            "index": index,
            "keys": keys,
            "advice": tips,
        }, ensure_ascii=False, indent=2))
        return
    typer.echo(f"=== RAG 健康体检报告 ===")
    typer.echo(f"配置文件      : {cfg_path}")
    typer.echo(f"应用环境      : {cfg.app.name} / {cfg.app.env}")
    typer.echo("")
    typer.echo(f"[data/raw]    : {raw.get('path')}  存在={raw.get('exists')}  "
               f"文件数={raw.get('file_count', 0)}  "
               f"总大小={_fmt_bytes(raw.get('total_bytes', 0))}")
    if raw.get("by_ext"):
        typer.echo(f"              分布: {raw['by_ext']}")
    typer.echo(f"[processed]   : {processed.get('path')}  存在={processed.get('exists')}  "
               f"md={processed.get('md_count', 0)}  "
               f"总大小={_fmt_bytes(processed.get('total_bytes', 0))}")
    typer.echo("")
    typer.echo(f"[vectorstore] : {cfg.vectorstore.provider} @ {index.get('path')}")
    typer.echo(f"              vectors.npy: exists={index['vec_exists']}  "
               f"rows={index['vec_count']}  dim={index['vec_dim']}  "
               f"size={_fmt_bytes(index.get('vec_bytes', 0))}")
    typer.echo(f"              docs.jsonl : exists={index['meta_exists']}  "
               f"lines={index['meta_count']}  "
               f"size={_fmt_bytes(index.get('meta_bytes', 0))}")
    if index.get("sources"):
        top_srcs = sorted(index["sources"].items(), key=lambda x: -x[1])[:5]
        typer.echo(f"              前 5 来源: {top_srcs}")
    typer.echo("")
    typer.echo(f"[embedding]   : {cfg.embedding.provider} / {cfg.embedding.model} "
               f"(device={cfg.embedding.device}, dim={cfg.embedding.dim}, fp16={cfg.embedding.fp16})")
    typer.echo(f"[llm]         : {cfg.llm.provider} / {cfg.llm.model} "
               f"(host={cfg.llm.host or '<default>'}, max_tokens={cfg.llm.max_tokens})")
    for k, v in keys.items():
        if v:
            state = "OK" if v.get("present") else "MISSING"
            typer.echo(f"              key.{k}: {v.get('env')} -> {state}")
    typer.echo("")
    typer.echo("[建议]")
    for t in tips:
        typer.echo(f"  - {t}")


@app.command()
def main(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="配置文件路径"),
    json_out: bool = typer.Option(False, "--json", help="以 JSON 输出，方便 CI/脚本解析"),
):
    cfg_path = config or default_config_path()
    cfg = load_config(cfg_path)
    raw = _scan_raw(Path(cfg.paths.raw_dir))
    processed = _scan_processed(Path(cfg.paths.processed_dir))
    index_dir = cfg.vectorstore.persist_dir or cfg.paths.index_dir
    index = _scan_numpy_index(Path(index_dir))
    keys = _check_keys(cfg)
    tips = _advise(raw, processed, index, cfg, keys)
    _print_report(cfg_path, cfg, raw, processed, index, keys, tips, json_out)
    # 有严重 tip 时用非 0 退出码，方便 CI 用
    bad = any(t.startswith("[数据]") or t.startswith("[索引]") or t.startswith("[LLM]") for t in tips)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    app()
