"""把 PDF/Word 批量提取为 markdown，写入 processed_dir。

用法：
    python -m src.extract run                 # 使用默认配置
    python -m src.extract run --raw-dir X --out-dir Y
    RAG_CONFIG=configs/aliyun.yaml python -m src.extract run
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import hashlib
import json
import re

import typer
from tqdm import tqdm
from pypdf import PdfReader
from docx import Document

from src.config import load_config, default_config_path

app = typer.Typer(add_completion=False)


def _clean(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_pdf(path: Path) -> Optional[str]:
    try:
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            pages.append(t)
        return _clean("\n\n".join(pages))
    except Exception as e:
        print(f"[skip PDF] {path.name}: {e}")
        return None


def extract_docx(path: Path) -> Optional[str]:
    try:
        doc = Document(str(path))
        paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        tables_text = []
        for tb in doc.tables:
            for row in tb.rows:
                cells = [c.text.strip() for c in row.cells if c.text.strip()]
                if cells:
                    tables_text.append(" | ".join(cells))
        combined = "\n".join(paras)
        if tables_text:
            combined += "\n\n" + "\n".join(tables_text)
        return _clean(combined)
    except Exception as e:
        print(f"[skip DOCX] {path.name}: {e}")
        return None


def _doc_id(rel_path: Path) -> str:
    return hashlib.md5(str(rel_path).encode("utf-8")).hexdigest()[:16]


@app.command()
def run(
    config: str = typer.Option(None, help="Config file path"),
    raw_dir: Optional[str] = typer.Option(None),
    out_dir: Optional[str] = typer.Option(None),
    min_chars: int = typer.Option(100, help="Skip docs shorter than this"),
    overwrite: bool = typer.Option(False, help="Re-extract even if output exists"),
):
    cfg = load_config(config or default_config_path())
    raw = Path(raw_dir or cfg.paths.raw_dir)
    out = Path(out_dir or cfg.paths.processed_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not raw.exists():
        raise typer.BadParameter(f"raw_dir not found: {raw}")

    files: list[Path] = []
    for pattern in ("*.pdf", "*.PDF", "*.docx", "*.DOCX", "*.doc"):
        files.extend(raw.rglob(pattern))
    print(f"Found {len(files)} files under {raw}")

    ok = skipped = failed = 0
    for f in tqdm(files, desc="extracting"):
        rel = f.relative_to(raw)
        out_path = out / rel.with_suffix(".md")
        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        if f.suffix.lower() == ".pdf":
            text = extract_pdf(f)
        else:
            text = extract_docx(f)

        if not text or len(text) < min_chars:
            failed += 1
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "doc_id": _doc_id(rel),
            "source_relpath": str(rel),
            "source_ext": f.suffix.lower().lstrip("."),
            "size_bytes": f.stat().st_size,
        }
        front_matter = "---\n" + "\n".join(f"{k}: {json.dumps(v, ensure_ascii=False)}"
                                            for k, v in meta.items()) + "\n---\n\n"
        out_path.write_text(front_matter + text, encoding="utf-8")
        ok += 1

    print(f"Extract done. ok={ok}, skipped_existing={skipped}, failed={failed}")


if __name__ == "__main__":
    app()
