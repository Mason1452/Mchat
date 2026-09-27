"""FastAPI HTTP 服务。

启动：
    uvicorn src.serve:app --host 0.0.0.0 --port 8000
    或：python -m src.serve
    RAG_CONFIG=configs/aliyun.yaml uvicorn src.serve:app --port 8000
"""
from __future__ import annotations
import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.config import load_config, default_config_path
from src.query import RAGPipeline


logger = logging.getLogger("enterprise_rag.serve")


_cfg = load_config(default_config_path())
_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(_cfg)
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 服务启动时完成模型加载与 kernel 初始化，第一个用户请求不再承担冷启动开销
    if _cfg.warmup.on_startup:
        get_pipeline().warmup()
    yield


app = FastAPI(title=_cfg.app.name, version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cfg.server.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    stream: bool = False


class Citation(BaseModel):
    source: str
    score: float
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]


def _sse(event: str, data: dict) -> str:
    # SSE 规范：event / data 分行，最后必须有空行分隔消息。
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.get("/healthz")
def healthz():
    return {"status": "ok", "env": _cfg.app.env}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    pipe = get_pipeline()
    res = pipe.answer(req.question, top_k=req.top_k)
    citations = [
        Citation(
            source=h.metadata.get("source", h.id),
            score=h.score,
            text=h.text[:500],
        )
        for h in res.hits
    ]
    return AskResponse(question=res.question, answer=res.answer, citations=citations)


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    """端到端流式：SSE 事件流。

    事件序列:
      1) event: meta   —— 请求 id（rid），前端可用于日志追踪
      2) event: sources —— 命中片段（带来源、相关度、预览）
      3) event: delta   —— 生成增量文本，重复多次
      4) event: done    —— 结束（含累计答案长度、来源列表）
      5) event: error   —— 任一阶段异常，携带 rid + message

    SSE 保活:
      - 首字节前先发一条 `event: ping` 空数据，避免中间代理判定连接卡住
      - 若 LLM 生成时间较长，前端可自行加 EventSource 超时策略
    """
    pipe = get_pipeline()
    rid = uuid.uuid4().hex[:12]

    def _gen():
        # 首行 ping：立即刷出，让 HTTP 首字节尽早到达客户端 / 反代
        yield _sse("ping", {"rid": rid})
        yield _sse("meta", {"rid": rid, "question": req.question})

        try:
            gen = pipe.stream_answer(req.question, top_k=req.top_k)
            # 手动取第一次以拿到 hits；若一开始就抛错也能被统一捕获
            try:
                first_piece, hits = next(gen)
                got_first = True
            except StopIteration:
                # LLM 直接空输出：仍要发 sources / done，避免前端一直等
                hits, first_piece, got_first = [], "", False

            citations = [
                {
                    "source": h.metadata.get("source", h.id),
                    "score": round(h.score, 4),
                    "text": h.text[:500],
                }
                for h in hits
            ]
            yield _sse("sources", {"citations": citations})

            answer_len = 0
            if got_first and first_piece:
                answer_len += len(first_piece)
                yield _sse("delta", {"text": first_piece})
            for piece, _hits in gen:
                if piece:
                    answer_len += len(piece)
                    yield _sse("delta", {"text": piece})

            sources = sorted({h.metadata.get("source", h.id) for h in hits})
            yield _sse("done", {"rid": rid, "answer_len": answer_len, "sources": sources})
        except Exception as e:  # noqa: BLE001 — 网关兜底
            logger.exception("ask_stream failed rid=%s", rid)
            yield _sse("error", {"rid": rid, "message": f"{type(e).__name__}: {e}"})

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",  # 禁止代理压缩/缓冲 SSE
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",                   # Nginx: 禁止 buffer
            "X-Request-Id": rid,                          # 便于抓包/日志对齐
        },
    )


def main():
    import uvicorn
    uvicorn.run(app, host=_cfg.server.host, port=_cfg.server.port)


if __name__ == "__main__":
    main()
