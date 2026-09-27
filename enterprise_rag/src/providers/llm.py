"""LLM provider 抽象层。

支持：
- ollama: 本地 Ollama HTTP API
- dashscope: 阿里云百炼（qwen-plus / qwen-max / qwen-turbo）
- deepseek: DeepSeek 官方 API（OpenAI Chat Completions 兼容协议）
"""
from __future__ import annotations
import json
import logging
import random
import time
import uuid
from abc import ABC, abstractmethod
from typing import Iterable, Optional
import os

import httpx

from src.config import LLMConfig


logger = logging.getLogger("enterprise_rag.llm")


class BaseLLM(ABC):
    @abstractmethod
    def generate(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """一次性生成，返回完整文本。"""

    def stream(
        self,
        system: str,
        user: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterable[str]:
        """流式生成，逐 chunk yield 增量文本。默认降级为一次性返回。"""
        yield self.generate(system, user, temperature, max_tokens)

    def warmup(self) -> None:
        """可选的启动预热（触发模型加载），默认什么都不做。"""


class OllamaLLM(BaseLLM):
    def __init__(self, cfg: LLMConfig):
        self.host = (cfg.host or "http://127.0.0.1:11434").rstrip("/")
        self.model = cfg.model
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens
        self.timeout = cfg.timeout_s
        # keep_alive 让模型在 Ollama 端常驻，避免空闲后被卸载、下次请求重新加载（3B q4 重新加载约需数秒）
        self.keep_alive = cfg.keep_alive or "30m"
        # runtime_options: num_ctx / num_batch / num_thread / flash_attention 等透传给 Ollama options
        self.runtime_options = dict(cfg.runtime_options or {})
        # 复用连接池：省去每次请求的 TCP 建连开销（localhost 约 5-15ms/次）
        self._client = httpx.Client(
            timeout=self.timeout,
            limits=httpx.Limits(max_keepalive_connections=8, max_connections=16),
            http2=False,
        )

    def _payload(self, system, user, temperature, max_tokens, stream):
        # 每请求覆盖 > 全局 runtime_options > 内置默认；这样调优参数不会被 per-call 参数意外抹掉
        options = {
            **self.runtime_options,
            "temperature": temperature if temperature is not None else self.temperature,
            "num_predict": max_tokens if max_tokens is not None else self.max_tokens,
        }
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": stream,
            "keep_alive": self.keep_alive,
            "options": options,
        }

    def generate(self, system, user, temperature=None, max_tokens=None):
        resp = self._client.post(
            f"{self.host}/api/chat",
            json=self._payload(system, user, temperature, max_tokens, stream=False),
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    def stream(self, system, user, temperature=None, max_tokens=None):
        with self._client.stream(
            "POST",
            f"{self.host}/api/chat",
            json=self._payload(system, user, temperature, max_tokens, stream=True),
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                piece = obj.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if obj.get("done"):
                    break

    def warmup(self) -> None:
        # 用最小生成把模型加载进内存并按目标 num_ctx 预分配 KV cache，
        # 首个真实请求就不用再触发一次重分配 / kernel 首编译。
        try:
            self._client.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": "hi",
                    "stream": False,
                    "keep_alive": self.keep_alive,
                    "options": {**self.runtime_options, "num_predict": 1},
                },
            )
        except Exception:
            pass


class DashScopeLLM(BaseLLM):
    def __init__(self, cfg: LLMConfig):
        import dashscope
        api_key = cfg.api_key or os.environ.get(cfg.api_key_env or "DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DashScope API key not found in env")
        dashscope.api_key = api_key
        self._dashscope = dashscope
        self.model = cfg.model
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens

    def generate(self, system, user, temperature=None, max_tokens=None):
        resp = self._dashscope.Generation.call(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            result_format="message",
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"DashScope LLM failed: {resp.message}")
        return resp.output.choices[0].message.content

    def stream(self, system, user, temperature=None, max_tokens=None):
        stream = self._dashscope.Generation.call(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            result_format="message",
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            stream=True,
            incremental_output=True,
        )
        for resp in stream:
            if resp.status_code != 200:
                raise RuntimeError(f"DashScope LLM stream failed: {resp.message}")
            piece = resp.output.choices[0].message.content
            if piece:
                yield piece


class DeepSeekLLM(BaseLLM):
    """DeepSeek 官方 API 客户端（走 OpenAI Chat Completions 兼容协议）。

    - 认证：Authorization: Bearer $DEEPSEEK_API_KEY
    - 端点：POST {host}/chat/completions
    - 流式：SSE，每行 `data: {json}`，遇 `data: [DONE]` 结束
    - 复用 httpx.Client 连接池，避免每次请求重建 TLS/TCP
    - 重试：对 429 / 500 / 502 / 503 / 504 / 网络类异常自动指数退避重试
    - 分层超时：connect / read / write / pool 独立配置，避免建连慢就整包超时
    - usage 日志：非流式直接读；流式打开 stream_options.include_usage，从末尾 chunk 读
    """

    _DEFAULT_HOST = "https://api.deepseek.com/v1"
    _RETRYABLE_STATUS = {429, 500, 502, 503, 504}
    _RETRYABLE_EXC = (
        httpx.ConnectError,
        httpx.ReadError,
        httpx.WriteError,
        httpx.RemoteProtocolError,
        httpx.PoolTimeout,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
    )

    def __init__(self, cfg: LLMConfig):
        api_key = cfg.api_key or os.environ.get(cfg.api_key_env or "DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "DeepSeek API key not found. "
                "Set env DEEPSEEK_API_KEY or configure llm.api_key_env in yaml."
            )
        self.host = (cfg.host or self._DEFAULT_HOST).rstrip("/")
        self.model = cfg.model
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens
        # runtime_options 支持自定义 retry / timeout 子字段，透传其余到 payload。
        opts = dict(cfg.runtime_options or {})
        self.max_retries = int(opts.pop("max_retries", 3))
        self.retry_base_delay = float(opts.pop("retry_base_delay", 0.5))
        self.retry_max_delay = float(opts.pop("retry_max_delay", 8.0))
        # 分层超时：connect 快失败（DNS/TLS），read 给大模型生成留足时间。
        connect_to = float(opts.pop("connect_timeout_s", 5.0))
        read_to = float(opts.pop("read_timeout_s", cfg.timeout_s))
        write_to = float(opts.pop("write_timeout_s", 10.0))
        pool_to = float(opts.pop("pool_timeout_s", 5.0))
        self.log_usage = bool(opts.pop("log_usage", True))
        self.runtime_options = opts
        self._client = httpx.Client(
            timeout=httpx.Timeout(
                connect=connect_to, read=read_to, write=write_to, pool=pool_to
            ),
            limits=httpx.Limits(max_keepalive_connections=8, max_connections=16),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )

    def _payload(self, system, user, temperature, max_tokens, stream):
        # 每请求覆盖 > runtime_options > 默认。runtime_options 里若配 stream/model/messages 会被下方强字段覆盖，避免误配。
        body = {
            **self.runtime_options,
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "stream": stream,
        }
        if stream and self.log_usage:
            # DeepSeek/OpenAI 协议：流式默认不含 usage，显式开启后在结束前多推一条 usage-only chunk。
            body["stream_options"] = {"include_usage": True}
        return body

    def _sleep_for_retry(self, attempt: int, retry_after: Optional[str]) -> float:
        # 优先遵循服务端 Retry-After（限流时最靠谱），否则指数退避 + 抖动。
        if retry_after:
            try:
                return max(0.0, min(float(retry_after), self.retry_max_delay))
            except ValueError:
                pass
        delay = min(self.retry_base_delay * (2 ** attempt), self.retry_max_delay)
        return delay * (0.5 + random.random())  # jitter: 50%~150%

    def _log_usage(self, rid: str, phase: str, usage: Optional[dict]):
        if not (self.log_usage and usage):
            return
        logger.info(
            "deepseek usage rid=%s phase=%s prompt=%s completion=%s total=%s cache_hit=%s",
            rid,
            phase,
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
            usage.get("prompt_cache_hit_tokens"),
        )

    def generate(self, system, user, temperature=None, max_tokens=None):
        rid = uuid.uuid4().hex[:12]
        url = f"{self.host}/chat/completions"
        payload = self._payload(system, user, temperature, max_tokens, stream=False)
        last_exc: Optional[Exception] = None
        last_status: Optional[int] = None
        for attempt in range(self.max_retries + 1):
            try:
                t0 = time.monotonic()
                resp = self._client.post(url, json=payload, headers={"x-request-id": rid})
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                if resp.status_code in self._RETRYABLE_STATUS:
                    last_status = resp.status_code
                    if attempt < self.max_retries:
                        delay = self._sleep_for_retry(attempt, resp.headers.get("retry-after"))
                        logger.warning(
                            "deepseek retryable http rid=%s status=%s attempt=%s/%s sleep=%.2fs elapsed_ms=%s",
                            rid, resp.status_code, attempt + 1, self.max_retries, delay, elapsed_ms,
                        )
                        time.sleep(delay)
                        continue
                    # 用尽重试：跳出循环，统一由末尾抛错。
                    break
                resp.raise_for_status()
                data = resp.json()
                self._log_usage(rid, "generate", data.get("usage"))
                try:
                    return data["choices"][0]["message"]["content"] or ""
                except (KeyError, IndexError) as e:
                    raise RuntimeError(f"Unexpected DeepSeek response shape: {data}") from e
            except self._RETRYABLE_EXC as e:
                last_exc = e
                if attempt >= self.max_retries:
                    break
                delay = self._sleep_for_retry(attempt, None)
                logger.warning(
                    "deepseek retryable network rid=%s err=%r attempt=%s/%s sleep=%.2fs",
                    rid, e, attempt + 1, self.max_retries, delay,
                )
                time.sleep(delay)
        raise RuntimeError(
            f"DeepSeek generate failed after {self.max_retries + 1} attempts "
            f"(rid={rid}, last_status={last_status})"
        ) from last_exc

    def stream(self, system, user, temperature=None, max_tokens=None):
        rid = uuid.uuid4().hex[:12]
        url = f"{self.host}/chat/completions"
        payload = self._payload(system, user, temperature, max_tokens, stream=True)
        last_exc: Optional[Exception] = None
        last_status: Optional[int] = None
        for attempt in range(self.max_retries + 1):
            try:
                with self._client.stream(
                    "POST", url, json=payload, headers={"x-request-id": rid}
                ) as resp:
                    if resp.status_code in self._RETRYABLE_STATUS:
                        last_status = resp.status_code
                        try:
                            resp.read()  # 释放连接
                        except Exception:
                            pass
                        if attempt < self.max_retries:
                            delay = self._sleep_for_retry(attempt, resp.headers.get("retry-after"))
                            logger.warning(
                                "deepseek stream retryable http rid=%s status=%s attempt=%s/%s sleep=%.2fs",
                                rid, resp.status_code, attempt + 1, self.max_retries, delay,
                            )
                            time.sleep(delay)
                            continue
                        # 用尽重试
                        break
                    resp.raise_for_status()
                    usage: Optional[dict] = None
                    for raw in resp.iter_lines():
                        if not raw:
                            continue
                        line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
                        if not line.startswith("data:"):
                            continue
                        payload_str = line[5:].strip()
                        if payload_str == "[DONE]":
                            break
                        try:
                            obj = json.loads(payload_str)
                        except json.JSONDecodeError:
                            continue
                        # 末尾会有一条 choices=[] 只带 usage 的 chunk（当 include_usage=True 时）
                        u = obj.get("usage")
                        if u:
                            usage = u
                        choices = obj.get("choices") or []
                        if not choices:
                            continue
                        try:
                            piece = choices[0].get("delta", {}).get("content") or ""
                        except (KeyError, IndexError, AttributeError):
                            piece = ""
                        if piece:
                            yield piece
                    self._log_usage(rid, "stream", usage)
                    return
            except self._RETRYABLE_EXC as e:
                last_exc = e
                if attempt >= self.max_retries:
                    break
                delay = self._sleep_for_retry(attempt, None)
                logger.warning(
                    "deepseek stream retryable network rid=%s err=%r attempt=%s/%s sleep=%.2fs",
                    rid, e, attempt + 1, self.max_retries, delay,
                )
                time.sleep(delay)
        raise RuntimeError(
            f"DeepSeek stream failed after {self.max_retries + 1} attempts "
            f"(rid={rid}, last_status={last_status})"
        ) from last_exc

    def warmup(self) -> None:
        # 云 API 无 "模型加载" 概念，预热会白花一次调用费；这里保持 no-op。
        return None


def build_llm(cfg: LLMConfig) -> BaseLLM:
    provider = cfg.provider.lower()
    if provider == "ollama":
        return OllamaLLM(cfg)
    if provider == "dashscope":
        return DashScopeLLM(cfg)
    if provider == "deepseek":
        return DeepSeekLLM(cfg)
    raise ValueError(f"Unknown LLM provider: {cfg.provider}")
