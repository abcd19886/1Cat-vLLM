#!/usr/bin/env python3
"""
Local Context Compressor — 用本地 vLLM (qwen3.8-27b-fp8) 替代远程 compact task。

用法:
  # 压缩文本
  python context_compressor.py --mode text --text "要压缩的长文本..." --ratio 0.3

  # 压缩文件
  python context_compressor.py --mode file --input long_doc.txt --ratio 0.3 -o summary.txt

  # 压缩对话历史 (JSON: [{"role":"user","content":"..."}, ...])
  python context_compressor.py --mode conversation --input history.json --keep-recent 4

  # 启动 HTTP 服务 (供上层应用调用, 替代 remote compact task)
  python context_compressor.py --serve --port 9100
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

import requests

DEFAULT_BASE_URL = "http://localhost:8002/v1"
DEFAULT_API_KEY = "2026"
DEFAULT_MODEL = "qwen3.8-27b-fp8"
MAX_MODEL_LEN = 262144

COMPRESS_PROMPT = (
    "请将以下内容压缩为摘要。要求：直接输出摘要结果，不要输出分析过程。"
    "保留关键事实、数字、结论、代码片段和专有名词，去除冗余描述和重复内容。"
    "目标长度约 {target_tokens} token。\n\n"
    "内容：\n{text}"
)

CONVERSATION_SUMMARY_PROMPT = (
    "以下是多轮对话历史，请将其压缩为一段简洁的摘要。"
    "要求：直接输出摘要，不要分析过程。"
    "保留关键决策、结论、用户意图和技术细节，去除寒暄和重复。\n\n"
    "对话历史：\n{text}"
)


@dataclass
class CompressorConfig:
    base_url: str = DEFAULT_BASE_URL
    api_key: str = DEFAULT_API_KEY
    model: str = DEFAULT_MODEL
    max_model_len: int = MAX_MODEL_LEN
    temperature: float = 0.1
    max_tokens: int = 4096
    compression_ratio: float = 0.3
    keep_recent_messages: int = 4
    chunk_size: int = 32000
    timeout: int = 180


class ContextCompressor:
    def __init__(self, config: Optional[CompressorConfig] = None):
        self.config = config or CompressorConfig()
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        })

    def _call_llm(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        url = f"{self.config.base_url}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens or self.config.max_tokens,
            "temperature": self.config.temperature,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = self.session.post(url, json=payload, timeout=self.config.timeout)
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        content = msg.get("content")
        if content is None:
            reasoning = msg.get("reasoning", "")
            if reasoning:
                print("  [warn] content is null, falling back to reasoning field",
                      file=sys.stderr)
                return reasoning
            raise RuntimeError(
                f"LLM returned empty response "
                f"(finish_reason={data['choices'][0].get('finish_reason')})")
        return content.strip()

    def compress_text(self, text: str, target_ratio: Optional[float] = None) -> str:
        ratio = target_ratio or self.config.compression_ratio
        estimated_tokens = max(1, len(text) // 2)
        target_tokens = max(128, int(estimated_tokens * ratio))

        if estimated_tokens < 4000:
            prompt = COMPRESS_PROMPT.format(
                target_tokens=target_tokens, text=text)
            return self._call_llm(prompt,
                                  max_tokens=min(target_tokens * 3,
                                                 self.config.max_tokens))

        return self._hierarchical_compress(text, ratio)

    def _hierarchical_compress(self, text: str, target_ratio: float) -> str:
        chunk_size = self.config.chunk_size
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        if len(chunks) == 1:
            return self.compress_text(text, target_ratio)

        summaries = []
        for i, chunk in enumerate(chunks):
            print(f"  [hierarchical] chunk {i + 1}/{len(chunks)} "
                  f"({len(chunk)} chars)...", file=sys.stderr)
            s = self.compress_text(chunk, target_ratio=0.5)
            summaries.append(s)

        merged = "\n---\n".join(summaries)
        if len(merged) > chunk_size:
            return self._hierarchical_compress(merged, target_ratio)
        return self.compress_text(merged, target_ratio)

    def compress_conversation(self, messages: list[dict],
                              keep_recent: Optional[int] = None) -> list[dict]:
        keep = keep_recent or self.config.keep_recent_messages
        if len(messages) <= keep:
            return list(messages)

        old = messages[:-keep]
        recent = messages[-keep:]

        old_text = "\n".join(
            f"[{m.get('role', 'unknown')}]: {m.get('content', '')}"
            for m in old
        )
        prompt = CONVERSATION_SUMMARY_PROMPT.format(text=old_text)
        summary = self._call_llm(prompt, max_tokens=self.config.max_tokens)

        return [
            {"role": "system", "content": f"[历史对话摘要]\n{summary}"},
            *recent,
        ]

    def compress_messages_api(self, messages: list[dict],
                              target_ratio: float = 0.3) -> dict:
        start = time.time()
        total_chars = sum(len(m.get("content", "")) for m in messages)

        if len(messages) <= self.config.keep_recent_messages:
            compressed = list(messages)
        else:
            compressed = self.compress_conversation(messages)

        compressed_chars = sum(len(m.get("content", "")) for m in compressed)
        elapsed = time.time() - start

        return {
            "success": True,
            "original_message_count": len(messages),
            "compressed_message_count": len(compressed),
            "original_chars": total_chars,
            "compressed_chars": compressed_chars,
            "compression_ratio": round(compressed_chars / max(1, total_chars), 4),
            "elapsed_seconds": round(elapsed, 2),
            "messages": compressed,
        }


class _CompressHandler(BaseHTTPRequestHandler):
    compressor: ContextCompressor = None

    def do_POST(self):
        if self.path not in ("/compress", "/v1/compact"):
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            self._respond(400, {"success": False, "error": "invalid JSON"})
            return

        messages = data.get("messages")
        if not messages or not isinstance(messages, list):
            self._respond(400, {"success": False, "error": "missing 'messages' array"})
            return

        ratio = data.get("ratio", 0.3)
        keep_recent = data.get("keep_recent")

        try:
            if keep_recent:
                self.compressor.config.keep_recent_messages = keep_recent
            result = self.compressor.compress_messages_api(messages,
                                                           target_ratio=ratio)
            self._respond(200, result)
        except Exception as exc:
            self._respond(500, {"success": False, "error": str(exc)})

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok",
                                "model": self.compressor.config.model})
        else:
            self.send_error(404)

    def _respond(self, code: int, data: dict):
        payload = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        print(f"[compressor] {fmt % args}", file=sys.stderr)


def serve(config: CompressorConfig, port: int):
    compressor = ContextCompressor(config)
    _CompressHandler.compressor = compressor
    server = HTTPServer(("0.0.0.0", port), _CompressHandler)
    print(f"Context Compressor listening on 0.0.0.0:{port}", file=sys.stderr)
    print(f"  Model: {config.model} @ {config.base_url}", file=sys.stderr)
    print(f"  POST /compress  or  /v1/compact", file=sys.stderr)
    print(f"  GET  /health", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Local Context Compressor (qwen3.8-27b-fp8 via vLLM)")
    parser.add_argument("--mode",
                        choices=["text", "conversation", "file", "serve"],
                        default="text")
    parser.add_argument("--input", "-i", help="Input file")
    parser.add_argument("--text", "-t", help="Input text")
    parser.add_argument("--ratio", "-r", type=float, default=0.3)
    parser.add_argument("--keep-recent", type=int, default=4)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", "-o", help="Output file")
    parser.add_argument("--port", type=int, default=9100,
                        help="HTTP port for --serve")
    args = parser.parse_args()

    config = CompressorConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        compression_ratio=args.ratio,
        keep_recent_messages=args.keep_recent,
    )

    if args.mode == "serve":
        serve(config, args.port)
        return

    compressor = ContextCompressor(config)

    if args.mode == "text":
        text = args.text or sys.stdin.read()
        if not text.strip():
            print("Error: no input text", file=sys.stderr)
            sys.exit(1)
        print(f"Compressing {len(text)} chars (ratio={args.ratio})...",
              file=sys.stderr)
        start = time.time()
        result = compressor.compress_text(text, target_ratio=args.ratio)
        print(f"Done in {time.time() - start:.1f}s: "
              f"{len(text)} -> {len(result)} chars", file=sys.stderr)
        output = result

    elif args.mode == "file":
        if not args.input:
            print("Error: --input required", file=sys.stderr)
            sys.exit(1)
        with open(args.input) as f:
            text = f.read()
        print(f"Compressing file {len(text)} chars (ratio={args.ratio})...",
              file=sys.stderr)
        start = time.time()
        result = compressor.compress_text(text, target_ratio=args.ratio)
        print(f"Done in {time.time() - start:.1f}s: "
              f"{len(text)} -> {len(result)} chars", file=sys.stderr)
        output = result

    elif args.mode == "conversation":
        if not args.input:
            print("Error: --input required "
                  "(JSON: [{role, content}, ...])", file=sys.stderr)
            sys.exit(1)
        with open(args.input) as f:
            messages = json.load(f)
        print(f"Compressing {len(messages)} messages "
              f"(keep_recent={args.keep_recent})...", file=sys.stderr)
        start = time.time()
        result = compressor.compress_conversation(messages,
                                                   keep_recent=args.keep_recent)
        print(f"Done in {time.time() - start:.1f}s: "
              f"{len(messages)} -> {len(result)} messages", file=sys.stderr)
        output = json.dumps(result, ensure_ascii=False, indent=2)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Saved to {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
