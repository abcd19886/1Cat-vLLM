#!/bin/bash
# 启动本地上下文压缩服务
# 用法: ./start_context_compressor.sh [port]

PORT="${1:-9100}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Starting Context Compressor on port $PORT..."
echo "  Model: qwen3.8-27b-fp8 @ http://localhost:8002/v1"
echo "  API:   POST http://localhost:$PORT/compress"
echo "  Health: GET http://localhost:$PORT/health"
echo ""

python3 "$SCRIPT_DIR/context_compressor.py" \
  --mode serve \
  --port "$PORT" \
  --base-url "http://localhost:8002/v1" \
  --api-key "2026" \
  --model "qwen3.8-27b-fp8"
