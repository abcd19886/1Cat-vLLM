docker run -d \
  --name qwen3.8-27b-dflash2 \
  --gpus '"device=0,1,2,3"' \
  --shm-size=32g \
  -e VLLM_SH70_BACKEND=turbomind \
  -e VLLM_SH70_FLASH_ATTN_V100=1 \
  -e VLLM_SH70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT=0 \
  -v /data/models/Qwen/Qwen3___8-27B-FP8:/models/Qwen3___8-27B-FP8 \
  -v /data/models/incoai/Qwen3___8-27B-DFlash2:/models/Qwen3___8-27B-DFlash2 \
  -v /data/vllm_cache:/cache \
  -p 8002:8000 \
ghcr.1ms.run/abcd19886/1cat-vllm:latest \
  --model /models/Qwen3___8-27B-FP8 \
  --served-model-name qwen3.8-27b-fp8 \
  --trust-remote-code \
  --attention-backend FLASH_ATTN_V100 \
  --tensor-parallel-size 4 \
  --dtype half \
  --kv-cache-dtype fp8_e5m2 \
  --gpu-memory-utilization 0.905 \
  --max-model-len 262144 \
  --max-num-seqs 2 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --max-num-batched-tokens 2048 \
  --limit-mm-per-prompt '{"image":999,"video":0}' \
  --mm-processor-kwargs '{"max_pixels":2073680}' \
  --mm-encoder-tp-mode weights \
  --mm-processor-cache-gb 0 \
  --speculative-config '{"method":"dflash","model":"/models/Qwen3___8-27B-DFlash2","kv_cache_dtype":"fp8_e5m2","revision":"dedf8df68adfb1afeaf7b7480c0a0243108177b4","draft_sample_method":"greedy"}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --chat-template /models/Qwen3___8-27B-FP8/chat_template.jinja \
  --default-chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"medium"}' \
  --async-scheduling \
  --enable-prompt-tokens-details \
  --enable-flashinfer-autotune \
  --optimization-level 3 \
  --gdn-prefill-backend flashqla_sm70 \
  --api-key "2026" \
  --host 0.0.0.0 \
  --port 8000






1Cat-vLLM 构建修复过程（2026-09-06 01:47 – 05:22）
重要信息要写入到readme.txt里面,已经实现的要说明一下
仓库 abcd19886/1Cat-vLLM · 目标：V100 / SM70 专用 vLLM fork 构建成功
① 认证与环境（~01:47–02:05）
Fine-grained PAT 登录；gh CLI 未安装、写 ~/.git-credentials 被自动模式拦截 → 改用 export GITHUB_TOKEN + curl 调 GitHub API，验证 token 有效
② 构建①失败 · CMake 配置阶段（约 5 分钟挂）
根因：workflow 中 TORCH_CUDA_ARCH_LIST="7.0" 的引号泄漏进 CMake；PyTorch 2.10.0+cu128 把带引号的 "7.0" 当未知架构名 → Found Unknown CUDA Architecture Name。已用 CMake 4.4.3 本地复现确认
③ 修复① · 已提交并推送（commit a1ab4be14）
a) build-wheel.yml 去掉 build-arg 引号；b) setup.py build_extensions() 增加 TORCH_CUDA_ARCH_LIST 引号规范化（纵深防御）。2 文件 +26/-1
④ 触发阻塞 → 换 Token（~03:00–03:37）
Fine-grained PAT 无 Actions 写权限 → dispatch 返回 403。用户改提供 classic PAT（完整权限） → dispatch 成功（HTTP 204），构建 run 34009466236 启动
⑤ 构建② · 编译约 45–71 分钟，97 个 CUDA targets 全部编译成功，打包阶段失败
修复①生效、通过 configure；随后失败于 No such file or directory: …/vllm/third_party/triton_kernels。根因：setup.py 无条件从 build_lib 复制 triton_kernels，而 SM70 Docker 构建（Dockerfile sed）已移除该扩展，目录不存在；下方 deep_gemm 有 os.path.exists 保护，triton_kernels 没有
⑥ 修复② · 已写入工作区，未提交（~05:11）
给 triton_kernels 的 shutil.copytree 加存在性判断，与 deep_gemm 模式对齐；语法校验通过（setup.py OK）。提交动作被用户拒绝
⑦ 用户打断（05:21）· 新诉求：复用成功构建产物
“每次失败后都要重新构建……下次构建能否复用已经成功的构建，不然太花时间了” → 指向启用 USE_SCCACHE（sccache 缓存），避免每次 71 分钟全量重编
当前状态：构建未成功（卡在 wheel 打包）；修复②待提交并重新触发；待办：①提交修复② ②重新构建验证 ③启用 sccache 缓存复用编译产物 ④（次要）pre-commit 既有失败：workflow lint（run: 块、debug: true）与 docker/versions.json 与 Dockerfile 不同步

⑧ Dockerfile 809 行修复（2026-09-06 后续）
根因：RUN 指令 --mount 参数后紧跟 && 导致 shell 命令以 && 开头，dash 语法错误 exit 2。已修复：改用 set -e; + ; \ 续行，添加 CUDA_TAG 变量避免重复计算。

关于构建缓存：
- workflow 已启用 cache-from/to: type=gha（GitHub Actions 10GB 缓存）
- USE_SCCACHE=1 会用 sccache（不是 ccache）缓存编译产物到 /root/.cache/sccache
- 第一次构建会写入缓存，第二次起只要不改 Dockerfile 就能复用，97 targets 会从缓存读取
- 监控 sccache 统计：构建日志搜 "sccache --show-stats"，看 Cache hits 数量
⑨ MiniMax-H3 INT8 ConvRot 文本编码器 + GPU 租约阈值（2026-09-12）
背景：/data/models/MiniMax-H3 的文本编码器换成了 ComfyUI INT8 量化版
qwen3vl_32b_minimax_h3_int8_convrot.safetensors（27GB，单文件）。
检查点结构：50 层 × 7 个投影（q/k/v/o/gate/up/down）= 350 个 INT8 张量，
全部 int8_tensorwise + ConvRot（groupsize 256），每层带 .comfy_quant
uint8 JSON 标记张量；视觉塔/embedding/layernorm 保持 BF16（视觉塔带 bias）。
DiT 已有完整 ConvRot-int8 路径（DiffusionInt8ConvRotConfig +
Int8ConvRotLinearMethod + w8a16 CUDA 扩展），本次把它移植到文本编码器。

已实现（工作区未提交）：
1) vllm/model_executor/models/minimax_h3/encoder.py
   - build_encoder_int8_config()：解析检查点 .comfy_quant 标记，映射到编码器
     内部模块前缀（q/k/v→qkv_proj、gate/up→gate_up_proj 融合层），
     同一融合层多个标记冲突时报错；无标记返回 None（走 BF16/FP16 原路径）。
   - _map_weight_name()：兼容 ComfyUI 扁平命名（model.layers.* / visual.* /
     model.embed_tokens.*），跳过 .comfy_quant 元数据张量。
   - MiniMaxH3Qwen3VLEncoder 接受 quant_config；INT8 路径跳过整体 .to(dtype)
     （INT8 权重 + FP32 scale 不能上转换）；构造后调用
     validate_model_bindings 确保 350 个标记全部绑定到可执行层。
   - 修复 RowParallelLinear.weight_loader：weight_scale 是每输出行 scale
     （[N,1]，无 input_dim），行并行不切分，必须整体拷贝；原实现按 dim1
     narrow 会越界崩溃（o_proj/down_proj 都是行并行）。
2) vllm/model_executor/models/minimax_h3/pipeline.py
   - 构造编码器前 build_encoder_int8_config(shared/"text_encoder")，
     传入 quant_config；load_weights 后对所有 Int8ConvRotLinearMethod 层
     调 process_weights_after_loading（校验 INT8/FP32 scale 并整理布局）。
3) vllm/video/gpu.py（GPU 租约阈值补丁）
   - 原硬编码 30GB 总量/30GB 空闲/256MB 外部占用阈值，16GB V100 永远租不到。
   - 新增环境变量覆盖：VLLM_H3_MIN_TOTAL_GIB（默认 30）、
     VLLM_H3_MIN_FREE_GIB（默认 30）、VLLM_H3_MAX_FOREIGN_MB（默认 256）。
     16GB 卡示例：VLLM_H3_MIN_TOTAL_GIB=15 VLLM_H3_MIN_FREE_GIB=15。

TP4 ConvRot 对齐验证（config: hidden 5120 / intermediate 25600 /
heads 64 / kv_heads 8 / head_dim 128）：
   qkv/gate_up 输入 5120%256=0；o_proj 本地 8192/4=2048%256=0；
   down_proj 本地 25600/4=6400%256=0。全部满足。

验证结果（1cat-vllm 镜像 + 工作区代码，CPU 参考路径 + V100 SM70 CUDA 路径）：
   - TP1 CPU：200 个量化层（350 标记按融合层归并）全部加载，50 层
     encode_ids 前向通过，输出 (4,5120) 有限值。
   - TP4 CPU（4 线程 barrier all_reduce 模拟）：与 TP1 参考 mean_rel=3.3e-3
     （fp16 舍入量级），4 个 rank 输出完全一致 → 列切分/行切分/scale 加载/
     ConvRot 分组对齐全部正确。
   - CUDA（V100 GPU4）：layer0 四种投影（qkv/o/gate_up/down）走 w8a16
     扩展（rotate+dequantize+fp16_gemm）与 CPU 参考 mean_rel≈3e-6，通过。
   - ruff check + format（v0.14.0，仓库规则）：encoder.py / pipeline.py 通过。

注意：
   - 编码器 INT8 权重不做 h3_fp16_weight 缓存（DiT 有预算缓存），每次前向
     现场 dequantize——16GB 卡上缓存 fp16 副本（每卡 ~13.5GB）放不下。
   - FL2VA/text_encoder/ 目录里旧的 model.safetensors.index.json 指向已不存在的
     HF 分片，无害（iter_checkpoint_weights 只 glob *.safetensors）；
     int8 文件以软链接放入该目录。
   - 测试脚本：/tmp/test_encoder_int8.py（tp1/tp4 两阶段）、
     /tmp/test_encoder_cuda.py（GPU 单层对照）。
