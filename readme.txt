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