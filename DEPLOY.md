# MCTS-docker 部署指南

三种部署方式按场景选一种：**A** 目标机有外网（最简单）；**B** 离线内网机（用导出的镜像包）；**C** 不用 Docker。**D** 为 2026-08-27 新增的瘦身传输方案（JumpServer 多硬件测试用，镜像从 14.5GB 减到 ~5.5GB、压缩传输 ~2GB）。

前置要求（所有方式通用）：

- Linux + **NVIDIA 驱动 >= 525**（`nvidia-smi` 右上角显示 CUDA Version >= 12.6）
- LLM API：任一 OpenAI 兼容端点（DeepSeek / GLM / GPT 等）的 url + model + api_key

---

## 方式 D：瘦身镜像 + 小体积传输（JumpServer / 多硬件部署，2026-08-27）

### 为什么原来 14.5GB

`pytorch:2.6.0-cuda12.6-cudnn9-devel` 基础镜像里 **CUDA toolkit 占 4.9GB + nsight 占 1.2GB，流水线完全用不到**——Triton 编译走 wheel 自带的 ptxas，NCU 默认是 `hw_profiler.backend: noop`。真正需要的只有 `/opt/conda`（torch/triton/依赖，6.4GB）+ 源码（46MB）。

### 瘦身三件套

| 工具 | 作用 |
|------|------|
| `build-local-slim.sh` | 从本地已有完整镜像多阶段抽取：239MB cuda-base + 剥完 AKG 栈的 /opt/conda + /app → `directune-mcts:slim`（~5.5GB）。**零网络依赖**（本机 Docker Hub 拉不动大层也能做） |
| `Dockerfile` 的 `ARG BASE` | 基础镜像可参数化，默认已改为 `runtime` 变体（能直连拉取的机器上 `./build.sh` 直接出 ~5GB 镜像） |
| `requirements-slim.txt` | 去掉 AKG fallback 栈（langchain/transformers 等 ~1.2GB），naive+MCTS 主路径零影响（generator 对 AKG 是惰性导入+优雅降级） |

`STRIP=0 ./build-local-slim.sh` 保留 AKG 栈（需要 `gen_mode: akg` fallback 对照时用，~6.7GB）。

### 传输（`deploy-transfer.sh`，按目标机网络条件选）

```bash
./deploy-transfer.sh stream user@target   # ① ssh 直连：docker save|zstd|ssh docker load，无中间文件
./deploy-transfer.sh file                 # ② JumpServer 网页上传：zstd-19 压缩 + 1GB 分卷
./deploy-transfer.sh src user@target      # ③ 只传源码 ~20MB，目标机拉 runtime 基础镜像重建
```

- **①stream**（~2GB 流量）：目标机能从开发机 ssh 反向到达时最快，一条管道完成。
- **②file**：分卷大小 `CHUNK=1G` 可调（JumpServer 网页上传单文件常限 2-4GB）。目标机恢复：`cat directune-mcts.tar.zstd.* | zstd -d | docker load`。
- **③src**（传输最小）：目标机能连 Docker 镜像源（daocloud/1ms 等前缀）时，只传 20MB 源码上下文，基础镜像在目标机侧拉取。**优先试这条**，失败再回退 ②。

### 多硬件注意事项

- **驱动**：宿主机驱动 >= 525 即可（CUDA 12.6 runtime 向下兼容驱动）。镜像内无 toolkit，不存在本地 nvcc 版本匹配问题。
- **NCU**：slim 镜像无 nsight-compute。要采硬件指标时：目标机宿主机装 `nsight-compute` 后用 `hw_profiler.ncu_python` 指向宿主机 Python；或 `STRIP=0` + devel 基础镜像。
- **L2 权重**（24GB）：永远不进镜像，按「跑 L2 题」节挂载。

---

## 方式 A：目标机能上网（从 GitHub 直接部署）

```bash
# 1. 装 Docker + NVIDIA Container Toolkit（已装可跳过）
curl -fsSL https://get.docker.com | sh
sudo apt install -y nvidia-container-toolkit && sudo systemctl restart docker

# 2. 拉代码
git clone https://github.com/VVycccc/MCTS-docker.git
cd MCTS-docker

# 3. 构建镜像（首次约 10-20 分钟，下载 torch/triton 依赖）
./build.sh

# 4. 写配置（填你的 LLM API）
mkdir -p ../run/output && cd ../run
docker run --rm directune-mcts:latest cat /app/config.example.yaml > config.yaml
vi config.yaml    # 填 model.url / model.model / model.api_key 三处

# 5. 跑一道 L1 验证
docker run --gpus all -it --rm \
    -v $PWD/config.yaml:/app/config.yaml \
    -v $PWD/output:/app/output \
    directune-mcts:latest \
    python main.py --config config.yaml \
        --problem problems/kb_level1/01_square_matrix_multiplication.json \
        --initial problems/kb_level1/01_square_matrix_multiplication_initial.py \
        --rounds 3 --breadth 2 --num-samples 1
```

结果在宿主机 `output/` 下（champion 内核、树日志、`final_results.json`）。

> 国内网络构建慢的话：给 Docker 配镜像加速器（`/etc/docker/daemon.json` 的 `registry-mirrors`），pip 源在容器内可 `-e PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple` 覆盖。

## 方式 B：离线内网机（用导出的镜像包）

开发机已导出 `directune-mcts.tar.gz`（约 7.5GB，gzip 校验通过）。内网机只需 Docker（含 nvidia-container-toolkit），不需要外网：

```bash
# 1. U盘/scp 拷贝 directune-mcts.tar.gz 到目标机
# 2. 导入
docker load < directune-mcts.tar.gz
# 3. GPU 检查（应输出 cuda: True）
docker run --rm --gpus all directune-mcts:latest \
    python -c "import torch; print('cuda:', torch.cuda.is_available())"
# 4. 之后同方式 A 的第 4-5 步（配置 + 运行）
```

## 方式 C：不用 Docker（venv 直装）

```bash
git clone https://github.com/VVycccc/MCTS-docker.git && cd MCTS-docker
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
cp config.example.yaml config.yaml    # 填 API key
CUDA_VISIBLE_DEVICES=0 python main.py --config config.yaml \
    --problem problems/kb_level1/01_square_matrix_multiplication.json \
    --initial problems/kb_level1/01_square_matrix_multiplication_initial.py
```

---

## 跑 L2 题（含权重的融合题）

镜像/仓库默认**不含** L2 的 24GB `.pt` 权重（只含 json/initial.py）。跑 L2 需要把权重放到题目同目录再挂载：

```bash
# 宿主机准备：problems/kb_level2/ 下放 <题名>_weights.pt
docker run --gpus all -it --rm \
    -v $PWD/config.yaml:/app/config.yaml \
    -v $PWD/output:/app/output \
    -v /path/to/kb_level2_with_weights:/app/problems/kb_level2 \
    directune-mcts:latest \
    python main.py --config config.yaml \
        --problem problems/kb_level2/14_Gemm_Divide_Sum_Scaling.json
```

权重路径已可移植化：`load_problem()` 会把题目 JSON 里写死的路径重写为「JSON 同目录」解析，因此题集放任意位置都能找到权重（要求权重文件与 JSON 同目录）。

## 常用参数

```bash
--rounds N          # 搜索轮数（MCTS rollout 预算）
--breadth N         # 每 candidate 采样数
--num-samples N     # 并行采样数
--output PATH       # 输出目录（容器内 /app/output，挂出来落到宿主机）
```

config.yaml 关键字段：

| 字段 | 说明 |
|------|------|
| `model.{url,model,api_key}` | LLM 端点（前后端可分别用 `model_frontend`/`model_backend` 覆盖） |
| `search_mode` | `mcts`（默认）/ `unified` / `classic` |
| `search_time_budget` | 搜索阶段墙钟预算（秒） |
| `gen_mode` | `naive`（默认，纯 LLM）/ `v3` / `akg`（fallback） |
| `hw_profiler.backend` | `noop`（默认）/ `ncu`（容器内不可用，需宿主机跑） |

## 已验证

- `./build.sh` 从 git 仓库状态（symlink 已换真实目录）构建正常
- 镜像内 `import main, agents, mcts` 全通过；`--gpus all` 下 CUDA 可用、Triton kernel 实际编译运行 OK（RTX 3090 宿主机）
- L1/L2 权重路径在容器 `/app` 下正确重定位

## 故障排查

| 症状 | 处置 |
|------|------|
| `docker: 'gpus' is not supported` | 目标机没装 nvidia-container-toolkit（方式 A 第 1 步） |
| `cuda: False` | 驱动太旧（< 525）或没加 `--gpus all` |
| 容器内连不上 LLM API | 公司内网需给容器配代理：`-e HTTPS_PROXY=http://x.x.x.x:port` |
| build 拉不动基础镜像 | 见方式 A 尾部的镜像加速器说明 |
