# DirecTune-MCTS 部署指南

打包目标：让项目可以部署在任意有 NVIDIA GPU 的 Linux 机器上，无需 conda、无需 CUDA toolkit、无需手工配置环境。

## 一、Docker 方式（推荐）

### 前提（目标机器只需要两样）

1. Linux + NVIDIA **驱动**（CUDA 12.6 基础镜像需要驱动 >= 525，`nvidia-smi` 右上角显示的 CUDA Version >= 12.6 即可）
2. Docker + NVIDIA Container Toolkit（`docker run --gpus all` 可用）

### 构建

```bash
cd DirecTune-MCTS
./build.sh            # 自动：解引用 akg_frontend/problems 两个 symlink → .build/ 临时上下文 → docker build
./build.sh mytag      # 自定义 tag
```

产物：`directune-mcts:latest`（约 14.5GB，含 torch 2.6 + cu126 + triton 3.7 + AKG vendored 前端 + L1/L2 题集）。
镜像里**不含任何 API key**（`config.yaml`、`.akg/settings.json` 均被排除，镜像内是 `config.example.yaml` 的脱敏副本）。

### 运行

```bash
# 挂载真实配置 + 输出目录（output 写到宿主机，容器删了结果还在）
docker run --gpus all -it --rm \
    -v $PWD/config.yaml:/app/config.yaml \
    -v $PWD/output:/app/output \
    directune-mcts:latest \
    python main.py --config config.yaml \
        --problem problems/kb_level1/01_square_matrix_multiplication.json \
        --initial problems/kb_level1/01_square_matrix_multiplication_initial.py
```

`config.yaml` 从 `config.example.yaml` 复制后填入 `model.url / model.model / model.api_key`（任意 OpenAI 兼容端点）。

### L2 权重（大文件按需挂载）

镜像默认**不含** `problems/kb_level2/*.pt`（24GB）。跑 L2 时把宿主机权重目录挂进去：

```bash
-v /path/to/kb_level2:/app/problems/kb_level2
```

题目的 `.pt` 权重路径已做可移植化处理：`load_problem()` 加载时会把 reference 里写死的绝对/相对 `_weights_path` 重写为「problem JSON 同目录」解析，因此把题集放在任意位置都能找到权重（前提：权重文件与 JSON 同目录，这也是当前两套题集的实际布局）。

### 分发到其他机器

```bash
docker save directune-mcts:latest | gzip > directune-mcts.tar.gz
# 拷到目标机后：
docker load < directune-mcts.tar.gz
```

## 二、免 Docker 方式（tarball + venv）

```bash
# 1) 打源码包（同样解引用 symlink）
tar -czhf directune-mcts-src.tar.gz \
    --exclude=output --exclude=.git --exclude=__pycache__ \
    --exclude='*.pt' --exclude=.backup_pre_deakg \
    --dereference -C /home/wangyichen DirecTune-MCTS

# 2) 目标机器
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
cp config.example.yaml config.yaml   # 填 API key
python main.py --config config.yaml --problem ... --initial ...
```

注意：tarball 方式需要目标机自行装对 torch/triton 版本；Docker 方式版本已锁定并验证。

## 三、路径可移植化改动清单（本次修改）

| 文件 | 改动 |
|------|------|
| `triton_backend.py` `load_problem()` | `_weights_path` 统一重写为按 problem JSON 所在目录解析的绝对路径（原 L2 是 `/home/wangyichen/...` 绝对路径，迁移即失效） |
| `hardware_profiler.py` | NCU 用的 python 解释器默认值 `sys.executable`，可用环境变量 `DT_NCU_PYTHON` 或 config `hw_profiler.ncu_python` 覆盖（原来是写死的 forge env 路径） |
| `bench_inductor.py` | `REPO`/`PROBLEM_DIR` 改为相对本文件 |
| `naive_seed_gen.py` / `naive_seed_l2.py` / `blind_ablation.py` / `validate_seeds.py` | `DIR_PROBE` 默认 `../dir_probe`，环境变量 `DT_DIR_PROBE` 可覆盖 |
| `naive_seed_l2.py` `DT` | 改为本仓库 `problems/kb_level2`（原指向老 DirecTune） |
| `measure_baseline.py` / `scripts/plot_*.py` / `verify_convert*.py` / `fix_level1_universal.py` / `scripts/convert_kb_level2.py` | 全部改为相对路径或 `KB_L1`/`KB_L2` 环境变量（KernelBench 源目录默认 `../../KernelBench/...`） |

## 四、已验证

- 镜像内 `import main, agents, mcts, generator, search, hardware_profiler` + AKG 前端 `LangGraphTask` 全部通过
- `--gpus all` 容器内 `torch.cuda.is_available() == True`，Triton kernel 实际编译+运行通过（RTX 3090 宿主机）
- L1/L2 权重路径解析在容器内 `/app` 下正确重定位（L2 权重文件本体按需挂载）

## 五、已知限制

- 基础镜像用的是本机缓存的 `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel`（本机 registry 直连失败）。开发环境是 torch 2.12 + CUDA 13；如目标环境需要完全一致的版本，把 Dockerfile `FROM` 换成 `pytorch/pytorch:2.12.0-cuda13.0-cudnn9-runtime`（驱动需 >= 580）并重新 `./build.sh`。
- NCU profiling（`hw_profiler.backend: ncu`）在容器内需要容器内有 `ncu` 二进制和 passwordless sudo，默认建议 `noop`（v6 默认就是）。
- forge_tle 后端（FlagTree triton）不在镜像内，如需 TLE 方向请用对应环境启动。
