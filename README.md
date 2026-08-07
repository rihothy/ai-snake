# AI 贪吃蛇

基于深度强化学习（Dueling DQN）训练的 AI 贪吃蛇，游戏引擎与状态构建在 C++ 侧实现（pybind11），前端通过 TensorFlow.js 在浏览器内推理，支持任意尺寸地图。

## 核心设计

- **Dueling DQN**：状态价值与优势分头，Double-DQN 目标网络 + N-step（3 步）回报 + Prioritized Experience Replay
- **C++ 游戏引擎**：`train/engine.cpp` 承载游戏逻辑、状态构建（23×23×5 局部视野）与动作 mask，训练侧完全并行
- **动作空间 mask**（训练与网页端完全一致）：
  - L1：即时致死动作（撞墙/撞蛇）禁用
  - L2：BFS 尾巴可达性——每步保证新头能到达尾巴，蛇永远不会自己困死（经典永生不变式）
- **显存控制**：TF 默认预占整卡显存，`main.py` 已将训练池限制为 2GB（实测进程占用约 4GB）

## 目录结构

```
src/           前端源码（js/ 游戏逻辑与智能体、model/ 部署模型、img/）
dist/          webpack 打包产物（index.js）
train/         训练代码（engine.cpp、main.py、agent.py、sim.py、测试）
deploy_web.sh  Keras 模型 → TF.js 部署脚本
```

## 前端运行

```bash
npm install
npm run release   # 生产打包，输出 dist/index.js（约 1.4MB）
python3 -m http.server 8899 --bind 0.0.0.0
```

浏览器打开 `http://localhost:8899` 即可。

## GitHub Pages

项目已配置为 GitHub Pages 部署（`https://rihothy.github.io/ai-snake/`）。所有资源引用均为相对路径，可直接在子路径下工作：

- `index.html` → `dist/index.js`、`src/img/icon.png`（相对路径）
- 智能体模型 → `src/model/model.json`（TF.js 权重路径同为相对路径）
- `.nojekyll` 已加入，GitHub Pages 不会经过 Jekyll 处理

发布步骤：

```bash
git add -A
git commit -m "update model & dist"
git push origin main
```

推送后等待约 1 分钟，Pages 自动重建。若首次部署，需在仓库 Settings → Pages 中确认 Source 为 `main` 分支的 `/ (root)`。模型文件（`src/model/`）与打包产物（`dist/index.js`）均已提交到仓库，无需在 GitHub 侧构建。

## 训练

环境：Python 3.12 + uv + TensorFlow GPU。

```bash
# 1. 编译 C++ 引擎（首次或 engine.cpp 变更后）
bash train/build_engine.sh

# 2. 启动训练（自动从 train/model/model4.h5 + train_state.json 恢复）
cd train
LD_LIBRARY_PATH=$(find ../.venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -name lib -type d | tr '\n' ':') \
../.venv/bin/python main.py
```

主要参数（`train/agent.py`、`train/main.py`）：sampleSize=1024、batchSize=32、γ=0.95、ε 从 1.0 按 0.9995 衰减至 0.10，每 8000 轮温和重置到 0.15。检查点每 1000 轮保存到 `train/model/model4_XXXXX.h5`。

## 部署新模型到网页

```bash
./deploy_web.sh train/model/model4_XXXXX.h5
```

脚本会把 Keras h5 转换为 TF.js 格式、逐位校验权重一致性、备份当前线上模型到 `src/model_backup_*`。

## 模型

- 当前线上：`src/model/` 为 **iter 160000** 检查点，固定条件贪心评估（64 场 × 3000 tick）平均存活约 1823 tick、吃食约 97 个
- `train/model/model4_160000.h5` 为最佳检查点；`model4_35000.h5` 为历史里程碑

## 测试

```bash
cd train
../.venv/bin/python test_game_engine.py      # 引擎确定性/碰撞/食物
../.venv/bin/python test_state_builder.py    # C++ 状态构建 vs Python 参考
../.venv/bin/python test_mask.py             # L1+L2 mask 正确性 + 永生不变式
../.venv/bin/python test_graph_train.py      # 图训练 vs eager 参考一致性
```

## 已知事项

- 训练进程长时间运行后 RSS 可能缓慢增长（分配器碎片，Python 侧容器均有界），建议监控 `/proc/<pid>/status` 或定期重启
- 多蛇同场时 L2 将其他蛇视为静态障碍（近似）；单蛇场景下防困死是严格的
