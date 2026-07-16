# 转账回执字段识别（LRCNN）

这个工程从**干净的原始截图或拍照图**中定位并读取以下五个区域：

1. `time`：左上角状态栏时间（如 `00:01:09`）
2. `amount`：金额（如 `¥199.93`）
3. `transfer_status`：**勾和“转账成功”文字合在同一个框**
4. `recipient_field`：整条“收款方 → 收款人值”行；输出时提取右侧收款人值
5. `payment_method_field`：整条“付款/交易方式 → 方式值”行；输出时提取右侧付款方式

模型为 `LRCNNDetector`：MobileNetV3-FPN + RPN + RoIAlign + Fast R-CNN 检测头。它只负责**位置定位**；文字内容由 PaddleOCR 从定位后的裁图读取。这样时间、金额、姓名、付款方式不会被模型当成固定文字死记。

> 重要：数据集里的图片必须是没有红框、红线、圆圈的原图。红圈只在推理结果副本上绘制，原始图片不被改动。

> 类别顺序已经固定为上述五类。此前把勾和“转账成功”拆成两类的旧标注或 checkpoint 不能续用，必须按本规范重新标注并从头训练。

## 处理流程

```text
干净原图
  → EXIF 方向修正 / 四向文字方向评分
  → 手机屏幕四角检测 + 透视拉伸纠正
  → LRCNN 定位五个区域
  → PaddleOCR 读取时间、金额、转账状态、收款人、付款方式
  → JSON 结构化结果 + 原图/纠正图上的圆圈结果
```

拍照图不会被简单非等比拉伸。工程会保存单应矩阵，推理时把圈从纠正图反投影回原图；因此斜拍后的圈也会与图片透视一致。

## GPU 服务端安装

在服务端先按 [PyTorch 官方安装器](https://pytorch.org/get-started/locally/) 选择和 CUDA 驱动匹配的 `torch` / `torchvision`。再安装其余依赖和本工程。`requirements.txt` 故意不再安装 PyTorch，避免 Windows 从 PyPI 自动装成 CPU 版：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# 安装与服务器 CUDA 对应的 PaddlePaddle 后：
python -m pip install -r requirements-ocr.txt
python -m pip install -e .
```

验证 GPU：

```bash
python scripts/check_gpu.py
```

如果服务端没有外网，先将 PyTorch、TorchVision、PaddlePaddle 和 PaddleOCR 的 wheel 下载到服务器，再从本地 wheel 安装。训练从未缓存权重时，`--pretrained` 会下载一次约 74 MB 的 TorchVision COCO 权重；离线环境可改为 `--no-pretrained`。

### Windows Server（PowerShell）

工程的路径、训练和推理代码都兼容 Windows Server，不依赖 Bash、Linux 路径或 `fork`。推荐 Python 3.10–3.12；先通过 `nvidia-smi` 确认 NVIDIA 驱动与 CUDA 环境，再在 [PyTorch Windows 安装器](https://docs.pytorch.org/get-started/locally/) 选择 **Windows / Pip / 对应 CUDA**，使用它给出的 `torch` 与 `torchvision` 安装命令。

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# 若先前装过 CPU 版，先卸载它：
python -m pip uninstall -y torch torchvision torchaudio
# 先执行 PyTorch 官方安装器为当前 CUDA 给出的命令；RTX 4090 + CUDA 13.3 驱动可用 CUDA 13.0 wheel：
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
# 按 PaddlePaddle Windows GPU 安装页选择匹配 CUDA 的 paddlepaddle-gpu wheel 后：
python -m pip install -r requirements-ocr.txt
python -m pip install -e .
python scripts/check_gpu.py
```

PaddleOCR 的 GPU wheel 也必须与 Windows/CUDA 对应，按 [PaddlePaddle Windows pip 文档](https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/windows-pip_en.html) 选择；即使 PaddleOCR 暂时装 CPU 版，LRCNN 的 PyTorch 训练仍会使用 NVIDIA GPU。首次在 Windows 上训练建议加 `--workers 0` 验证数据路径，通畅后再改为 `--workers 4` 或更高。

## 数据集：原图、标注、切分

目录约定：

```text
data/
  raw/                         # 只放无红框的原始截图/拍照图，保持不变
  rectified/
    images/                    # 自动纠正后、供标注与训练使用的副本
    rectification_manifest.jsonl
  labels/
    all/                       # LabelMe 的 JSON 标注，和 rectified/images 对应
  annotations/
    all.json                   # COCO 格式
    splits/train.json
    splits/val.json
    splits/test.json
```

### 1. 纠正方向和透视

把所有未标注原图放入 `data/raw/`，生成可标注副本：

```bash
python scripts/prepare_photos.py \
  --input data/raw \
  --output data/rectified \
  --corrections data/corrections.json \
  --ocr-orientation
```

- 默认会应用 EXIF 方向；宽图默认转成竖屏；`--ocr-orientation` 会用 OCR 对 0/90/180/270° 打分，能处理上下颠倒的文字。
- 默认会保守地找手机/屏幕四边形并进行 `warpPerspective`；横向广告卡、收款人卡等 UI 面板会被排除。直接截图会保留完整画面。
- 反光、边缘缺失的照片可在 `data/corrections.json` 写人工四角或固定方向，示例见 [data/corrections.example.json](data/corrections.example.json)。人工修正优先于自动判断。
- 如果这一批全是完整截图，第一次准备时可加 `--no-auto-screen`，确保绝不裁切页面；拍照图单独一批再使用默认自动矫正。
- `--max-side 0` 表示保持原图分辨率，不进行长边缩放。

### 2. 标注（不在图片上画红框）

用 LabelMe 打开 `data/rectified/images/`。框只保存到同名 `.json`，输出到 `data/labels/all/`；不要导出带框图片再作为训练输入。

| LabelMe 标签 | 应框选的内容 | 示例 |
|---|---|---|
| `time` | 左上角状态栏显示的完整时间；不框网络、信号、电量图标 | `00:01:09` |
| `amount` | 货币符号和完整金额 | `¥199.93` |
| `transfer_status` | **勾图标 + “转账成功”文字作为同一个框** | `◯✓ 转账成功` |
| `recipient_field` | **整条**“收款方 → 收款人值”行（红线左端到右端） | `收款方    上平(**平)` |
| `payment_method_field` | **整条**“付款/交易方式 → 方式值”行（红线左端到右端） | `交易方式    账户余额` |

不要标广告卡片里的其他金额、其他勾图标，也不要标下面白色“立即通知收款人”卡片。收款方与付款方式两类必须按上表框**完整红线行**；OCR 会从行内自动拆出右侧实际值。对文字框，在 LabelMe 的 `description` 填入人工确认过的文本，可留作 OCR 真值；收款人中的 `*` 必须原样保留，不能尝试还原脱敏信息。

转换为 COCO：

```bash
python scripts/labelme_to_coco.py \
  --labels data/labels/all \
  --images data/rectified/images \
  --output data/annotations/all.json
```

### 3. 为什么不能把全部图片都用于训练

**所有独立原图都应该进入数据集，但不能全部进入梯度训练。** 推荐初始切分：

| 集合 | 比例 | 用途 |
|---|---:|---|
| train | 70% | 更新模型参数 |
| val | 15% | 每轮检查 AP，自动选 `best.pt`，调阈值/轮数 |
| test | 15% | 最后一次独立验收，训练期间不看结果 |

同一交易的截图、该截图的拍照版、旋转版、压缩版、裁剪版必须在**同一个集合**。否则训练集已经看过几乎一样的图，测试准确率会虚高，在新手机/新角度上却不准。

创建分组文件（格式见 [data/groups.example.json](data/groups.example.json)），然后切分：

```bash
python scripts/split_dataset.py \
  --annotations data/annotations/all.json \
  --groups data/groups.json \
  --output-dir data/annotations/splits \
  --train-ratio 0.70 --val-ratio 0.15 --test-ratio 0.15
```

数据量建议：布局非常接近的截图至少 200–300 张已标注图；如果要覆盖真实拍照、斜拍、反光、模糊、不同主题和 UI 版本，建议从 500–1000+ 张开始。关键不是机械地堆相同图片，而是覆盖不同设备、金额长度、收款人格式、付款方式、转向和失败/处理中等负例。

## 训练

训练脚本会优先选择 CUDA GPU（`--device auto`），保存 `last.pt`、按验证集 `mAP@0.50` 自动选出 `best.pt`，并记录 `history.jsonl`。示例：

```bash
python scripts/train.py \
  --train-images data/rectified/images \
  --train-annotations data/annotations/splits/train.json \
  --val-images data/rectified/images \
  --val-annotations data/annotations/splits/val.json \
  --output checkpoints/receipt_lrcnn_v1 \
  --device cuda \
  --epochs 30 \
  --batch-size 2 \
  --workers 4 \
  --pretrained
```

显存不够时先将 `--batch-size` 改为 `1`，再把 `--max-size` 从 `1536` 降为 `1280`；不要把左上角时间或“勾+转账成功”整体裁没。训练中优先关注验证集的每类 `AP50` 和 `recall50`，而不是训练 loss。需要续训：

```bash
python scripts/train.py ... --resume checkpoints/receipt_lrcnn_v1/last.pt
```

最终只对 `test.json` 做一次独立评估/推理，并人工抽查时间、金额、转账状态、收款人和付款方式的 OCR 完全匹配率。

## 推理与圈选结果

```bash
python scripts/infer.py \
  --checkpoint checkpoints/receipt_lrcnn_v1/best.pt \
  --input data/test_raw \
  --output runs/test \
  --device cuda \
  --ocr paddle \
  --ocr-orientation
```

每张输入会生成：

- `*_rectified_annotated.jpg`：纠正后图上的彩色圆圈；
- `*_original_annotated.jpg`：原始照片上的透视回投圆圈；
- `*.json`：时间、金额（含分）、转账状态、收款人、付款方式、置信度、坐标和变换矩阵；
- `inference_manifest.json`：整批结果索引。

低于 `--score-threshold` 的结果会被省略，字段 JSON 会明确标为 `absent` 或 `unreadable`，不会用空字符串伪装成已识别。

## 隐私

回执中可能包含姓名、余额、交易金额。原图、标注、日志和 checkpoint 都应限制访问；用于共享或调试时应删除不必要的个人数据，并保留脱敏字符原样。
