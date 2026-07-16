# 转账回执字段识别（LRCNN）

这个工程从**干净的原始截图或拍照图**中定位并读取以下五个区域：

1. `amount`：金额（如 `¥199.93`）
2. `success_icon`：`转账成功` 前面的勾
3. `success_text`：`转账成功` 文字
4. `recipient_value`：收款方的实际值（不包含左侧“收款方”标签）
5. `payment_method_value`：付款方式的实际值（不包含左侧“付款方式”标签）

模型为 `LRCNNDetector`：MobileNetV3-FPN + RPN + RoIAlign + Fast R-CNN 检测头。它只负责**位置定位**；中文内容由 PaddleOCR 从定位后的裁图读取。这样金额、姓名、付款方式不会被模型当成固定文字死记。

> 重要：数据集里的图片必须是没有红框、红线、圆圈的原图。红圈只在推理结果副本上绘制，原始图片不被改动。

## 处理流程

```text
干净原图
  → EXIF 方向修正 / 四向文字方向评分
  → 手机屏幕四角检测 + 透视拉伸纠正
  → LRCNN 定位五个区域
  → PaddleOCR 读取金额、成功文字、收款人、付款方式
  → JSON 结构化结果 + 原图/纠正图上的圆圈结果
```

拍照图不会被简单非等比拉伸。工程会保存单应矩阵，推理时把圈从纠正图反投影回原图；因此斜拍后的圈也会与图片透视一致。

## GPU 服务端安装

在服务端先按 [PyTorch 官方安装器](https://pytorch.org/get-started/locally/) 选择和 CUDA 驱动匹配的 `torch` / `torchvision`。再安装其余依赖和本工程：

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
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

如果服务端没有外网，先将 PyTorch、TorchVision、PaddlePaddle 和 PaddleOCR 的 wheel 下载到服务器，再从本地 wheel 安装。训练从未缓存权重时，`--pretrained` 会下载一次约 74 MB 的 TorchVision COCO 权重；离线环境可改为 `--no-pretrained`。

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
- 默认会找手机/屏幕四边形并进行 `warpPerspective`。直接截图通常会保留完整画面。
- 反光、边缘缺失的照片可在 `data/corrections.json` 写人工四角或固定方向，示例见 [data/corrections.example.json](data/corrections.example.json)。人工修正优先于自动判断。

### 2. 标注（不在图片上画红框）

用 LabelMe 打开 `data/rectified/images/`。框只保存到同名 `.json`，输出到 `data/labels/all/`；不要导出带框图片再作为训练输入。

| LabelMe 标签 | 应框选的内容 | 示例 |
|---|---|---|
| `amount` | 货币符号和完整金额 | `¥199.93` |
| `success_icon` | 成功文字左侧那个勾 | 圆形勾图标 |
| `success_text` | 仅成功文字 | `转账成功` |
| `recipient_value` | 收款人右侧实际值 | `上平(**平)` |
| `payment_method_value` | 付款方式右侧实际值 | `账户余额` |

不要标广告卡片里的其他金额、其他勾图标，也不要把左侧“收款方”“付款方式”标题和右侧值框在同一个框里。对文字框，在 LabelMe 的 `description` 填入人工确认过的文本，可留作 OCR 真值；收款人中的 `*` 必须原样保留，不能尝试还原脱敏信息。

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

显存不够时先将 `--batch-size` 改为 `1`，再把 `--max-size` 从 `1536` 降为 `1280`；不要把小勾图标裁没。训练中优先关注验证集的每类 `AP50` 和 `recall50`，而不是训练 loss。需要续训：

```bash
python scripts/train.py ... --resume checkpoints/receipt_lrcnn_v1/last.pt
```

最终只对 `test.json` 做一次独立评估/推理，并人工抽查金额、收款人和付款方式的 OCR 完全匹配率。

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
- `*.json`：金额（含分）、成功状态/勾、收款人、付款方式、置信度、坐标和变换矩阵；
- `inference_manifest.json`：整批结果索引。

低于 `--score-threshold` 的结果会被省略，字段 JSON 会明确标为 `absent` 或 `unreadable`，不会用空字符串伪装成已识别。

## 隐私

回执中可能包含姓名、余额、交易金额。原图、标注、日志和 checkpoint 都应限制访问；用于共享或调试时应删除不必要的个人数据，并保留脱敏字符原样。
