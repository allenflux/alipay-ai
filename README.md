# 转账回执字段识别（LRCNN）

这个工程从**干净的原始截图或拍照图**中定位并读取以下五个区域：

1. `time`：左上角状态栏时间（如 `00:01:09`）
2. `amount`：金额（如 `¥199.93`）
3. `transfer_status`：**勾和“转账成功/支付成功”文字合在同一个框**
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
| `amount` | 货币符号和业务目标金额；多金额模板按统一规则框指定的小金额 | `¥199.93` |
| `transfer_status` | **勾图标 + “转账成功/支付成功”文字作为同一个框** | `◯✓ 支付成功` |
| `recipient_field` | **整条**“收款方 → 收款人值”行（红线左端到右端） | `收款方    上平(**平)` |
| `payment_method_field` | **整条**“付款/交易方式 → 方式值”行（红线左端到右端） | `交易方式    账户余额` |

不要标广告卡片里的其他金额、其他勾图标，也不要标下面白色“立即通知收款人”卡片。收款方与付款方式两类必须按上表框**完整红线行**；OCR 会从行内自动拆出右侧实际值。对文字框，在 LabelMe 的 `description` 填入人工确认过的文本，可留作 OCR 真值；收款人中的 `*` 必须原样保留，不能尝试还原脱敏信息。

`amount` 按减免信息判断：页面没有优惠/立减金额时，标中间的主金额；页面出现优惠券、立减金及 `-¥0.14` 等减免明细时，标下面或右侧显示的**减免前原始交易金额**，中间的减免后金额和各条负数优惠金额都不标。`recipient_field` 的完整行可以与右侧 `amount` 小框重叠，检测训练允许不同类别的框发生包含或重叠。

转换为 COCO：

```bash
python scripts/labelme_to_coco.py \
  --labels data/labels/all \
  --images data/rectified/images \
  --output data/annotations/all.json \
  --require-complete
```

`--require-complete` 会检查每张图必须恰好包含上述五类各一个框，并输出每类框数；少框或重复框时会直接指出对应的 LabelMe JSON。

如果希望先用已经标完整的图片训练第一版，而暂时保留未完成 JSON，可把 `--require-complete` 换成 `--complete-only`。转换器会跳过不完整/重复标注，只将五类各一个框的图片写入 COCO，并列出所有被跳过的文件。

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

训练结束后，用验证阶段选出的 `best.pt` 对从未参与训练/选模的 `test.json` 做一次最终检测评估：

```powershell
python scripts/evaluate.py `
  --checkpoint checkpoints/receipt_lrcnn_v1/best.pt `
  --images data/rectified/images `
  --annotations data/annotations/splits_v1/test.json `
  --output runs/evaluation_v1/test_metrics.json `
  --device cuda `
  --batch-size 2 `
  --workers 0
```

控制台和 `test_metrics.json` 会同时记录总 `mAP@0.50`，以及五类各自的 `AP50` 和 `Recall50`。不要改用 `last.pt`，也不要根据这 21 张测试图反复调参；后续调参继续只看验证集。

## 半自动标注：模型画框，人工只复核

不需要把所有图片从零手工画框。先用当前约 150–200 张人工标注固定 `train/val/test` 并训练 `receipt_lrcnn_v1`；之后让 `best.pt` 给尚未标注的纠正图生成 LabelMe JSON。自动标注直接在 `data/rectified/images` 上预测，不会再次旋转或透视变换。

先只预标 10 张验证效果：

```powershell
python scripts/auto_label.py `
  --checkpoint checkpoints/receipt_lrcnn_v1/best.pt `
  --images data/rectified/images `
  --output data/labels/all `
  --device cuda `
  --score-threshold 0.50 `
  --review-threshold 0.80 `
  --limit 10 `
  --manifest runs/auto_label_v1/pilot.jsonl
```

- 已存在的人工 JSON 默认跳过且绝不覆盖；不要添加 `--overwrite`。
- 少于五个框的预测也会写入，LabelMe 中只需补缺失框。
- `pilot.jsonl` 会记录每张图的框分数、缺失类别和是否需要重点复核；置信度不是人工真值，因此不会写入 LabelMe 的 `description`。
- 确认首批效果后，可将 `--limit 10` 改成 `--limit 100` 分批处理，避免一次生成大量未经复核的伪标签。

继续用原命令打开 LabelMe：

```powershell
labelme data\rectified\images --output data\labels\all --labels data\labelme_labels.txt --no-sort-labels
```

自动框只用于减少拖框工作，必须逐张查看后保存。尤其是出现优惠、立减、`-¥` 时，即使模型置信度很高，也要确认 `amount` 框的是减免前的小金额。全部复核完成后再运行 `labelme_to_coco.py --require-complete`。

第一轮纯人工 `val.json` 和 `test.json` 必须永久保留，自动标注图片只能扩充训练集，不能进入验证集或测试集。每轮优先复核缺框、低置信、新模板、拍照模糊和减免金额页面；高置信结果也随机抽查 5%–10%。

复核完成后只转换五框齐全的自动标注，并与原来的训练集合并。冻结的验证集和测试集只用于防止数据泄漏，不能合入 v2 训练：

```powershell
python scripts/labelme_to_coco.py `
  --labels data/labels/auto_v1 `
  --images data/rectified/images `
  --output data/annotations/auto_reviewed_v1.json `
  --complete-only

python scripts/merge_coco.py `
  --input data/annotations/splits_v1/train.json `
  --input data/annotations/auto_reviewed_v1.json `
  --holdout data/annotations/splits_v1/val.json `
  --holdout data/annotations/splits_v1/test.json `
  --output data/annotations/train_v2.json

python scripts/train.py `
  --train-images data/rectified/images `
  --train-annotations data/annotations/train_v2.json `
  --val-images data/rectified/images `
  --val-annotations data/annotations/splits_v1/val.json `
  --init-checkpoint checkpoints/receipt_lrcnn_v1/best.pt `
  --output checkpoints/receipt_lrcnn_v2 `
  --device cuda `
  --epochs 20 `
  --batch-size 2 `
  --learning-rate 0.001 `
  --workers 4
```

`--init-checkpoint` 只继承 v1 的模型权重，v2 的优化器、学习率和 epoch 从头开始；它不同于中断后继续同一次训练所用的 `--resume`。

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

大量图片应固定分片并启用断点续跑。`scripts/run_bulk_infer.ps1` 已为 `D:\download\TempFakeImages` 配置成 60 个稳定分片；分片大小约为输入总数除以 60。当前 124,323 张输入平均每片约 2,072 张。单张图片缺少任一字段或损坏时会进入该分片的 `inference_errors*.jsonl`，不会中止其余图片。先在第 0 片固定抽取 100 张试运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_bulk_infer.ps1 -StartShard 0 -EndShard 0 -Limit 100
```

`-Limit 100` 会在选中分片内按稳定哈希顺序固定选取最多 100 张源图；完整结果可能少于 100，因为缺框和损坏图片仍属于这批样本。试跑期间不要向源目录增删文件。试跑通过后，正式运行第 0 分片时去掉 `-Limit`；已经成功的图片会被断点续跑逻辑跳过，失败图片会重试，其余图片继续处理。

核对试运行的识别文字、五个圈、耗时和磁盘占用后，再分阶段放量。每个阶段完成后抽查正常结果并查看全部错误清单：

```powershell
# 完成后累计约 10,000 张
powershell -ExecutionPolicy Bypass -File scripts\run_bulk_infer.ps1 -StartShard 1 -EndShard 4

# 之后每批约 10,000 张；继续使用相同 checkpoint 和输出目录
powershell -ExecutionPolicy Bypass -File scripts\run_bulk_infer.ps1 -StartShard 5 -EndShard 9
powershell -ExecutionPolicy Bypass -File scripts\run_bulk_infer.ps1 -StartShard 10 -EndShard 14
# 其余继续按 15-19、20-24……55-59 分批运行
```

中断后直接重跑相同命令即可。脚本固定启用 `--skip-existing`；它只有在结果 JSON 可解析、两张圈选图都存在且不早于原图时才会跳过。不要为同一个输出目录更换 checkpoint、阈值或渲染代码；新模型或新规则必须使用新的版本化输出目录。

每个分片完成后，脚本还会把缺框、损坏或处理失败的原图复制到输出目录下的
`_active_learning_errors/shard-NNN-of-060/raw/`。这里只会复制原图，不会移动、删除或修改输入目录；
脚本也会拒绝把结果目录或错误池设置在输入目录内部。重复运行时，内容相同的文件会安全跳过，
`raw_cohort_manifest.jsonl` 会累计记录曾经失败过的困难样本；`current_inference_errors.jsonl` 表示
本次重试后仍未解决的错误，`history/` 保留每次错误清单快照。若要把困难样本放到独立磁盘目录，
可传入 `-ErrorCohortDir "D:\download\TempFakeActiveLearning_v1\first_pass"`。

困难样本用于 v2 前先运行照片纠正，再在 LabelMe 中复核。有效回执应修正并补齐五个真实字段框；
损坏图、非回执或本身不存在五个字段的图片必须隔离，不能为了满足数量而伪造五框。

直立截图首轮不启用 OCR 四向判断。全量首轮完成后，可用同一个输出目录仅重试此前没有正常结果的图片；已有完整结果会自动跳过：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_bulk_infer.ps1 -StartShard 0 -EndShard 59 -OcrOrientation
```

`-OcrOrientation` 会额外进行四次整图 OCR，因此只在第二轮重试中使用。暂时一次只运行一个分片：当前 PaddleOCR 运行在 CPU，并发多个进程通常会争抢 CPU 和内存，未必更快。

## 隐私

回执中可能包含姓名、余额、交易金额。原图、标注、日志和 checkpoint 都应限制访问；用于共享或调试时应删除不必要的个人数据，并保留脱敏字符原样。
