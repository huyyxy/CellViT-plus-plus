[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31014/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![CodeFactor](https://www.codefactor.io/repository/github/tio-ikim/cellvit-plus-plus/badge)](https://www.codefactor.io/repository/github/tio-ikim/cellvit-plus-plus)
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=Pytorch&logoColor=white"/></a>
![Visitors](https://api.visitorbadge.io/api/combined?path=https%3A%2F%2Fgithub.com%2FTIO-IKIM%2FCellViT-plus-plus&label=Visitors&countColor=%2337d67a&style=flat)
[![arXiv](https://img.shields.io/badge/arXiv-2501.05269-b31b1b.svg)](https://arxiv.org/abs/2501.05269)

> [!IMPORTANT]
> [![Update](https://img.shields.io/badge/Update-PyPI-green?labelColor=Red&style=flat&logo=pypi&link=https://pypi.org/project/cellvit/)](https://pypi.org/project/cellvit/)
>
> 如果你只想运行推理，请查看 [PyPI 包](https://pypi.org/project/cellvit/) 和相应的 [GitHub 仓库](https://github.com/TIO-IKIM/CellViT-Inference)
___
<p align="center">
  <img src="./docs/figures/banner.png"/>
</p>

___

# CellViT++: 基于基础模型的高能效自适应细胞分割与分类
<div align="center">

[关键特性](#关键特性) • [安装](#安装) • [推理](#推理) • [示例](#示例) • [重新训练](#重新训练你自己的分类器工作流程) • [可重现性](#可重现性) • [查看器](#基于web的查看器) • [标注](#标注工具) • [致谢](#致谢) • [引用](#引用)

</div>


> [!TIP]
> 要访问之前的版本 (CellViT)，请访问此 [链接](https://github.com/TIO-IKIM/CellViT)


## 关键特性

---

> **更新 08.08.2023**:
>
> :ballot_box_with_check: 添加基于token的分类器，可用于多种细胞分类分类体系
>
> :ballot_box_with_check: 通过使用缓存实现高效的微调运行时间
>
> :ballot_box_with_check: 改进的CLI和功能 - 查看示例
>
> :ballot_box_with_check: 基于Web的查看器 (参见 [可视化](#可视化))
>
> :ballot_box_with_check: 我们包含了 [PathoPatcher](https://github.com/TIO-IKIM/PathoPatcher) 作为预处理框架
>
> :ballot_box_with_check: 更加稳定 - 如果出现问题，请联系我们！
---


#### 可视化
<div align="center">

![Example](docs/figures/web-viewer.gif)

有关查看器的更多信息，请参见 [下文](#基于web的查看器) 
</div>

## 安装

### 硬件要求

- 🚀 **支持CUDA的GPU**: 至少24 GB显存的GPU（推荐48 GB以获得更快的推理速度，例如RTX-A6000）。我们使用一块具有80GB显存的NVIDIA A100进行实验。
- 🧠 **内存**: 最少32 GB RAM。
- 💾 **存储**: 至少30 GB磁盘空间。
- 🖥️ **CPU**: 最少16个CPU核心。

### 本地安装


<details>
  <summary>安装 (conda + pip) - 快速</summary>

1. 创建环境
    ```bash
    conda env create -f environment_verbose.yaml
    ```
2. 激活你的环境
    ```bash
    conda activate cellvit_env
    ```
3. 安装pip包
    ```bash
    pip install -r requirements.txt
    ```
4. 为你的系统安装pytorch  
   我们使用了以下pytorch版本:
   - torch==2.2.1
   - torchaudio==2.2.1
   - torchvision==0.17.1

    你可以在这里找到关于pytorch的安装说明: https://pytorch.org/get-started/previous-versions/

    :bulb: 即使在环境构建期间安装了PyTorch，也请专门为你的系统安装它

    示例 (但请在网站上检查你的版本！):
    ```bash
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
    ```

</details>

<details>
  <summary>安装 (conda) - 慢速</summary>



1. 创建环境
    ```bash
    conda env create -f environment.yaml
    ```
    > :hourglass_flowing_sand: 这个过程可能需要一些时间，在我们的硬件上大约需要30分钟来安装所有包，但这可能需要更长时间。如果你想跟踪安装进度，你可以考虑使用上述的 **conda + pip** 分步方法。

2. 激活你的环境
    ```bash
    conda activate cellvit_env
    ```

3. 为你的系统安装pytorch  
   我们使用了以下pytorch版本:
   - torch==2.2.2
   - torchaudio==2.2.2
   - torchvision==0.17.2

    你可以在这里找到关于pytorch的安装说明: https://pytorch.org/get-started/previous-versions/

    :bulb: 即使在环境构建期间安装了PyTorch，也请专门为你的系统安装它

    示例 (但请在网站上检查你的版本！):
    ```bash
    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
    ```

</details>


##### **检查你的安装** 📦

完成上述步骤并激活环境后，运行以下命令来验证你的环境是否正确设置:

```bash
python3 check_environment.py
```
注意: 如果脚本成功运行，你的环境已准备就绪。这也会下载测试数据库



##### 故障排除:
* **CuPY**: pip和conda中有多个cupy版本 -> 从pip (`pip uninstall cupy`) **和** conda (`conda uninstall cupy`) 中删除所有cupy包。如果CuPY和你的GPU出现问题，请卸载所有cupy包，并为你的系统(CUDA)配置安装相应的版本。有关帮助，请参见: https://docs.cupy.dev/en/stable/install.html
* 我们提供两个环境文件。第一个是完整的conda导出，用于获取我们用于跟踪环境及其依赖项的所有包(environment_full.yaml)。另一个是清理后的文件，只包含使代码工作的重要包(environment.yaml)。

### Docker镜像
我们创建了一个beta版本的docker镜像。你可以从这里拉取镜像: [ikimhoerst/cellvit:beta](https://hub.docker.com/repository/docker/ikimhoerst/cellvit/general)。docker容器预装了所有要求，旨在运行在linux/amd64平台上。我们在容器中包含了CellViT-SAM-H检查点和此模型的所有分类器。

#### 运行容器
要启动容器，请使用以下命令。将 `/path/to/local/input` 和 `/path/to/local/output` 替换为系统上适当的本地路径，以便将切片加载到容器中并在本地保存结果:
```sh
docker run --name cellvit++ \
  --gpus all \ # 要使用的gpu（默认全部）
  --memory=8g \ # 内存，我们建议更多（16g）
  -v /path/to/local/input:/workspace/CellViT-plus-plus/input-data \ 此目录包含挂载到容器中的输入数据。
  -v /path/to/local/output:/workspace/CellViT-plus-plus/output-data \ 此目录包含从容器挂载的输出数据。
  -it \ # 在工作目录中启动终端
  ikimhoerst/cellvit:beta
```
要运行推理，你可以使用下面 [示例](#示例) 中给出的相同命令。但是，你不再需要传递模型检查点，因为CellViT-SAM-H是容器中的默认模型。请相应地考虑你的输入和输出路径。分类器位于 `./checkpoints/classifier/sam-h` 文件夹中，类似于此git仓库。

#### 使用docker compose
你也可以使用给定的 [`docker-compose.yaml`](./docker-compose.yaml) 文件。将 `/path/to/local/input` 和 `/path/to/local/output` (在volumes下) 替换为系统上适当的本地路径，以便将切片加载到容器中并在本地保存结果。如果出现内存问题，请考虑将内存增加到16g或32g。

启动:
```sh
docker-compose up
```

### 模型检查点
检查点可以从 [Google-Drive](https://drive.google.com/drive/folders/1ujtMcxAr5kYYuvnbglfYZZnRH3ZOli79?usp=sharing) 下载。它们应该放在 `./checkpoints` 文件夹中。分类器检查点已经位于 `./checkpoints/classifier` 文件夹中。不幸的是，由于许可证原因，我们不能共享所有检查点。

## 框架概述

该框架由3个关键组件组成: **CellViT++ 算法**、**细胞分类模块**、**基于Web的WSI查看器**

<div align="center" style="max-width: 400px; margin: 0 auto;">
    <img src="docs/figures/framework.jpeg" style="width: 100%; max-width: 400px;" alt="Example">
        <figcaption style="text-align: center; font-style: italic; padding-top: 5px;">在BioRender中创建。Hörst, F. (2025) https://BioRender.com/t54t384</figcaption>

</div>
我们随后介绍所有模型，从算法开始。

## 推理
推理可以在内存中执行（推荐），也可以使用旧版本（先提取patch）。我们强烈建议使用新的内存版本，因为它更快并且有更多选项。旧脚本仍然存在是出于遗留原因，但并非所有功能都受支持。

模型检查点可以从 [Google-Drive](https://drive.google.com/drive/folders/1ujtMcxAr5kYYuvnbglfYZZnRH3ZOli79?usp=sharing) 下载，应该放在 `./checkpoint` 文件夹中。细胞分类器模块已经在此仓库中提供（参见checkpoints文件夹内的分类器）。

示例在 [下方](#示例) 给出。

推理脚本的关键方面:

> :heavy_plus_sign: 选择你想要提取的内容（json、geojson、graph）
> :heavy_plus_sign: Snappy压缩以节省存储空间
> :heavy_plus_sign: 无需预处理即可处理单个或多个WSI

### CLI（内存版）
> [!CAUTION]
> Ray在不使用环境变量更改GPU ID时会出现问题。我们建议设置环境变量 `export CUDA_VISIBLE_DEVICES=xxx`，而不是使用 `--gpu` 标志更改默认GPU。
.

如果数据已准备好，使用 `cellvit` 文件夹中的 [`detect_cells.py`](cellvit/detect_cells.py) 脚本执行推理:

`python3 ./cellvit/detect_cells.py --OPTIONS`

选项在这里列出（在终端中使用 `--help` 获取帮助）:
```bash
用法: detect_cells.py [-h]
  --model MODEL
  [--binary | --classifier_path CLASSIFIER_PATH]
  [--gpu GPU]
  [--resolution {0.25,0.5}]
  [--enforce_amp]
  [--batch_size BATCH_SIZE]
  --outdir OUTDIR
  [--geojson]
  [--graph]
  [--compression]
  {process_wsi,process_dataset} ...

执行CellViT推理

选项:
  -h, --help            显示此帮助消息并退出
  --binary              使用此选项进行仅细胞检测/分割，不使用分类器。
                        不能与--classifier_path一起使用。（默认: False）
  --classifier_path CLASSIFIER_PATH
                        分类器路径（.pth），用于将PanNuke分类结果替换为新方案。
                        示例分类器可以在./checkpoints/classifiers文件夹中找到。
                        每个分类器的README中提供了带有概述的标签映射。
                        不能与--binary一起使用。（默认: None）
  --gpu GPU             用于推理的Cuda-GPU ID。默认: 0（默认: 0）
  --resolution {0.25,0.5}
                        MPP中的网络分辨率。
                        用于检查patch分辨率，以便我们为网络使用正确的分辨率。
                        我们强烈建议使用0.25，0.50已弃用，将在后续版本中删除。
                        默认: 0.25（默认: 0.25）
  --enforce_amp         是否对推理使用混合精度（强制）。
                        否则使用网络默认训练设置。默认: False（默认: False）
  --batch_size BATCH_SIZE
                        推理批大小。默认: 8（默认: 8）
  --outdir OUTDIR       存储结果的输出目录。（默认: None）
  --geojson             设置此标志以将结果导出为额外的geojson文件，以便将它们加载到QuPath等软件中。（默认: False）
  --graph               设置此标志以将结果导出为包含嵌入（.pt）文件的pytorch图。（默认: False）
  --compression         设置此标志以将结果导出为snappy压缩文件（默认: False）

必需的命名参数:
  --model MODEL         用于推理的模型检查点文件（.pth）。这是分割模型，通常具有PanNuke核类。（默认: None）

子命令:
  在单个WSI文件或整个数据集上执行推理的主运行命令

  {process_wsi,process_dataset}
```

**处理单个WSI**
```bash
process_wsi
  -h, --help            显示此帮助消息并退出
  --wsi_path WSI_PATH   WSI文件的路径
  --wsi_properties WSI_PROPERTIES
                        用于处理的WSI元数据，字段为slide_mpp和magnification。作为JSON字符串提供。
  --preprocessing_config PREPROCESSING_CONFIG
                        包含预处理配置的.yaml文件路径，可选
```
**处理多个WSI**
```bash
process_dataset
  --wsi_folder WSI_FOLDER
                        存储所有WSI的文件夹路径
  --filelist FILELIST   要处理的WSI文件列表。
                        必须是具有一行'path'的.csv文件，表示要处理的所有WSI的路径。
                        此外，可以通过添加
                        两个额外的列来提供WSI属性，名为'slide_mpp'和'magnification'。其他列被丢弃。
  --wsi_extension WSI_EXTENSION
                        用于WSI文件的扩展类型，参见configs.python.config（WSI_EXT）
  --preprocessing_config PREPROCESSING_CONFIG
                        包含预处理配置的.yaml文件路径，可选
```

### 旧版CLI（已弃用）
请不要使用这个，由于可维护性，我们打算很快删除这个版本。如果你仍然想使用它，请查看脚本 `python3 ./cellvit/detect_cells_disk.py --help`

### 示例

在运行示例之前，请通过运行以下命令下载它们:
`python3 ./cellvit/utils/download_example_files.py`
这将下载放在 [`./test_database`](/test_database) 文件夹中的示例文件。

<details>
  <summary>1. 没有图且仅有json的示例</summary>

  ```bash
  python3 ./cellvit/detect_cells.py \
    --model ./checkpoints/CellViT-SAM-H-x40-AMP.pth \
    --outdir ./test-results/x_40/minimal \
    process_wsi \
    --wsi_path ./test_database/x40_svs/JP2K-33003-2.svs
  ```

</details>

<details>
  <summary>2. 没有图但有geojson的示例</summary>

  ```bash
  python3 ./cellvit/detect_cells.py \
    --model ./checkpoints/CellViT-SAM-H-x40-AMP.pth \
    --outdir ./test-results/x_40/full_geojson \
    --geojson \
    process_wsi \
    --wsi_path ./test_database/x40_svs/JP2K-33003-2.svs
  ```

</details>

<details>
  <summary>3. 有图和压缩的示例</summary>

  ```bash
  python3 ./cellvit/detect_cells.py \
    --model ./checkpoints/CellViT-SAM-H-x40-AMP.pth \
    --outdir ./test-results/x_40/compression \
    --geojson \
    --compression \
    --graph \
    process_wsi \
    --wsi_path ./test_database/x40_svs/JP2K-33003-2.svs
  ```

</details>

<details>
  <summary>4. 不同mpp的示例（0.50而不是0.25，使用调整大小）</summary>

  ```bash
  python3 ./cellvit/detect_cells.py \
    --model ./checkpoints/CellViT-SAM-H-x40-AMP.pth \
    --outdir ./test-results/x20/1 \
    process_wsi \
    --wsi_path ./test_database/x20_svs/CMU-1-Small-Region.svs
  ```

</details>

<details>
  <summary>5. 单个文件的元数据传递（覆盖OpenSlide元数据）</summary>

  ```bash
  python3 ./cellvit/detect_cells.py \
    --model ./checkpoints/CellViT-SAM-H-x40-AMP.pth \
    --outdir ./test-results/x20/2 \
    --geojson \
    process_wsi \
    --wsi_path ./test_database/x20_svs/CMU-1-Small-Region.svs \
    --wsi_properties "{\"slide_mpp\": 0.50}"
  ```

</details>

<details>
  <summary>6. tiff文件的元数据传递（没有适当元数据的文件）</summary>

  ```bash
  python3 ./cellvit/detect_cells.py \
    --model ./checkpoints/CellViT-SAM-H-x40-AMP.pth \
    --outdir ./test-results/x_40/minimal \
    process_wsi \
    --wsi_path ./test_database/MIDOG/001_pyramid.tiff
    --wsi_properties "{\"slide_mpp\": 0.25, \"magnification\": 40}"
  ```

</details>

<details>
  <summary>7. 处理具有特定文件类型的整个图像文件夹</summary>

  ```bash
  python3 ./cellvit/detect_cells.py \
    --model ./checkpoints/CellViT-SAM-H-x40-AMP.pth \
    --outdir ./test-results/filelist \
    --geojson \
    process_dataset \
    --filelist ./test_database/MIDOG/example_filelist.csv
  ```

</details>

<details>
  <summary>8. 处理整个文件列表（通过文件列表设置属性）</summary>

  ```bash
  python3 ./cellvit/detect_cells.py \
    --model ./checkpoints/CellViT-SAM-H-x40-AMP.pth \
    --outdir ./test-results/MIDOG/filelist \
    process_dataset \
    --filelist ./test_database/MIDOG/example_filelist.csv
  ```

</details>

<details>
  <summary>9. 使用自定义分类器</summary>

  ```bash
  python3 ./cellvit/detect_cells.py \
    --model ./checkpoints/CellViT-SAM-H-x40-AMP.pth \
    --outdir ./test-results/x_40/minimal \
    --classifier_path ./checkpoints/classifier/sam-h/consep.pth
    process_wsi \
    --wsi_path ./test_database/x40_svs/JP2K-33003-2.svs
  ```

</details>

<details>
  <summary>10. 二值细胞分割（有细胞或没有细胞）</summary>

  ```bash
  python3 ./cellvit/detect_cells.py \
      --model ./checkpoints/CellViT-SAM-H-x40-AMP.pth \
      --outdir ./test-results/x20/binary \
      --binary \
      --geojson \
      process_wsi \
      --wsi_path ./test_database/x20_svs/CMU-1-Small-Region.svs
  ```

</details>

## 重新训练你自己的分类器: 工作流程

### 1. 检测标注

#### 1.1 文件夹结构
要定义检测数据集，你应该具有以下文件夹结构:
```bash
├── label_map.yaml        [可选，但建议用于跟踪你的标签]
├── splits                [包含分割的文件夹]
│   ├── fold_0  
│   │   ├── train.csv
│   │   └── val.csv
│   ├── fold_1
│   │   ├── train.csv
│   ...
├── train                 [训练数据集]
│   ├── images            [png或jpeg格式的训练图像]
│   │   ├── train_1.png
│   │   ├── train_2.png
│   │   ├── train_3.png
│   ...
│   ├── labels            [csv格式的细胞标注]
│   │   ├── train_1.csv
│   │   ├── train_2.csv
│   │   ├── train_3.csv
│   ...  
├── test                  [测试数据集]  
│   ├── images
│   │   ├── test_1.png
│   │   ├── test_2.png
│   ...  
│   ├── labels
│   │   ├── test_1.csv
│   │   ├── test_2.csv
│   ...  
└── train_configs         [配置文件]
    └── ViT256
        ├── fold_0.yaml
        ├── fold_0_sweep.yaml
...
```
**我们提供两个示例数据集:**
- 简单: 256x256 px大小的正方形图像: [`./test_database/training_database/Example-Detection`](test_database/training_database/Example-Detection)
- 高级: 260x288 px大小的非正方形图像: [`./test_database/training_database/Example-Detection-Non-Squared`](test_database/training_database/Example-Detection-Non-Squared)

#### 1.2 数据集要求
- **分割**: 训练和验证分割由splits文件夹中的CSV文件定义。
- **测试图像**: 分离到专用的test文件夹中。
- **图像尺寸**: 转换后必须能被32整除。图像可以是非正方形的（见下文）。
  - 如果你的图像尺寸与256 x 256不同，你需要在训练配置中定义:
    ```yaml
    data:
      dataset: DetectionDataset
      dataset_path: ./test_database/training_database/Example-Detection-larger
      input_shape: [256, 288] # 高度，宽度
      ...
    ```
    这里，input_shape是转换后的形状（网络输入形状）。如果输入图像小于定义的input_shapes，我们执行填充，如果图像大于定义的input_shapes，我们执行中心裁剪。我们在 [`./test_database/training_database/Example-Detection-Non-Squared`](test_database/training_database/Example-Detection-Non-Squared) 文件夹中用形状为(260, 288)的图像演示了这一点
  - input_shape支持的大小: 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024
- **标注**: 包含x、y坐标和标签的CSV文件。标签是从0开始的整数。
- **标签映射**: 建议在文件夹中跟踪你的标签（参见label_map.yaml）。

### 2. 分割标注

#### 2.1 文件夹结构
与上述检测相同的文件夹结构，但标签现在是numpy（*.npy）文件。
- 标签是整数，必须从1开始!!!
- 与上述结构相同，但标签是.npy文件而不是.csv文件
- 我们遵循HoVer-Net定义的标签结构:
  每个numpy文件包含一个实例映射（键: inst_map），每个细胞作为从1开始的唯一整数
  和一个标签映射（键: label_map），具有每个整数的细胞类别（形状...）
- 以下文件夹中给出了示例
- 如果你想查看标签是如何加载的，请查看我们的示例笔记本（一些示例标签的绘图以及图像，以及数据是如何加载的和数据形状）

#### 2.2 数据集要求

- **标签**: 存储为.npy文件（NumPy数组），里面需要有一个包含两个键的字典:
  - **inst_map**: 实例映射，每个细胞作为从1开始的唯一整数（0 = 背景），形状 H x W（高度，宽度）
  - **type_map**: 每个整数的细胞类别（值从1开始，而不是检测中的0）

有关标签可视化和数据加载过程，请参考分割数据集示例中的示例笔记本。

### 3. 查找超参数并训练你的模型
- 定义sweep配置，为两个数据集都提供了示例（例如，[`./test_database/training_database/Example-Detection/train_configs/ViT256/fold_0_sweep.yaml`](/test_database/training_database/Example-Detection/train_configs/ViT256/fold_0_sweep.yaml))
- 运行WandB sweep:
    ```bash
    python3 ./cellvit/train_cell_classifier_head.py --config /path/to/your/config.yaml --sweep
    ```
- 查找最佳HP配置: 运行 `python3 ./script/find_best_hyperparameter.py`:
  ```python3
  用法: find_best_hyperparameter.py [-h] [--metric METRIC] [--minimize] folder

  从sweep文件夹检索最佳超参数配置文件。

  位置参数:
    folder           包含JSON配置文件的sweep文件夹的路径。

  选项:
    -h, --help       显示此帮助消息并退出
    --metric METRIC  要优化的指标。默认: AUROC/Validation
    --minimize       如果指标应该最小化。默认: False（最大化）
  ```
- 不使用sweep运行: 每个数据集文件夹中也给出了不使用sweep的配置示例。训练命令 `python3 ./cellvit/train_cell_classifier_head.py --config /path/to/your/config.yaml`

### 4. 最终评估
评估取决于你的设置。你可以在运行推理脚本时添加其路径来使用你的算法进行推理（参见 [推理](#推理)）。如果你需要计算特定的指标，我们在 [`./cellvit/training/evaluate/inference_cellvit_experiment_detection.py`](/cellvit/training/evaluate/inference_cellvit_experiment_detection.py) 下提供了基于检测的数据集的评估脚本。其CLI在下面的示例中解释，可以通过运行 `python3 ./cellvit/training/evaluate/inference_cellvit_experiment_detection.py --help` 来显示。请注意传递正确的输入形状作为参数列表（高度，宽度）。否则，如果你需要来自其他工作的数据集特定指标，示例评估脚本放在 [`./cellvit/training/evaluate`](cellvit/training/evaluate) 文件夹中。如果你有类似于CoNSeP/HoVer-Net的分割掩码，你可以从CoNSeP脚本开始。

### 示例
我们使用检测数据集来示范我们的工作流程。首先，请从 [Google-Drive](https://drive.google.com/drive/folders/1ujtMcxAr5kYYuvnbglfYZZnRH3ZOli79?usp=sharing) 文件夹（Zenodo）下载ViT256检查点，并将其放在checkpoint文件夹中。然后依次执行以下步骤:
```bash
# 创建logs文件夹
mkdir logs_local

# 运行sweep: 你需要登录到wandb才能使用此功能（提示时选择选项1或2，而不是3（不可视化我的结果）
python3 ./cellvit/train_cell_classifier_head.py --config ./test_database/training_database/Example-Detection/train_configs/ViT256/fold_0_sweep.yaml --sweep

# 找到你的最佳配置
python3 ./scripts/find_best_hyperparameter.py /path/to/your/sweep --metric AUROC/Validation

# 在测试集上运行评估
python3 ./cellvit/training/evaluate/inference_cellvit_experiment_detection.py \
  --logdir /path/to/your/run_log \
  --dataset_path ./test_database/training_database/Example-Detection \
  --cellvit_path ./checkpoints/CellViT-256-x40-AMP.pth \
  --input_shape 256 256
# 请注意给出用于训练的正确输入形状
```

WandB提示: 你可以在wandb网站的Quickstart下找到你的token。

## 可重现性
所有日志都在 [`./logs/Classifiers`](./logs/Classifiers/) 文件夹中。里面有重要的分割配置。对于每个数据集，还有一个jupyter笔记本解释如何准备数据集或在哪里下载文件。所有分类器都是通过运行 [`./cellvit/train_cell_classifier_head.py`](cellvit/train_cell_classifier_head.py) 训练脚本进行训练的。数据集特定的评估在 [`./cellvit/training/evaluate`](cellvit/training/evaluate) 文件夹中为每个数据集提供。

### 数据集准备

所有数据集准备信息可以在 [`./logs/Datasets`](./logs/Datasets/) 文件夹中找到

### 训练配置
对于每个数据集，使用了略有不同的配置。请查看每个数据集的日志文件夹 [`./logs/Classifiers`](./logs/Classifiers/) 以获取示例。如果你想重新训练和重现结果，首先准备数据集，然后调整相应配置文件中的路径。

### 有用的脚本

- 训练: [`./cellvit/train_cell_classifier_head.py`](./cellvit/train_cell_classifier_head.py) 和 [`./cellvit/training/trainer/`](./cellvit/training/trainer/)
- 评估: [`./cellvit/training/evaluate/`](./cellvit/training/evaluate/)
- 其他: [`./scripts`](./scripts/)

## 基于Web的查看器
查看器已Docker化，可以使用 `docker compose -f ./viewer/docker-compose-deploy.yaml up` 启动。然后在 `http:localhost` 下访问网站。我们支持所有OpenSlide格式，包括DICOM图像。但是，DICOM图像必须作为压缩的zip文件夹中的单个文件上传。检测和/或轮廓可以作为.geojson或压缩的.geojson.snappy文件上传。

**示例**: 要测试查看器，请使用 [./test_database/x40_svs](/test_database/x40_svs) 内的文件

## 标注工具
### 设置
1. 要使查看工具运行，你首先需要准备你的数据集。
2. 请在 [annotation_tool](annotation_tool) 中的.env文件中更改用户（管理员和标注者）的设置
3. 在 [docker-compose.yaml](annotation_tool/docker-compose.yaml) 文件中添加数据集的路径
4. 使用以下命令启动工具: `docker compose -f ./annotation_tool/docker-compose.yaml up`
5. 在 [127.0.0.1:8000](127.0.0.1:8000) 下访问工具
6. 如果这是你的首次设置，以管理员身份登录并将路径设置为 `/dataset`

<details>
  <summary>数据集准备</summary>


   - 在目录中，必须有一个配置文件和一个名为 **dataset** 的文件夹。
   - dataset文件夹必须包含细胞图像，并可选地为每个细胞提供上下文图像，全部为 **PNG格式** 或 **jpg格式**。
   - 细胞图像及其对应的上下文图像的文件名应该相同，除了细胞图像应以 **roi.png** 结尾，上下文图像应以 **context** 结尾，例如 **context.png**。
   - 图像的边界框可以编码在其名称中。例如，在文件名 *wsi_01_R_2_C_00_cell_9935_bb_118_120_139_137_bb2_451_365_472_382_image.png* 中，部分 *bb_18_120_139_137_* 表示细胞图像的边界框坐标（左上xy，右下xy），*bb2_451_365_472_382_* 表示上下文图像中细胞图像的坐标。对于图像，WSI首先被分成区域，每个上下文图像从区域中裁剪。因此名称包含区域id R、上下文图像id C和细胞名称（图像名称）*wsi_01_R_2_C_00_cell_9935*。
</details>

<details>
  <summary>配置文件</summary>
   - 可以在主目录中包含 `config.yaml` 文件。通过 `config.yaml`，用户可以定义标签。
   - **YAML** 格式的配置文件可以放在主目录中（dataset文件夹也在那里）。
   - 目前，用户可以提供两种配置:
       1. **label codes**: 每个标签必须分配一个整数。
       2. **draw_bounding_box**: 应设置为 `True` 或 `False`。
       3. 如果没有给出配置文件，则使用以下默认配置:
           ```yaml
           label_codes: {"Tumor": 1, "Non-Tumor": 0, "Apoptosis": "A1"}
           draw_bounding_box: False
           ```
       注意: 在此示例中，标注过程使用了6个标签

</details>

<details>
  <summary>标注过程</summary>
   - 一旦数据集的路径被上传，用户将被重定向到主页。
   - 在主页上，图像及其上下文图像以随机顺序从数据集目录中选择。
   - 标注图像后，下一个图像将自动加载。
   - 如果无法标注图像，可以使用 `Skip Image` 按钮跳过它。
   - 用户可以通过点击 `Back to Last Annotation` 按钮编辑他们的上一个标注。
   - 目录中剩余图像的数量显示在细胞图像下方。
   - 通过勾选 `Annotate All`，上下文图像中的所有剩余细胞将一次性标注。
   - annotations.json文件可以从配置选项卡下载
</details>


我们在annotation_tool文件夹中提供了一个 **示例数据集**，可以开箱即用。初始启动后，只需在网站上输入路径 `/dataset`。登录凭据在 [annotation_tool/.env](annotation_tool/.env) 文件中（用户: admin，密码: admin1234）。
在接受后，我们将添加脚本将cellvit标注转换为标注工具数据集格式。

## 引用

**CellViT++**
```latex
@misc{hörst2025cellvitenergyefficientadaptivecell,
      title={CellViT++: Energy-Efficient and Adaptive Cell Segmentation and Classification Using Foundation Models}, 
      author={Fabian Hörst and Moritz Rempe and Helmut Becker and Lukas Heine and Julius Keyl and Jens Kleesiek},
      year={2025},
      eprint={2501.05269},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.05269}, 
}
```


**CellViT**
```latex
@ARTICLE{Horst2024,
  title    = "{CellViT}: Vision Transformers for precise cell segmentation and
              classification",
  author   = "H{\"o}rst, Fabian and Rempe, Moritz and Heine, Lukas and Seibold,
              Constantin and Keyl, Julius and Baldini, Giulia and Ugurel, Selma
              and Siveke, Jens and Gr{\"u}nwald, Barbara and Egger, Jan and
              Kleesiek, Jens",
  journal  = "Med. Image Anal.",
  volume   =  94,
  pages    = "103143",
  month    =  may,
  year     =  2024,
  keywords = "Cell segmentation; Deep learning; Digital pathology; Vision
              transformer",
  language = "en"
}
```
