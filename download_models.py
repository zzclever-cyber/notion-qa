"""
从 ModelScope（魔搭社区）下载模型到本地 models/ 目录
国内网络秒下，替代 HuggingFace 直连

运行一次即可：python download_models.py
"""
from pathlib import Path
import os

MODELS_DIR = Path(__file__).parent / "models"

MODELS = {
    # 模型名 → ModelScope 数据集 ID
    "BAAI/bge-large-zh-v1.5": "BAAI/bge-large-zh-v1.5",
    "BAAI/bge-reranker-large": "BAAI/bge-reranker-large",
}


def download_via_modelscope():
    """使用 ModelScope SDK 下载"""
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("请先安装 modelscope：pip install modelscope")
        return False

    for model_name, ms_id in MODELS.items():
        target_dir = MODELS_DIR / model_name
        if target_dir.exists() and any(target_dir.iterdir()):
            print(f"[跳过] {model_name} 已存在: {target_dir}")
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"[下载] {model_name} ...")
        try:
            downloaded = snapshot_download(ms_id, cache_dir=str(MODELS_DIR))
            print(f"  → {downloaded}")
        except Exception as e:
            print(f"  [失败] {e}")
            return False

    print("\n全部模型下载完成！可以运行 python build_index.py 了")
    return True


def download_via_hf_mirror():
    """备用方案：通过 HF 镜像 + huggingface_hub 下载"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("请先安装 huggingface_hub：pip install huggingface_hub")
        return False

    for model_name in MODELS:
        target_dir = MODELS_DIR / model_name
        if target_dir.exists() and any(target_dir.iterdir()):
            print(f"[跳过] {model_name} 已存在: {target_dir}")
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"[下载] {model_name} (HF Mirror) ...")
        try:
            downloaded = snapshot_download(
                model_name,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
                endpoint="https://hf-mirror.com",
            )
            print(f"  → {downloaded}")
        except Exception as e:
            print(f"  [失败] {e}")
            return False

    print("\n全部模型下载完成！")
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("RAG Agent — 模型下载工具")
    print(f"目标目录: {MODELS_DIR.absolute()}")
    print("=" * 50)

    # 先尝试 ModelScope，失败则 HF Mirror
    if not download_via_modelscope():
        print("\nModelScope 下载失败，尝试 HF Mirror ...")
        if not download_via_hf_mirror():
            print("\n两种方式都失败了，请手动下载模型放到 models/ 目录下")
            print("ModelScope: https://modelscope.cn/models/BAAI/bge-large-zh-v1.5")
            print("ModelScope: https://modelscope.cn/models/BAAI/bge-reranker-large")
