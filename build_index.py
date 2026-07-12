"""
索引构建脚本
从知识库JSON文件构建 FAISS 向量索引和 BM25 语料库
"""
import asyncio
import json
import time
from pathlib import Path
from config.settings import settings
from retrieval.bm25_retriever import BM25Retriever
from retrieval.faiss_retriever import FAISSRetriever
from utils.logger import setup_logger, log


async def build_indexes_async(knowledge_base_path: Path = None, output_dir: Path = None):
    """
    构建所有索引（异步版本）
    """
    kb_path = knowledge_base_path or settings.knowledge_base_path
    out_dir = output_dir or settings.data_dir

    log.info(f"开始构建索引: KB={kb_path}, 输出={out_dir}")

    # 加载知识库文档
    if not kb_path.exists():
        log.error(f"知识库文件不存在: {kb_path}")
        return False

    with open(kb_path, "r", encoding="utf-8") as f:
        documents = json.load(f)

    log.info(f"已加载 {len(documents)} 篇文档")
    for doc in documents:
        log.info(f"  - [{doc['id']}] {doc['title']} ({doc['category']})")

    # 构建 FAISS 索引（异步 — API 模式不阻塞）
    t0 = time.time()
    faiss_retriever = FAISSRetriever()
    await faiss_retriever.build_index(documents)
    faiss_retriever.save(out_dir / "faiss_index")
    t_faiss = time.time() - t0
    log.info(f"FAISS 索引构建完成，耗时: {t_faiss:.1f}s")

    # 构建 BM25 索引
    t0 = time.time()
    bm25_retriever = BM25Retriever()
    bm25_retriever.build_index(documents)
    bm25_retriever.save(settings.bm25_corpus_path)
    t_bm25 = time.time() - t0
    log.info(f"BM25 索引构建完成，耗时: {t_bm25:.1f}s")

    # 生成测评数据集
    from evaluation.dataset import create_default_dataset
    log.info("生成默认测评数据集...")
    ds = create_default_dataset(settings.eval_dataset_path)
    log.info(f"测评数据集已生成: {ds.size} 条样本")
    log.info(f"分布: {ds.type_distribution}")

    log.info("=" * 50)
    log.info("所有索引构建完成！")
    log.info(f"  FAISS: {out_dir / 'faiss_index'} (维度={faiss_retriever.dimension})")
    log.info(f"  BM25:  {settings.bm25_corpus_path}")
    log.info(f"  评测集: {settings.eval_dataset_path}")
    return True


def build_indexes(knowledge_base_path: Path = None, output_dir: Path = None):
    """同步入口（兼容命令行直接调用）"""
    return asyncio.run(build_indexes_async(knowledge_base_path, output_dir))


if __name__ == "__main__":
    setup_logger()
    success = build_indexes()
    if success:
        print("\n✅ 索引构建成功！可以运行 main.py 启动服务。")
    else:
        print("\n❌ 索引构建失败，请检查日志。")
