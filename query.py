"""Phase 2：把用户问题转成向量，并从 FAISS 中检索相似文本片段。"""

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# 所有路径都以当前脚本所在目录为基准，因此可以从任意工作目录运行本脚本。
PROJECT_ROOT = Path(__file__).resolve().parent
INDEX_PATH = PROJECT_ROOT / "vector_store" / "faiss.index"
MAPPING_PATH = PROJECT_ROOT / "vector_store" / "chunks_metadata.json"
DEFAULT_TOP_K = 3
MIN_SCORE = 0.40

# Phase 1 使用“归一化向量 + 内积”。查询端只接受同一种相似度配置。
EXPECTED_SIMILARITY = "inner_product_on_normalized_vectors"


def load_mapping(mapping_path: Path) -> dict:
    """加载 chunk/metadata 映射，并检查它是否与查询流程兼容。"""
    if not mapping_path.exists():
        raise FileNotFoundError(f"找不到映射文件：{mapping_path}")

    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    required_fields = {
        "embedding_model",
        "embedding_dimension",
        "similarity",
        "chunk_count",
        "chunks",
    }
    missing_fields = required_fields - mapping.keys()
    if missing_fields:
        raise ValueError(f"映射文件缺少字段：{sorted(missing_fields)}")

    if mapping["similarity"] != EXPECTED_SIMILARITY:
        raise ValueError(
            "查询端要求使用与建库阶段相同的归一化内积，但映射文件中的配置为："
            f"{mapping['similarity']}"
        )

    chunks = mapping["chunks"]
    if mapping["chunk_count"] != len(chunks):
        raise ValueError("映射文件中的 chunk_count 与实际 chunks 数量不一致")

    # IndexFlatIP 返回的是向量在索引中的位置，因此必须保证列表位置与 chunk_id 一致。
    for position, record in enumerate(chunks):
        if record.get("chunk_id") != position:
            raise ValueError(
                f"映射顺序错误：列表位置 {position} 的 chunk_id 不是 {position}"
            )

    return mapping


def load_index(index_path: Path, mapping: dict):
    """加载 FAISS 索引，并验证索引和映射文件属于同一批数据。"""
    if not index_path.exists():
        raise FileNotFoundError(f"找不到 FAISS 索引：{index_path}")

    index = faiss.read_index(str(index_path))
    if index.metric_type != faiss.METRIC_INNER_PRODUCT:
        raise ValueError("FAISS 索引不是建库阶段使用的内积（Inner Product）索引")
    if index.d != mapping["embedding_dimension"]:
        raise ValueError(
            f"索引维度为 {index.d}，映射文件记录的维度为 "
            f"{mapping['embedding_dimension']}"
        )
    if index.ntotal != mapping["chunk_count"]:
        raise ValueError(
            f"索引包含 {index.ntotal} 个向量，映射文件包含 "
            f"{mapping['chunk_count']} 个 chunks"
        )

    return index


def create_query_embedding(
    question: str,
    model: SentenceTransformer,
    expected_dimension: int,
) -> np.ndarray:
    """使用与建库阶段相同的方式，把一个问题转换成归一化向量。"""
    # 输入使用列表，因此输出形状是 (1, embedding_dimension)，可直接交给 FAISS。
    query_embedding = model.encode(
        [question],
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)

    if query_embedding.shape != (1, expected_dimension):
        raise ValueError(
            f"问题向量形状为 {query_embedding.shape}，预期为 "
            f"(1, {expected_dimension})"
        )

    return query_embedding


def search_similar_chunks(
    index,
    query_embedding: np.ndarray,
    chunks: list[dict],
    top_k: int,
    min_score: float = MIN_SCORE,
    show_rejection_message: bool = True,
) -> list[dict]:
    """执行向量检索，并把 FAISS 返回的编号映射回 chunk 原文。"""
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    # 如果 top_k 大于库中的向量数量，只检索实际存在的数量，避免得到 -1 编号。
    actual_top_k = min(top_k, index.ntotal)
    if actual_top_k == 0:
        if show_rejection_message:
            print("\n知识库中没有可供检索的向量。")
        return []

    # scores 和 vector_ids 的形状都是 (问题数量, actual_top_k)。
    # 这里只有一个问题，所以读取第 0 行。
    scores, vector_ids = index.search(query_embedding, actual_top_k)
    top1_score = float(scores[0][0])
    if top1_score < min_score:
        if show_rejection_message:
            print(
                "\n未找到足够相关的知识。"
                f"最高相似度为 {top1_score:.6f}，"
                "建议补充问题信息或转为人工处理。"
            )
        return []

    results: list[dict] = []
    for score, vector_id in zip(scores[0], vector_ids[0]):
        # IndexFlatIP 没有自定义 ID；返回编号就是 add 时的向量顺序。
        chunk = chunks[int(vector_id)]
        results.append(
            {
                "similarity_score": float(score),
                "chunk": chunk,
            }
        )

    return results 


def print_results(question: str, results: list[dict]) -> None:
    """以便于阅读的格式打印检索结果，不生成任何 RAG 回答。"""
    print("\n" + "=" * 72)
    print(f"查询问题：{question}")
    print(f"返回结果数：{len(results)}")
    print("=" * 72)

    for rank, result in enumerate(results, start=1):
        chunk = result["chunk"]
        metadata = chunk["metadata"]

        print(f"\n[排名 {rank}]")
        print(f"similarity score：{result['similarity_score']:.6f}")
        print(f"来源文件：{metadata.get('来源文件', '未知')}")
        print(f"设备型号：{metadata.get('设备型号', '未知')}")
        print(f"版本：{metadata.get('版本', '未知')}")
        print(f"文档类型：{metadata.get('文档类型', '未知')}")
        print("chunk 原文：")
        print(chunk["text"])
        print("-" * 72)


def parse_args() -> argparse.Namespace:
    """读取可选的 Top-K 参数；未指定时默认为 3。"""
    parser = argparse.ArgumentParser(description="企业 AI 知识助手：语义检索演示")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"返回最相似的 chunk 数量，默认 {DEFAULT_TOP_K}",
    )
    return parser.parse_args()


def main() -> None:
    """执行 Phase 2：读取问题、生成问题向量、检索并打印原始 chunk。"""
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k 必须大于 0")

    question = input("请输入中文问题：").strip()
    if not question:
        raise ValueError("问题不能为空")

    mapping = load_mapping(MAPPING_PATH)
    index = load_index(INDEX_PATH, mapping)

    # 模型名直接读取自建库时保存的映射，保证 query 和文档使用同一模型。
    model_name = mapping["embedding_model"]
    print(f"正在加载与建库阶段相同的 embedding 模型：{model_name}")
    model = SentenceTransformer(model_name)

    query_embedding = create_query_embedding(
        question,
        model,
        mapping["embedding_dimension"],
    )
    results = search_similar_chunks(
        index,
        query_embedding,
        mapping["chunks"],
        args.top_k,
    )
    if not results:
        return
    print_results(question, results)


if __name__ == "__main__":
    main()
