"""Phase 4A：只评估 Retrieval，不调用 LLM，也不修改任何检索配置。"""

import csv
import json
import sys
from pathlib import Path
from statistics import fmean
from time import perf_counter

from sentence_transformers import SentenceTransformer


# 直接运行 evaluation/evaluate_retrieval.py 时，Python 默认只把 evaluation/
# 放入模块搜索路径。这里加入项目根目录，以便复用现有 query.py。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from query import (  # noqa: E402  # 必须在加入项目根目录后导入
    DEFAULT_TOP_K,
    INDEX_PATH,
    MAPPING_PATH,
    MIN_SCORE,
    create_query_embedding,
    load_index,
    load_mapping,
)


EVALUATION_DIR = Path(__file__).resolve().parent
EVAL_CASES_PATH = EVALUATION_DIR / "eval_cases.json"
RESULTS_JSON_PATH = EVALUATION_DIR / "retrieval_results.json"
RESULTS_CSV_PATH = EVALUATION_DIR / "retrieval_results.csv"

EVAL_CASE_FIELDS = {
    "id",
    "category",
    "question",
    "should_answer",
    "expected_topics",
    "expected_sources",
    "risk_level",
}

RESULT_FIELDS = [
    "id",
    "category",
    "question",
    "should_answer",
    "top1_score",
    "retrieval_passed",
    "top1_source",
    "top2_source",
    "top3_source",
    "expected_sources",
    "source_hit_at_1",
    "source_hit_at_3",
    "content_hit_at_1",
    "content_hit_at_3",
    "top1_topic_match_count",
    "refusal_correct",
    "retrieval_latency_ms",
]


def load_eval_cases(path: Path) -> list[dict]:
    """读取并校验固定测试集，但不修改其中任何内容。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到测试集：{path}")

    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("测试集必须是非空 JSON 数组")

    seen_ids: set[str] = set()
    for position, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"第 {position} 个测试案例不是 JSON 对象")
        if set(case) != EVAL_CASE_FIELDS:
            raise ValueError(f"测试案例 {case.get('id', position)} 的字段不完整或有多余字段")
        if not isinstance(case["should_answer"], bool):
            raise ValueError(f"测试案例 {case['id']} 的 should_answer 必须是布尔值")
        if not isinstance(case["expected_sources"], list):
            raise ValueError(f"测试案例 {case['id']} 的 expected_sources 必须是数组")
        if not isinstance(case["expected_topics"], list):
            raise ValueError(f"测试案例 {case['id']} 的 expected_topics 必须是数组")
        if case["id"] in seen_ids:
            raise ValueError(f"测试案例 ID 重复：{case['id']}")
        seen_ids.add(case["id"])

    return cases


def evaluate_case(
    case: dict,
    model: SentenceTransformer,
    index,
    mapping: dict,
) -> dict:
    """执行单题 Retrieval，并计算该题的客观指标。"""
    if index.ntotal == 0:
        raise ValueError("FAISS 索引中没有向量，无法进行 Retrieval Evaluation")

    # latency 只覆盖：query embedding、FAISS search、metadata lookup。
    start_time = perf_counter()

    query_embedding = create_query_embedding(
        case["question"],
        model,
        mapping["embedding_dimension"],
    )

    actual_top_k = min(DEFAULT_TOP_K, index.ntotal)
    scores, vector_ids = index.search(query_embedding, actual_top_k)

    retrieved_chunks: list[dict | None] = []
    for vector_id in vector_ids[0]:
        if int(vector_id) < 0:
            retrieved_chunks.append(None)
            continue

        chunk = mapping["chunks"][int(vector_id)]
        retrieved_chunks.append(
            {
                "source": chunk["metadata"].get("来源文件"),
                "text": chunk["text"],
            }
        )

    latency_ms = (perf_counter() - start_time) * 1000

    # 当前索引有 9 个向量，因此会得到完整 Top-3；补 None 让脚本也能明确处理小索引。
    retrieved_chunks.extend([None] * (DEFAULT_TOP_K - len(retrieved_chunks)))
    top1_chunk, top2_chunk, top3_chunk = retrieved_chunks[:DEFAULT_TOP_K]

    def get_source(retrieved_chunk: dict | None) -> str | None:
        return None if retrieved_chunk is None else retrieved_chunk["source"]

    top1_source = get_source(top1_chunk)
    top2_source = get_source(top2_chunk)
    top3_source = get_source(top3_chunk)

    top1_score = float(scores[0][0])
    retrieval_passed = top1_score >= MIN_SCORE
    should_answer = case["should_answer"]
    expected_sources = case["expected_sources"]
    expected_topics = case["expected_topics"]

    def topic_match_count(retrieved_chunk: dict | None) -> int:
        """使用严格字面子串匹配，统计 chunk 命中的 expected_topics 数量。"""
        if retrieved_chunk is None:
            return 0
        return sum(topic in retrieved_chunk["text"] for topic in expected_topics)

    def is_content_relevant(retrieved_chunk: dict | None) -> bool:
        """内容相关必须同时命中预期来源和至少一个预期主题。"""
        if retrieved_chunk is None:
            return False
        return (
            retrieved_chunk["source"] in expected_sources
            and topic_match_count(retrieved_chunk) > 0
        )

    top1_topic_match_count = topic_match_count(top1_chunk)

    if should_answer:
        source_hit_at_1 = top1_source in expected_sources
        source_hit_at_3 = any(
            source in expected_sources
            for source in (top1_source, top2_source, top3_source)
        )
        content_hit_at_1 = is_content_relevant(top1_chunk)
        content_hit_at_3 = any(
            is_content_relevant(retrieved_chunk)
            for retrieved_chunk in (top1_chunk, top2_chunk, top3_chunk)
        )
        refusal_correct = None
    else:
        source_hit_at_1 = None
        source_hit_at_3 = None
        content_hit_at_1 = None
        content_hit_at_3 = None
        refusal_correct = not retrieval_passed

    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "should_answer": should_answer,
        "top1_score": round(top1_score, 6),
        "retrieval_passed": retrieval_passed,
        "top1_source": top1_source,
        "top2_source": top2_source,
        "top3_source": top3_source,
        "expected_sources": expected_sources,
        "source_hit_at_1": source_hit_at_1,
        "source_hit_at_3": source_hit_at_3,
        "content_hit_at_1": content_hit_at_1,
        "content_hit_at_3": content_hit_at_3,
        "top1_topic_match_count": top1_topic_match_count,
        "refusal_correct": refusal_correct,
        "retrieval_latency_ms": round(latency_ms, 3),
    }


def calculate_summary(results: list[dict]) -> dict:
    """严格按任务定义计算整体 Retrieval 指标。"""
    answer_results = [result for result in results if result["should_answer"]]
    no_answer_results = [result for result in results if not result["should_answer"]]

    def ratio(true_count: int, total_count: int) -> float | None:
        return true_count / total_count if total_count else None

    def average(values: list[float]) -> float | None:
        return fmean(values) if values else None

    summary = {
        "answer_case_count": len(answer_results),
        "no_answer_case_count": len(no_answer_results),
        "source_hit_at_1": ratio(
            sum(result["source_hit_at_1"] is True for result in answer_results),
            len(answer_results),
        ),
        "source_hit_at_3": ratio(
            sum(result["source_hit_at_3"] is True for result in answer_results),
            len(answer_results),
        ),
        "content_hit_at_1": ratio(
            sum(result["content_hit_at_1"] is True for result in answer_results),
            len(answer_results),
        ),
        "content_hit_at_3": ratio(
            sum(result["content_hit_at_3"] is True for result in answer_results),
            len(answer_results),
        ),
        "retrieval_answer_pass_rate": ratio(
            sum(result["retrieval_passed"] is True for result in answer_results),
            len(answer_results),
        ),
        "no_answer_refusal_accuracy": ratio(
            sum(result["refusal_correct"] is True for result in no_answer_results),
            len(no_answer_results),
        ),
        "average_answer_top1_score": average(
            [result["top1_score"] for result in answer_results]
        ),
        "average_no_answer_top1_score": average(
            [result["top1_score"] for result in no_answer_results]
        ),
        "average_retrieval_latency_ms": average(
            [result["retrieval_latency_ms"] for result in results]
        ),
    }

    # 控制结果文件中的小数长度，同时保持所有汇总值由逐题记录直接计算得出。
    for key, value in summary.items():
        if isinstance(value, float):
            summary[key] = round(value, 6)

    return summary


def save_results(results: list[dict], summary: dict, mapping: dict, index) -> None:
    """保存完整 JSON 和适合 Excel 打开的 UTF-8 BOM CSV。"""
    json_payload = {
        "configuration": {
            "top_k": DEFAULT_TOP_K,
            "min_score": MIN_SCORE,
            "embedding_model": mapping["embedding_model"],
            "embedding_dimension": mapping["embedding_dimension"],
            "similarity": mapping["similarity"],
            "faiss_index_type": type(index).__name__,
            "latency_scope": "query_embedding + faiss_search + metadata_lookup",
        },
        "summary": summary,
        "results": results,
    }
    RESULTS_JSON_PATH.write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with RESULTS_CSV_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for result in results:
            csv_row = dict(result)
            csv_row["expected_sources"] = json.dumps(
                result["expected_sources"],
                ensure_ascii=False,
            )
            writer.writerow(csv_row)


def yes_or_no(value: bool) -> str:
    """将布尔结果转成终端简表中的 Yes/No。"""
    return "Yes" if value else "No"


def print_case_result(result: dict) -> None:
    """打印一行逐题结果。"""
    status = "PASS" if result["retrieval_passed"] else "REFUSED"
    base = f"{result['id']} | {status} | score={result['top1_score']:.6f}"

    if result["should_answer"]:
        print(
            f"{base} | Source@1={yes_or_no(result['source_hit_at_1'])}"
            f" | Source@3={yes_or_no(result['source_hit_at_3'])}"
            f" | Content@1={yes_or_no(result['content_hit_at_1'])}"
            f" | Content@3={yes_or_no(result['content_hit_at_3'])}"
            f" | Top1Topics={result['top1_topic_match_count']}"
        )
    else:
        print(
            f"{base} | Correct Refusal="
            f"{yes_or_no(result['refusal_correct'])}"
        )


def format_rate(value: float | None) -> str:
    """把 0～1 比例格式化为百分比。"""
    return "N/A" if value is None else f"{value:.2%}"


def print_summary(summary: dict) -> None:
    """打印 Source Hit、Content Hit 及其他 Retrieval 汇总指标。"""
    print("\n" + "=" * 72)
    print("Retrieval Evaluation 汇总")
    print("=" * 72)
    print(f"1. 有答案题数量：{summary['answer_case_count']}")
    print(f"2. 无答案题数量：{summary['no_answer_case_count']}")
    print(f"3. Source Hit@1：{format_rate(summary['source_hit_at_1'])}")
    print(f"4. Source Hit@3：{format_rate(summary['source_hit_at_3'])}")
    print(f"5. Content Hit@1：{format_rate(summary['content_hit_at_1'])}")
    print(f"6. Content Hit@3：{format_rate(summary['content_hit_at_3'])}")
    print(
        "7. Retrieval Answer Pass Rate："
        f"{format_rate(summary['retrieval_answer_pass_rate'])}"
    )
    print(
        "8. No-answer Refusal Accuracy："
        f"{format_rate(summary['no_answer_refusal_accuracy'])}"
    )
    print(
        "9. 有答案问题的平均 Top-1 score："
        f"{summary['average_answer_top1_score']:.6f}"
    )
    print(
        "10. 无答案问题的平均 Top-1 score："
        f"{summary['average_no_answer_top1_score']:.6f}"
    )
    print(
        "11. 平均 retrieval latency："
        f"{summary['average_retrieval_latency_ms']:.3f} ms"
    )


def main() -> None:
    """一次加载模型与索引，然后按固定顺序评估全部案例。"""
    cases = load_eval_cases(EVAL_CASES_PATH)

    # 这三个对象都在逐题循环外加载，整个 evaluation run 只加载一次。
    mapping = load_mapping(MAPPING_PATH)
    index = load_index(INDEX_PATH, mapping)
    print(f"正在加载 embedding 模型：{mapping['embedding_model']}")
    model = SentenceTransformer(mapping["embedding_model"])

    print(
        f"开始评估 {len(cases)} 个案例：Top-K={DEFAULT_TOP_K}, "
        f"MIN_SCORE={MIN_SCORE:.2f}\n"
    )

    results: list[dict] = []
    for case in cases:
        result = evaluate_case(case, model, index, mapping)
        results.append(result)
        print_case_result(result)

    summary = calculate_summary(results)
    save_results(results, summary, mapping, index)
    print_summary(summary)

    print("\n结果文件：")
    print(f"- {RESULTS_JSON_PATH}")
    print(f"- {RESULTS_CSV_PATH}")


if __name__ == "__main__":
    main()
