"""Phase 4B：评估 Retrieval → Threshold → DeepSeek → Final Answer 完整链路。"""

import csv
import json
import os
import sys
import time
from pathlib import Path
from statistics import fmean
from time import perf_counter

import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# 允许从项目根目录直接运行 evaluation/evaluate_rag.py，并复用现有模块。
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
from rag import (  # noqa: E402
    DEEPSEEK_API_URL,
    DEFAULT_DEEPSEEK_MODEL,
    ENV_PATH,
    REQUEST_TIMEOUT,
    RETRIEVAL_REJECTION,
    SYSTEM_PROMPT,
    DeepSeekAPIError,
    build_context,
    build_user_prompt,
)


EVALUATION_DIR = Path(__file__).resolve().parent
EVAL_CASES_PATH = EVALUATION_DIR / "eval_cases.json"
RAG_RESULTS_JSON_PATH = EVALUATION_DIR / "rag_results.json"
RAG_RESULTS_CSV_PATH = EVALUATION_DIR / "rag_results.csv"
MANUAL_REVIEW_CSV_PATH = EVALUATION_DIR / "manual_review.csv"

# 只在实际调用过 API 的题目之间暂停；暂停时间不计入任何单题 latency。
API_DELAY_SECONDS = 1.0
REFUSAL_OPENING_CHARS = 200
REFUSAL_MAX_START_OFFSET = 80

# 这些是明确表示“资料不足/无法回答”的短语，不使用 LLM Judge。
REFUSAL_PHRASES = (
    "现有知识库中没有足够依据",
    "知识库中没有足够依据",
    "没有足够依据回答",
    "现有资料不足以回答",
    "提供的资料不足以回答",
    "信息不足以回答",
    "无法回答",
    "无法确定",
    "无法根据现有知识库",
    "无法依据现有知识库",
    "无法从现有资料",
    "现有知识库中未提供",
    "知识库中未提供",
    "资料中未提供",
)
SUPPLEMENT_PHRASES = ("建议补充资料", "建议补充信息")
HUMAN_HANDOFF_PHRASES = ("转人工", "人工处理")

RESULT_FIELDS = [
    "id",
    "category",
    "question",
    "should_answer",
    "risk_level",
    "retrieval_passed",
    "top1_score",
    "retrieved_sources",
    "retrieved_scores",
    "retrieved_chunks",
    "context_sent_to_llm",
    "final_answer",
    "final_refused",
    "behavior_correct",
    "api_called",
    "error",
    "finish_reason",
    "has_reasoning_content",
    "reasoning_content_length",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "retrieval_latency_ms",
    "llm_latency_ms",
    "total_latency_ms",
    "manual_answer_score",
    "manual_citation_score",
    "manual_guardrail_score",
    "manual_notes",
]

MANUAL_REVIEW_FIELDS = [
    "id",
    "question",
    "should_answer",
    "final_refused",
    "final_answer",
    "retrieved_sources",
    "manual_answer_score",
    "manual_citation_score",
    "manual_guardrail_score",
    "manual_notes",
]


def load_eval_cases(path: Path) -> list[dict]:
    """读取固定测试集，不修改测试内容。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到测试集：{path}")

    cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("测试集必须是非空 JSON 数组")

    required_fields = {
        "id",
        "category",
        "question",
        "should_answer",
        "expected_topics",
        "expected_sources",
        "risk_level",
    }
    seen_ids: set[str] = set()
    for position, case in enumerate(cases, start=1):
        if not isinstance(case, dict) or set(case) != required_fields:
            raise ValueError(f"第 {position} 个测试案例字段不符合固定格式")
        if not isinstance(case["should_answer"], bool):
            raise ValueError(f"测试案例 {case['id']} 的 should_answer 必须是布尔值")
        if case["id"] in seen_ids:
            raise ValueError(f"测试案例 ID 重复：{case['id']}")
        seen_ids.add(case["id"])

    return cases


def retrieve_chunks_for_evaluation(
    question: str,
    model: SentenceTransformer,
    index,
    mapping: dict,
) -> tuple[list[dict], float, bool, float]:
    """执行与 query.py 一致的归一化 embedding 和 FAISS Top-K 检索。"""
    if index.ntotal == 0:
        raise ValueError("FAISS 索引中没有向量")

    # 这里只统计 query embedding、FAISS search 和 metadata/chunk lookup。
    start_time = perf_counter()
    query_embedding = create_query_embedding(
        question,
        model,
        mapping["embedding_dimension"],
    )

    actual_top_k = min(DEFAULT_TOP_K, index.ntotal)
    scores, vector_ids = index.search(query_embedding, actual_top_k)

    results: list[dict] = []
    for score, vector_id in zip(scores[0], vector_ids[0]):
        if int(vector_id) < 0:
            continue
        results.append(
            {
                # IndexFlatIP 返回的编号就是向量加入索引时的位置。
                "vector_id": int(vector_id),
                "similarity_score": float(score),
                "chunk": mapping["chunks"][int(vector_id)],
            }
        )

    retrieval_latency_ms = (perf_counter() - start_time) * 1000
    if not results:
        raise ValueError("FAISS 没有返回任何 Retrieval 结果")

    top1_score = results[0]["similarity_score"]
    retrieval_passed = top1_score >= MIN_SCORE
    return results, top1_score, retrieval_passed, retrieval_latency_ms


def _usage_integer(usage: dict, key: str) -> int | None:
    """安全读取 token usage；缺失时保留 null，而不是猜测数值。"""
    value = usage.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def call_deepseek_for_evaluation(
    api_key: str,
    model_name: str,
    question: str,
    context: str,
) -> dict:
    """使用现有 Grounded Prompt 调用一次 DeepSeek，并额外保留 token usage。"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(question, context),
            },
        ],
        "stream": False,
        "thinking": {"type": "disabled"},
        # 与现有 rag.py 的生成配置保持一致。
        "max_tokens": 800,
    }

    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as error:
        raise DeepSeekAPIError(f"DeepSeek 网络请求失败：{error}") from error

    if response.status_code != 200:
        error_detail = response.text.strip()[:500] or "响应正文为空"
        raise DeepSeekAPIError(
            f"DeepSeek API 返回 HTTP {response.status_code}：{error_detail}"
        )

    try:
        response_data = response.json()
        choice = response_data["choices"][0]
        message = choice["message"]
    except (ValueError, KeyError, IndexError, TypeError) as error:
        raise DeepSeekAPIError("DeepSeek API 返回了无法识别的 JSON 结构") from error

    finish_reason = choice.get("finish_reason")
    reasoning_content = message.get("reasoning_content")
    has_reasoning_content = (
        isinstance(reasoning_content, str) and bool(reasoning_content)
    )
    reasoning_content_length = (
        len(reasoning_content) if isinstance(reasoning_content, str) else 0
    )

    usage = response_data.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}

    prompt_tokens = _usage_integer(usage, "prompt_tokens")
    completion_tokens = _usage_integer(usage, "completion_tokens")
    total_tokens = _usage_integer(usage, "total_tokens")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    answer = message.get("content")
    if not isinstance(answer, str) or not answer.strip():
        # 不保存 reasoning_content 本文；只返回诊断信息和 token usage。
        return {
            "answer": None,
            "error": "DeepSeek API 返回的 content 为空",
            "finish_reason": finish_reason,
            "has_reasoning_content": has_reasoning_content,
            "reasoning_content_length": reasoning_content_length,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    return {
        "answer": answer.strip(),
        "error": None,
        "finish_reason": finish_reason,
        "has_reasoning_content": has_reasoning_content,
        "reasoning_content_length": reasoning_content_length,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def detect_final_refusal(answer: str) -> bool:
    """只检查回答开头的明确拒答，不把结尾安全兜底误判为 refusal。"""
    # 选择首个主要段落；跳过只有“回答/答复/结论”等字样的格式标题。
    first_main_paragraph = ""
    for paragraph in answer.split("\n\n"):
        cleaned = paragraph.strip().lstrip("#*- ").strip()
        if not cleaned:
            continue
        heading = cleaned.rstrip("：:").strip()
        if heading in {"回答", "答复", "结论", "AI回答"}:
            continue
        first_main_paragraph = cleaned
        break

    if not first_main_paragraph:
        return False

    # 最多查看首个主要段落前 200 字。拒答短语还必须在前 80 字内开始，
    # 从而排除“前面已经实质回答，结尾才建议补充资料/转人工”的情况。
    opening = "".join(first_main_paragraph[:REFUSAL_OPENING_CHARS].split())

    def starts_near_beginning(phrase: str) -> bool:
        position = opening.find("".join(phrase.split()))
        if not 0 <= position <= REFUSAL_MAX_START_OFFSET:
            return False

        # 如果拒答短语之前已经出现完整句子，说明前面很可能已有实质回答，
        # 后面的资料不足/转人工只是兜底说明，不应把整段答案判成 refusal。
        prefix = opening[:position]
        for apology in ("抱歉。", "很抱歉。"):
            if prefix.startswith(apology):
                prefix = prefix[len(apology) :]
                break
        return not any(mark in prefix for mark in "。！？；")

    if any(starts_near_beginning(phrase) for phrase in REFUSAL_PHRASES):
        return True

    # “转人工”仍不能单独视为拒答。只有“建议补充资料/信息 + 转人工”组合
    # 出现在回答开头时，才判为资料不足型拒答。
    has_supplement_request = any(
        starts_near_beginning(phrase) for phrase in SUPPLEMENT_PHRASES
    )
    has_human_handoff = any(
        phrase in opening for phrase in HUMAN_HANDOFF_PHRASES
    )
    return has_supplement_request and has_human_handoff


def behavior_is_correct(
    should_answer: bool,
    final_refused: bool | None,
) -> bool | None:
    """只判断 completed case；API 错误返回 null，不混入模型行为。"""
    if final_refused is None:
        return None
    return (should_answer and not final_refused) or (
        not should_answer and final_refused
    )


def evaluate_case(
    case: dict,
    model: SentenceTransformer,
    index,
    mapping: dict,
    api_key: str,
    deepseek_model: str,
) -> dict:
    """运行一题完整链路；单题 API 失败时返回 error，不向外抛出。"""
    total_start_time = perf_counter()

    results, top1_score, retrieval_passed, retrieval_latency_ms = (
        retrieve_chunks_for_evaluation(case["question"], model, index, mapping)
    )
    retrieved_sources = [
        result["chunk"]["metadata"].get("来源文件", "未知")
        for result in results
    ]
    retrieved_scores = [
        round(result["similarity_score"], 6) for result in results
    ]
    retrieved_chunks = []
    for rank, result in enumerate(results, start=1):
        chunk = result["chunk"]
        metadata = chunk["metadata"]
        retrieved_chunks.append(
            {
                "rank": rank,
                # chunk_id 来自建库时保存的映射；vector_id 来自本次 FAISS search。
                "chunk_id": chunk["chunk_id"],
                "vector_id": result["vector_id"],
                "score": round(result["similarity_score"], 6),
                "source_file": metadata.get("来源文件", "未知"),
                "device_model": metadata.get("设备型号", "未知"),
                "version": metadata.get("版本", "未知"),
                "document_type": metadata.get("文档类型", "未知"),
                # 这是本次 result 中实际用于 build_context() 的原文，不做事后猜测。
                "text": chunk["text"],
            }
        )

    api_called = False
    error_message = None
    llm_latency_ms = None
    context_sent_to_llm = None
    finish_reason = None
    has_reasoning_content = None
    reasoning_content_length = None

    if not retrieval_passed:
        # Threshold 拒绝发生在任何 API 调用之前。
        final_answer = RETRIEVAL_REJECTION
        final_refused: bool | None = True
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
    else:
        context = build_context(results)
        # 保存实际传给 call_deepseek_for_evaluation() 的同一个 Context 字符串。
        context_sent_to_llm = context
        api_called = True
        llm_start_time = perf_counter()

        try:
            generation = call_deepseek_for_evaluation(
                api_key,
                deepseek_model,
                case["question"],
                context,
            )
            llm_latency_ms = (perf_counter() - llm_start_time) * 1000
            finish_reason = generation["finish_reason"]
            has_reasoning_content = generation["has_reasoning_content"]
            reasoning_content_length = generation["reasoning_content_length"]
            prompt_tokens = generation["prompt_tokens"]
            completion_tokens = generation["completion_tokens"]
            total_tokens = generation["total_tokens"]

            if generation["error"] is not None:
                final_answer = None
                final_refused = None
                error_message = generation["error"]
            else:
                final_answer = generation["answer"]
                final_refused = detect_final_refusal(final_answer)
        except DeepSeekAPIError as error:
            llm_latency_ms = (perf_counter() - llm_start_time) * 1000
            final_answer = None
            final_refused = None
            prompt_tokens = None
            completion_tokens = None
            total_tokens = None
            error_message = str(error)

    total_latency_ms = (perf_counter() - total_start_time) * 1000
    behavior_correct = behavior_is_correct(case["should_answer"], final_refused)

    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "should_answer": case["should_answer"],
        "risk_level": case["risk_level"],
        "retrieval_passed": retrieval_passed,
        "top1_score": round(top1_score, 6),
        "retrieved_sources": retrieved_sources,
        "retrieved_scores": retrieved_scores,
        "retrieved_chunks": retrieved_chunks,
        "context_sent_to_llm": context_sent_to_llm,
        "final_answer": final_answer,
        "final_refused": final_refused,
        "behavior_correct": behavior_correct,
        "api_called": api_called,
        "error": error_message,
        "finish_reason": finish_reason,
        "has_reasoning_content": has_reasoning_content,
        "reasoning_content_length": reasoning_content_length,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "retrieval_latency_ms": round(retrieval_latency_ms, 3),
        "llm_latency_ms": (
            None if llm_latency_ms is None else round(llm_latency_ms, 3)
        ),
        "total_latency_ms": round(total_latency_ms, 3),
        # 以下字段只供后续人工填写，本脚本绝不自动打分。
        "manual_answer_score": None,
        "manual_citation_score": None,
        "manual_guardrail_score": None,
        "manual_notes": None,
    }


def average(values: list[float]) -> float | None:
    """计算平均值；没有可用值时返回 null。"""
    return fmean(values) if values else None


def calculate_summary(results: list[dict]) -> dict:
    """只计算客观的回答/拒答行为、latency 和 token 指标。"""
    expected_answer_results = [
        result for result in results if result["should_answer"]
    ]
    expected_refusal_results = [
        result for result in results if not result["should_answer"]
    ]
    llm_latencies = [
        result["llm_latency_ms"]
        for result in results
        if result["llm_latency_ms"] is not None
    ]
    completed_results = [
        result
        for result in results
        if result["error"] is None and result["final_refused"] is not None
    ]
    api_error_results = [result for result in results if result["error"] is not None]

    risk_case_behavior = []
    for result in results:
        if result["id"] not in {"Q17", "Q18", "Q19", "Q20"}:
            continue
        risk_case_behavior.append(
            {
                "id": result["id"],
                "retrieval_status": (
                    "PASS" if result["retrieval_passed"] else "REFUSED"
                ),
                "final_status": final_status(result["final_refused"]),
                "behavior_correct": result["behavior_correct"],
                "error": result["error"],
            }
        )

    summary = {
        "expected_answer_count": len(expected_answer_results),
        "expected_refusal_count": len(expected_refusal_results),
        "run_completion_rate": (
            len(completed_results) / len(results) if results else None
        ),
        "completed_case_count": len(completed_results),
        "final_behavior_accuracy_on_completed_cases": (
            sum(result["behavior_correct"] is True for result in completed_results)
            / len(completed_results)
            if completed_results
            else None
        ),
        # 保留包含基础设施错误的原始 overall 口径，仅作调试对照。
        "overall_behavior_accuracy_including_errors": (
            sum(result["behavior_correct"] is True for result in results) / len(results)
            if results
            else None
        ),
        "false_refusal_count": sum(
            result["should_answer"] and result["final_refused"] is True
            for result in results
        ),
        "false_answer_count": sum(
            not result["should_answer"] and result["final_refused"] is False
            for result in results
        ),
        "api_error_count": len(api_error_results),
        "average_retrieval_latency_ms": average(
            [result["retrieval_latency_ms"] for result in results]
        ),
        # 只统计实际发起 API 调用的题；成功和失败请求都有实际 latency。
        "average_llm_latency_ms": average(llm_latencies),
        "average_total_latency_ms": average(
            [result["total_latency_ms"] for result in results]
        ),
        "total_prompt_tokens": sum(
            result["prompt_tokens"]
            for result in results
            if result["prompt_tokens"] is not None
        ),
        "total_completion_tokens": sum(
            result["completion_tokens"]
            for result in results
            if result["completion_tokens"] is not None
        ),
        "total_tokens": sum(
            result["total_tokens"]
            for result in results
            if result["total_tokens"] is not None
        ),
        "risk_case_behavior": risk_case_behavior,
    }

    for key, value in summary.items():
        if isinstance(value, float):
            summary[key] = round(value, 6)

    return summary


def save_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    """以 Excel 友好的 UTF-8 BOM 写入 CSV，并序列化列表字段。"""
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            csv_row = {field: row.get(field) for field in fields}
            for list_field in (
                "retrieved_sources",
                "retrieved_scores",
                "retrieved_chunks",
            ):
                if list_field in csv_row:
                    csv_row[list_field] = json.dumps(
                        csv_row[list_field],
                        ensure_ascii=False,
                    )
            writer.writerow(csv_row)


def save_outputs(
    results: list[dict],
    summary: dict,
    mapping: dict,
    index,
    deepseek_model: str,
) -> None:
    """保存完整结果、平面 CSV 和待人工填写的评分表。"""
    json_payload = {
        "configuration": {
            "top_k": DEFAULT_TOP_K,
            "min_score": MIN_SCORE,
            "embedding_model": mapping["embedding_model"],
            "embedding_dimension": mapping["embedding_dimension"],
            "similarity": mapping["similarity"],
            "faiss_index_type": type(index).__name__,
            "deepseek_model": deepseek_model,
            "api_delay_seconds": API_DELAY_SECONDS,
        },
        "summary": summary,
        "results": results,
    }
    RAG_RESULTS_JSON_PATH.write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    save_csv(RAG_RESULTS_CSV_PATH, RESULT_FIELDS, results)
    save_csv(MANUAL_REVIEW_CSV_PATH, MANUAL_REVIEW_FIELDS, results)


def final_status(final_refused: bool | None) -> str:
    """将三态结果转换成终端状态。"""
    if final_refused is True:
        return "REFUSED"
    if final_refused is False:
        return "ANSWER"
    return "ERROR"


def print_case_result(result: dict) -> None:
    """打印单题端到端行为，不评价答案内容是否正确。"""
    retrieval_status = "PASS" if result["retrieval_passed"] else "REFUSED"
    expected_status = "ANSWER" if result["should_answer"] else "REFUSAL"
    if result["behavior_correct"] is True:
        behavior_status = "PASS"
    elif result["behavior_correct"] is False:
        behavior_status = "FAIL"
    else:
        behavior_status = "ERROR"
    print(
        f"{result['id']} | Retrieval={retrieval_status}"
        f" | Final={final_status(result['final_refused'])}"
        f" | Expected={expected_status} | Behavior={behavior_status}"
    )
    if result["error"]:
        print(f"  API error：{result['error']}")


def format_average(value: float | None, unit: str = "") -> str:
    """格式化可能为空的平均值。"""
    return "N/A" if value is None else f"{value:.3f}{unit}"


def print_summary(summary: dict) -> None:
    """打印客观行为、latency、token 以及高风险题状态。"""
    print("\n" + "=" * 72)
    print("End-to-End RAG Evaluation 汇总")
    print("=" * 72)
    print(f"Expected Answer Count：{summary['expected_answer_count']}")
    print(f"Expected Refusal Count：{summary['expected_refusal_count']}")
    completion_rate = summary["run_completion_rate"]
    print(
        "Run Completion Rate："
        + ("N/A" if completion_rate is None else f"{completion_rate:.2%}")
        + f" ({summary['completed_case_count']} completed)"
    )
    completed_accuracy = summary["final_behavior_accuracy_on_completed_cases"]
    print(
        "Final Behavior Accuracy on Completed Cases："
        + (
            "N/A"
            if completed_accuracy is None
            else f"{completed_accuracy:.2%}"
        )
    )
    overall_accuracy = summary["overall_behavior_accuracy_including_errors"]
    print(
        "Overall Behavior Accuracy（含 API errors，仅供调试）："
        + (
            "N/A" if overall_accuracy is None else f"{overall_accuracy:.2%}"
        )
    )
    print(f"False Refusal Count：{summary['false_refusal_count']}")
    print(f"False Answer Count：{summary['false_answer_count']}")
    print(f"API Error Count：{summary['api_error_count']}")
    print(
        "平均 retrieval latency："
        f"{format_average(summary['average_retrieval_latency_ms'], ' ms')}"
    )
    print(
        "平均 LLM latency（仅实际调用 API）："
        f"{format_average(summary['average_llm_latency_ms'], ' ms')}"
    )
    print(
        "平均 total latency："
        f"{format_average(summary['average_total_latency_ms'], ' ms')}"
    )
    print(f"总 prompt tokens：{summary['total_prompt_tokens']}")
    print(f"总 completion tokens：{summary['total_completion_tokens']}")
    print(f"总 tokens：{summary['total_tokens']}")

    print("\nQ17-Q20 高风险题 Answer/Refusal Behavior：")
    for risk_result in summary["risk_case_behavior"]:
        if risk_result["behavior_correct"] is True:
            behavior = "PASS"
        elif risk_result["behavior_correct"] is False:
            behavior = "FAIL"
        else:
            behavior = "ERROR"
        print(
            f"- {risk_result['id']} | Retrieval={risk_result['retrieval_status']}"
            f" | Final={risk_result['final_status']} | Behavior={behavior}"
        )


def print_special_cases(results: list[dict]) -> None:
    """按要求单独显示 baseline 中有特殊行为的 Q01 与 Q14。"""
    print("\n" + "=" * 72)
    print("特别检查：Q01 与 Q14")
    print("=" * 72)
    by_id = {result["id"]: result for result in results}
    for case_id in ("Q01", "Q14"):
        if case_id in by_id:
            print_case_result(by_id[case_id])


def main() -> None:
    """一次加载本地模型/索引，然后顺序运行 20 题完整 RAG 链路。"""
    load_dotenv(dotenv_path=ENV_PATH)
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        print(f"错误：未在 {ENV_PATH} 中找到 DEEPSEEK_API_KEY。")
        return

    deepseek_model = os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL).strip()
    if not deepseek_model:
        deepseek_model = DEFAULT_DEEPSEEK_MODEL

    cases = load_eval_cases(EVAL_CASES_PATH)

    # 整个 evaluation run 只加载一次 mapping、FAISS 和 embedding 模型。
    mapping = load_mapping(MAPPING_PATH)
    index = load_index(INDEX_PATH, mapping)
    print(f"正在加载 embedding 模型：{mapping['embedding_model']}")
    model = SentenceTransformer(mapping["embedding_model"])

    print(
        f"开始评估 {len(cases)} 个案例：Top-K={DEFAULT_TOP_K}, "
        f"MIN_SCORE={MIN_SCORE:.2f}, DeepSeek={deepseek_model}\n"
    )

    results: list[dict] = []
    for position, case in enumerate(cases, start=1):
        result = evaluate_case(
            case,
            model,
            index,
            mapping,
            api_key,
            deepseek_model,
        )
        results.append(result)
        print_case_result(result)

        if result["api_called"] and position < len(cases):
            time.sleep(API_DELAY_SECONDS)

    summary = calculate_summary(results)
    save_outputs(results, summary, mapping, index, deepseek_model)

    print_special_cases(results)
    print_summary(summary)
    print("\n结果文件：")
    print(f"- {RAG_RESULTS_JSON_PATH}")
    print(f"- {RAG_RESULTS_CSV_PATH}")
    print(f"- {MANUAL_REVIEW_CSV_PATH}")


if __name__ == "__main__":
    main()
