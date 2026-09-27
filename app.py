"""Phase 5C：企业知识助手 Streamlit Portfolio 界面。"""

import html
import logging
import os
import re

import streamlit as st
from sentence_transformers import SentenceTransformer

from query import (
    DEFAULT_TOP_K,
    INDEX_PATH,
    MAPPING_PATH,
    MIN_SCORE,
    load_index,
    load_mapping,
)
from rag import DeepSeekAPIError, RAGConfigurationError, run_rag


LOGGER = logging.getLogger(__name__)

EXAMPLE_QUESTIONS = (
    "E107 报警后应该如何处理？",
    "更换伺服电机后需要检查哪些参数？",
    "更换电机后方向反了，我能直接交换动力线吗？",
    "A 型设备的保修期是几年？",
)

HIGH_RISK_KEYWORDS = (
    "维修",
    "拆装",
    "报警",
    "参数修改",
    "参数",
    "校准",
    "恢复生产",
    "生产",
    "电机",
    "传感器",
)

SIMULATION_NOTICE = (
    "演示环境 · 当前知识库、设备参数与维修流程均为模拟数据，"
    "不可用于真实设备维修。"
)
SAFETY_NOTICE = (
    "涉及维修、拆装、参数修改或生产恢复，请遵循企业安全流程"
    "并由专业人员人工复核。"
)


def apply_page_styles() -> None:
    """注入少量 UI 样式；不影响任何 RAG 数据或判断。"""
    st.markdown(
        """
        <style>
        html {
            font-size: 17px;
        }

        body,
        p,
        li,
        div {
            font-size: 17px;
        }

        .stMainBlockContainer,
        [data-testid="stMainBlockContainer"] {
            max-width: 1100px;
            padding-top: 2.65rem;
            padding-bottom: 3.8rem;
        }

        h1 {
            font-size: 34px !important;
            letter-spacing: -0.025em;
            color: #111827;
        }

        h2 {
            font-size: 25px !important;
        }

        h3 {
            font-size: 20px !important;
        }

        .page-subtitle {
            margin: -0.55rem 0 1.35rem;
            color: #4b5563;
            font-size: 1.15rem;
        }

        .simulation-notice {
            margin: 0 0 1.75rem;
            padding: 0.75rem 0.95rem;
            border: 1px solid #e7dcc0;
            border-radius: 8px;
            background: #faf7ef;
            color: #655a42;
            font-size: 0.98rem;
            line-height: 1.5;
        }

        [data-testid="stSidebar"] h2 {
            margin-top: 0.4rem;
            font-size: 1.12rem !important;
        }

        .sidebar-status {
            margin-bottom: 0.8rem;
        }

        [data-testid="stSidebar"] .sidebar-label {
            margin-top: 0.8rem;
            color: #6b7280;
            font-size: 0.82rem;
            line-height: 1.25;
        }

        [data-testid="stSidebar"] .sidebar-value {
            margin-top: 0.16rem;
            color: #1f2937;
            font-size: 0.98rem;
            line-height: 1.4;
        }

        [data-testid="stForm"] {
            padding: 1.15rem;
            border: 1px solid #e5e7eb;
            border-radius: 11px;
        }

        [data-testid="stTextArea"] textarea {
            padding: 0.8rem 0.9rem;
            border-color: #d1d5db;
            border-radius: 8px;
            box-shadow: none;
            font-size: 1rem;
            line-height: 1.5;
        }

        div[data-testid="stButton"] > button {
            min-height: 3.05rem;
            padding: 0.68rem 0.9rem;
            border-color: #d1d5db;
            background: #ffffff;
            color: #374151;
            font-size: 0.96rem;
            font-weight: 450;
            line-height: 1.35;
        }

        div[data-testid="stButton"] > button:hover {
            border-color: #9ca3af;
            color: #111827;
        }

        div[data-testid="stFormSubmitButton"] > button {
            min-height: 3.05rem;
            padding: 0.72rem 1rem;
            border-color: #111827 !important;
            background: #111827 !important;
            color: #ffffff !important;
            font-size: 1rem;
            font-weight: 600;
        }

        div[data-testid="stFormSubmitButton"] > button:hover {
            border-color: #1f2937 !important;
            background: #1f2937 !important;
        }

        .status-badge {
            display: inline-block;
            margin-bottom: 0.95rem;
            padding: 0.36rem 0.72rem;
            border: 1px solid;
            border-radius: 999px;
            font-size: 0.93rem;
            font-weight: 600;
            line-height: 1.3;
        }

        .status-answer {
            border-color: #b8d8c5;
            background: #edf7f0;
            color: #326647;
        }

        .status-refusal {
            border-color: #e6d09d;
            background: #fbf5e6;
            color: #755f2d;
        }

        .source-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
            gap: 0.75rem;
            margin: 0.7rem 0 1.15rem;
        }

        .source-card {
            min-height: 106px;
            padding: 0.88rem 1rem;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            background: #ffffff;
        }

        .source-type {
            color: #111827;
            font-size: 0.98rem;
            font-weight: 650;
        }

        .source-file {
            margin-top: 0.26rem;
            color: #374151;
            font-size: 0.93rem;
            overflow-wrap: anywhere;
        }

        .source-version {
            margin-top: 0.36rem;
            color: #6b7280;
            font-size: 0.86rem;
        }

        .safety-footer {
            margin-top: 1.55rem;
            padding-top: 0.98rem;
            border-top: 1px solid #e5e7eb;
            color: #6b6250;
            font-size: 0.94rem;
        }

        [data-testid="stExpander"] {
            border-color: #e5e7eb;
            border-radius: 10px;
        }

        [data-testid="stExpander"] summary p {
            font-size: 1rem;
        }

        [data-testid="stMetric"] {
            padding: 0.25rem 0.1rem;
        }

        [data-testid="stMetricLabel"] p {
            font-size: 0.95rem !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 2.35rem !important;
            line-height: 1.15;
        }

        [data-testid="stMetricValue"] div {
            font-size: inherit !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="正在加载本地向量索引与 embedding 模型……")
def load_cached_resources() -> dict:
    """跨 Streamlit rerun 缓存只读的模型、FAISS 索引和 metadata。"""
    mapping = load_mapping(MAPPING_PATH)
    index = load_index(INDEX_PATH, mapping)
    model = SentenceTransformer(mapping["embedding_model"])
    return {
        "model": model,
        "index": index,
        "mapping": mapping,
    }


def fill_example(question: str) -> None:
    """示例按钮只填充输入框，不提交问题。"""
    st.session_state["question_input"] = question


def yes_or_no(value: bool) -> str:
    return "Yes" if value else "No"


def format_latency(value: float | None) -> str:
    if value is None:
        return "N/A"
    if value >= 1000:
        return f"{value / 1000:.2f} s"
    return f"{value:.0f} ms"


def format_tokens(value: int | None) -> str:
    return "N/A" if value is None else str(value)


def render_sidebar(mapping: dict) -> None:
    """展示冻结的系统配置，不展示任何敏感信息或 Prompt。"""
    deployment_env = os.getenv("DEPLOYMENT_ENV", "Local PoC")

    with st.sidebar:
        st.header("System Status")
        st.markdown(
            f"""
            <div class="sidebar-status">
                <div class="sidebar-label">Embedding</div>
                <div class="sidebar-value">Multilingual MiniLM · {mapping['embedding_dimension']}d</div>
                <div class="sidebar-label">Vector Search</div>
                <div class="sidebar-value">FAISS</div>
                <div class="sidebar-label">Top-K</div>
                <div class="sidebar-value">{DEFAULT_TOP_K}</div>
                <div class="sidebar-label">Threshold</div>
                <div class="sidebar-value">{MIN_SCORE:.2f}</div>
                <div class="sidebar-label">LLM</div>
                <div class="sidebar-value">DeepSeek</div>
                <div class="sidebar-label">Environment</div>
                <div class="sidebar-value">{html.escape(deployment_env)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.header("Demo Scope")
        st.markdown(
            "\n".join(
                [
                    "- A-1000",
                    "- 报警代码",
                    "- 维修手册",
                    "- 历史故障案例",
                ]
            )
        )


def render_answer(result: dict) -> None:
    st.subheader("Answer")
    if result["final_refused"]:
        badge_text = "知识库依据不足 · 已拒答"
        badge_class = "status-refusal"
    else:
        badge_text = "已基于知识库生成回答"
        badge_class = "status-answer"

    with st.container(border=True):
        st.markdown(
            f'<span class="status-badge {badge_class}">{badge_text}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(result["final_answer"])


def render_retrieval_evidence(result: dict) -> None:
    with st.expander("查看检索证据"):
        st.markdown("#### Retrieved Evidence")
        for chunk in result["retrieved_chunks"]:
            with st.container(border=True):
                st.markdown(f"**Top {chunk['rank']}**")
                st.markdown(f"**{chunk['source_file']}**")
                st.caption(
                    f"{chunk['document_type']} · v{chunk['version']} · "
                    f"Similarity {chunk['score']:.4f}"
                )
                st.text(chunk["text"])
            if chunk["rank"] < len(result["retrieved_chunks"]):
                st.write("")


def citation_ranks(answer: str) -> list[int]:
    """只识别回答中明确出现的 [资料1]～[资料3]。"""
    return sorted({int(number) for number in re.findall(r"\[资料([1-3])\]", answer)})


def render_source_cards(chunks: list[dict], show_reference: bool) -> None:
    """以紧凑卡片展示来源，不改变 chunk 内容或顺序。"""
    cards = []
    for chunk in chunks:
        reference = f"资料{chunk['rank']} · " if show_reference else ""
        cards.append(
            "".join(
                [
                    '<div class="source-card">',
                    '<div class="source-type">',
                    html.escape(f"{reference}{chunk['document_type']}"),
                    "</div>",
                    '<div class="source-file">',
                    html.escape(str(chunk["source_file"])),
                    "</div>",
                    '<div class="source-version">',
                    html.escape(f"Version {chunk['version']}"),
                    "</div>",
                    "</div>",
                ]
            )
        )

    st.markdown(
        f'<div class="source-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def render_sources(result: dict) -> None:
    st.subheader("Sources")

    if result["final_refused"]:
        st.markdown("#### Inspected Sources")
        render_source_cards(result["retrieved_chunks"], show_reference=False)
        return

    st.markdown("#### Cited Sources")
    cited_ranks = citation_ranks(result["final_answer"])
    chunks_by_rank = {
        chunk["rank"]: chunk for chunk in result["retrieved_chunks"]
    }
    cited_chunks = [
        chunks_by_rank[rank] for rank in cited_ranks if rank in chunks_by_rank
    ]

    if not cited_chunks:
        st.caption("最终回答中未检测到明确的 [资料N] 引用。")
        return

    render_source_cards(cited_chunks, show_reference=True)


def render_technical_details(result: dict, mapping: dict) -> None:
    with st.expander("Technical Details"):
        top1_score = result["top1_score"]
        primary_metrics = st.columns(4)
        primary_metrics[0].metric(
            "Top-1 Similarity",
            "N/A" if top1_score is None else f"{top1_score:.4f}",
        )
        primary_metrics[1].metric(
            "Retrieval",
            format_latency(result["retrieval_latency_ms"]),
        )
        primary_metrics[2].metric(
            "LLM",
            format_latency(result["llm_latency_ms"]),
        )
        primary_metrics[3].metric(
            "Total",
            format_latency(result["total_latency_ms"]),
        )

        st.divider()
        secondary_metrics = st.columns(5)
        secondary_metrics[0].metric(
            "Retrieval Passed",
            yes_or_no(result["retrieval_passed"]),
        )
        secondary_metrics[1].metric(
            "API Called",
            yes_or_no(result["api_called"]),
        )
        secondary_metrics[2].metric(
            "Prompt Tokens",
            format_tokens(result["prompt_tokens"]),
        )
        secondary_metrics[3].metric(
            "Completion Tokens",
            format_tokens(result["completion_tokens"]),
        )
        secondary_metrics[4].metric(
            "Total Tokens",
            format_tokens(result["total_tokens"]),
        )

        st.caption(f"Embedding Model: `{mapping['embedding_model']}`")


def render_question_form() -> tuple[bool, str]:
    st.subheader("Question Input")
    st.caption("示例问题（点击后只填充输入框，不会自动发送）")

    first_row = st.columns(2)
    second_row = st.columns(2)
    example_columns = [*first_row, *second_row]
    for index, (column, question) in enumerate(
        zip(example_columns, EXAMPLE_QUESTIONS),
        start=1,
    ):
        with column:
            st.button(
                question,
                key=f"example_{index}",
                on_click=fill_example,
                args=(question,),
                width="stretch",
            )

    with st.form("question_form", clear_on_submit=False, enter_to_submit=False):
        question = st.text_area(
            "请输入关于 A 型设备的问题",
            key="question_input",
            height=110,
            placeholder="例如：E107 报警后应该如何处理？",
        )
        submitted = st.form_submit_button(
            "提交问题",
            type="primary",
            width="stretch",
        )

    return submitted, question


def main() -> None:
    st.set_page_config(
        page_title="企业知识助手 PoC",
        page_icon=None,
        layout="wide",
    )
    apply_page_styles()

    st.title("企业知识助手 PoC")
    st.markdown(
        '<div class="page-subtitle">A 型工业设备售后知识检索与问答</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="simulation-notice">{SIMULATION_NOTICE}</div>',
        unsafe_allow_html=True,
    )

    if not MAPPING_PATH.exists():
        st.error("知识库 metadata 文件不存在，请先完成文档建库。")
        return
    if not INDEX_PATH.exists():
        st.error("FAISS 索引文件不存在，请先完成文档建库。")
        return

    try:
        resources = load_cached_resources()
    except Exception:
        LOGGER.exception("加载本地 RAG 资源失败")
        st.error("本地知识库资源加载失败，请查看终端日志。")
        return

    render_sidebar(resources["mapping"])

    if "question_input" not in st.session_state:
        st.session_state["question_input"] = ""

    submitted, question = render_question_form()
    if not submitted:
        return

    question = question.strip()
    if not question:
        st.warning("请输入问题后再提交。")
        return

    try:
        with st.spinner("正在检索企业知识库并生成回答……"):
            result = run_rag(
                question,
                model=resources["model"],
                index=resources["index"],
                mapping=resources["mapping"],
            )
    except RAGConfigurationError:
        LOGGER.exception("DeepSeek API 配置缺失")
        st.error("未配置 DeepSeek API Key，请检查项目根目录的 .env 文件。")
        return
    except DeepSeekAPIError:
        LOGGER.exception("DeepSeek API 请求失败")
        st.error("问答服务暂时不可用，请稍后重试并查看终端日志。")
        return
    except FileNotFoundError:
        LOGGER.exception("本地索引或 metadata 文件缺失")
        st.error("本地知识库文件缺失，请先完成文档建库。")
        return
    except (ValueError, RuntimeError):
        LOGGER.exception("RAG 流程执行失败")
        st.error("知识检索执行失败，请查看终端日志。")
        return
    except Exception:
        LOGGER.exception("Streamlit Demo 出现未预期错误")
        st.error("系统出现未预期错误，请查看终端日志。")
        return

    st.divider()
    render_answer(result)
    render_sources(result)
    render_retrieval_evidence(result)
    render_technical_details(result, resources["mapping"])

    if any(keyword in question for keyword in HIGH_RISK_KEYWORDS):
        st.markdown(
            f'<div class="safety-footer">{SAFETY_NOTICE}</div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
