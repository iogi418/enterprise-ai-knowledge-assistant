"""Phase 1：读取文档、切分文本、生成向量并建立 FAISS 索引。"""

import json
import os
import re
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# 所有路径都以当前脚本所在目录为基准，因此可以从任意工作目录运行本脚本。
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "vector_store"
INDEX_PATH = OUTPUT_DIR / "faiss.index"
MAPPING_PATH = OUTPUT_DIR / "chunks_metadata.json"

# 该模型可处理中文和英文。第一次运行时 sentence-transformers 会下载模型文件，
# 之后会从本机缓存加载；这里没有调用任何大模型 API。
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def read_document(file_path: Path) -> tuple[dict[str, str], str]:
    """读取单个文档，并从开头到分隔线之间提取简单 metadata。"""
    # utf-8-sig 同时兼容普通 UTF-8 和带 BOM 的 UTF-8 文本。
    raw_text = file_path.read_text(encoding="utf-8-sig")
    lines = raw_text.splitlines()

    try:
        separator_index = next(
            index for index, line in enumerate(lines) if line.strip() == "---"
        )
    except StopIteration as error:
        raise ValueError(f"{file_path.name} 缺少 metadata 分隔线 '---'") from error

    metadata: dict[str, str] = {"来源文件": file_path.name}
    for line in lines[:separator_index]:
        line = line.strip()
        if not line:
            continue

        # 同时接受中文冒号和英文冒号，例如“设备型号：A-1000”。
        if "：" in line:
            key, value = line.split("：", maxsplit=1)
        elif ":" in line:
            key, value = line.split(":", maxsplit=1)
        else:
            raise ValueError(f"{file_path.name} 中的 metadata 格式无法识别：{line}")

        metadata[key.strip()] = value.strip()

    content = "\n".join(lines[separator_index + 1 :]).strip()
    if not content:
        raise ValueError(f"{file_path.name} 没有正文内容")

    return metadata, content


def split_text_by_characters(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    base_offset: int = 0,
) -> list[dict[str, int | str]]:
    """字符窗口 fallback：切分过长 section，并尽量在自然标点处结束。

    chunk_size 是每个片段的最大字符数，chunk_overlap 是相邻片段重复的字符数。
    base_offset 用于把 section 内的位置换算为整篇正文中的字符位置。
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须大于等于 0，并且小于 chunk_size")

    chunks: list[dict[str, int | str]] = []
    start = 0
    text_length = len(text)
    preferred_boundaries = ("。", "！", "？", "；", "\n")

    while start < text_length:
        end = min(start + chunk_size, text_length)

        # 如果还没到文末，就在窗口后半段寻找最近的自然边界，避免过早切断句子。
        if end < text_length:
            search_from = start + chunk_size // 2
            boundary_positions = [
                text.rfind(boundary, search_from, end)
                for boundary in preferred_boundaries
            ]
            best_boundary = max(boundary_positions)
            if best_boundary != -1:
                end = best_boundary + 1

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "char_start": base_offset + start,
                    "char_end": base_offset + end,
                }
            )

        if end >= text_length:
            break

        # 下一段向前回退一小段，形成 overlap；max 用于确保游标一定向前移动。
        start = max(end - chunk_overlap, start + 1)

    return chunks


def find_paragraphs(text: str) -> list[dict[str, int | str]]:
    """按空行找出候选段落，同时保留每段在正文中的字符位置。"""
    # 只把“两个换行之间允许有空格”的位置视为空行；单个换行仍属于同一段。
    paragraph_pattern = re.compile(
        r"\S(?:.*?\S)?(?=\n[ \t]*\n|\Z)",
        flags=re.DOTALL,
    )
    return [
        {
            "text": match.group(0),
            "char_start": match.start(),
            "char_end": match.end(),
        }
        for match in paragraph_pattern.finditer(text)
    ]


def looks_like_section_start(paragraph: str) -> bool:
    """根据常见标题、编号和短标题格式判断一个段落是否开启新 section。"""
    first_line = paragraph.splitlines()[0].strip()

    section_start_patterns = (
        # Markdown 标题，例如“## 传感器校准”。
        r"^#{1,6}\s+\S+",
        # 中文章节或编号，例如“第三章”“二、”“（三）”。
        r"^第[一二三四五六七八九十百零\d]+[章节篇部分]",
        r"^[一二三四五六七八九十百零]+[、.]",
        r"^[（(][一二三四五六七八九十百零\d]+[）)]",
        # 数字层级编号，例如“1.”“2.1”“3、”。
        r"^\d+(?:\.\d+)*[.、)]\s*\S+",
        # 通用案例编号，例如“案例一”“案例 2”。
        r"^案例\s*[一二三四五六七八九十百零\d]+",
        # 字母加数字的条目编号，例如 E107、ALM-203；不限定具体报警代码。
        r"^[A-Za-z]{1,8}[-_]?\d{2,8}\s*[：:]",
        # 简短的“标题：正文”结构，标题部分不能已经是一个完整句子。
        r"^[^：:。！？；\n]{2,40}[：:]",
    )
    return any(re.match(pattern, first_line) for pattern in section_start_patterns)


def infer_section_title(section_text: str) -> str:
    """从 section 首行提取便于查看的标题，不参与检索计算。"""
    first_line = section_text.splitlines()[0].strip().lstrip("#").strip()

    # 对“标题：正文”格式只取冒号前的标题。
    title_match = re.match(r"^([^：:。！？；]{1,40})[：:]", first_line)
    if title_match:
        return title_match.group(1).strip()

    # 没有显式标题时，用首句开头作为可读标签，而不是依赖具体文件内容。
    first_sentence = re.split(r"[。！？；]", first_line, maxsplit=1)[0].strip()
    if len(first_sentence) > 30:
        return first_sentence[:30] + "……"
    return first_sentence or "未命名章节"


def split_into_sections(text: str) -> list[dict[str, int | str]]:
    """先按空行取段落，再用结构信号把段落组织成自然语义 section。"""
    paragraphs = find_paragraphs(text)
    if not paragraphs:
        return []

    sections: list[dict[str, int | str]] = []
    current_start = int(paragraphs[0]["char_start"])
    current_end = int(paragraphs[0]["char_end"])

    for paragraph in paragraphs[1:]:
        paragraph_text = str(paragraph["text"])

        if looks_like_section_start(paragraph_text):
            section_text = text[current_start:current_end].strip()
            sections.append(
                {
                    "text": section_text,
                    "title": infer_section_title(section_text),
                    "char_start": current_start,
                    "char_end": current_end,
                }
            )
            current_start = int(paragraph["char_start"])

        # 没有标题信号的段落视为当前 section 的延续，例如说明或注意事项。
        current_end = int(paragraph["char_end"])

    final_text = text[current_start:current_end].strip()
    sections.append(
        {
            "text": final_text,
            "title": infer_section_title(final_text),
            "char_start": current_start,
            "char_end": current_end,
        }
    )
    return sections


def split_text(
    text: str,
    chunk_size: int = 450,
    chunk_overlap: int = 50,
) -> list[dict[str, int | str]]:
    """优先按自然 section 切分；section 过长时再使用字符窗口 fallback。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须大于等于 0，并且小于 chunk_size")

    chunks: list[dict[str, int | str]] = []
    for section in split_into_sections(text):
        section_text = str(section["text"])
        section_start = int(section["char_start"])

        if len(section_text) <= chunk_size:
            section_chunks = [
                {
                    "text": section_text,
                    "char_start": section["char_start"],
                    "char_end": section["char_end"],
                }
            ]
            split_method = "natural_section"
        else:
            section_chunks = split_text_by_characters(
                section_text,
                chunk_size,
                chunk_overlap,
                base_offset=section_start,
            )
            split_method = "character_fallback"

        for section_part, chunk in enumerate(section_chunks):
            chunk["section_title"] = section["title"]
            chunk["section_part"] = section_part
            chunk["section_part_count"] = len(section_chunks)
            chunk["split_method"] = split_method
            chunks.append(chunk)

    return chunks


def load_and_chunk_documents(
    data_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> list[dict]:
    """读取 data 下所有 txt 文件，并生成带来源信息的 chunk 记录。"""
    text_files = sorted(data_dir.rglob("*.txt"))
    if not text_files:
        raise FileNotFoundError(f"在 {data_dir} 下没有找到 txt 文件")

    records: list[dict] = []
    for file_path in text_files:
        metadata, content = read_document(file_path)
        document_chunks = split_text(content, chunk_size, chunk_overlap)

        for chunk_index, chunk in enumerate(document_chunks):
            # chunk_id 也是 FAISS 中的向量顺序，便于把搜索结果映射回原文。
            records.append(
                {
                    "chunk_id": len(records),
                    "text": chunk["text"],
                    "metadata": {
                        **metadata,
                        "文档内片段序号": chunk_index,
                        "章节标题": chunk["section_title"],
                        "章节内片段序号": chunk["section_part"],
                        "章节片段总数": chunk["section_part_count"],
                        "切分方式": chunk["split_method"],
                        "正文起始字符": chunk["char_start"],
                        "正文结束字符": chunk["char_end"],
                    },
                }
            )

        print(f"已读取 {file_path.name}，生成 {len(document_chunks)} 个 chunks")

    return records


def build_and_save_index(records: list[dict], model_name: str) -> None:
    """把 chunk 转成 embedding，建立 FAISS 索引并保存索引与映射。"""
    texts = [record["text"] for record in records]

    print(f"正在加载 embedding 模型：{model_name}")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    # FAISS 需要连续存放的 float32 数组。
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    # 对已经归一化的向量使用内积（Inner Product），结果等价于余弦相似度。
    embedding_dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(embedding_dimension)
    index.add(embeddings)

    if index.ntotal != len(records):
        raise RuntimeError("FAISS 中的向量数量与 chunk 数量不一致")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))

    # FAISS 只保存向量索引，不保存原文，所以必须另外保存顺序一致的映射文件。
    mapping = {
        "embedding_model": model_name,
        "embedding_dimension": embedding_dimension,
        "similarity": "inner_product_on_normalized_vectors",
        "chunk_count": len(records),
        "chunks": records,
    }
    MAPPING_PATH.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"FAISS 索引已保存：{INDEX_PATH}")
    print(f"chunk/metadata 映射已保存：{MAPPING_PATH}")
    print(f"共保存 {index.ntotal} 个向量，每个向量 {embedding_dimension} 维")


def main() -> None:
    """按 Phase 1 的固定顺序执行完整 ingestion 流程。"""
    load_dotenv()

    # 可以在 .env 中覆盖这些值；没有 .env 时直接使用便于学习的默认值。
    model_name = os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_MODEL_NAME)
    chunk_size = int(os.getenv("CHUNK_SIZE", "450"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "50"))

    print(f"数据目录：{DATA_DIR}")
    print(
        "切分策略：优先自然 section；过长时使用字符窗口 "
        f"(chunk_size={chunk_size}, chunk_overlap={chunk_overlap})"
    )

    records = load_and_chunk_documents(DATA_DIR, chunk_size, chunk_overlap)
    print(f"全部文档共生成 {len(records)} 个 chunks")
    build_and_save_index(records, model_name)


if __name__ == "__main__":
    main()
