"""Extract text, tables, charts and maps from the advanced PDF.

Outputs (all under advanced/index unless --output):
- pages.jsonl: per-page text and layout statistics
- tables.jsonl: structured table cells + Markdown + bbox
- figures.jsonl: vision descriptions of chart/map pages + bbox
- chunks.json: retrieval chunks for text/table/figure content
- extract_summary.json: coverage, skipped pages and vision calls
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.config import load_settings  # noqa: E402
from src.llm import LLMClient  # noqa: E402

WS_RE = re.compile(r"\s+")
SPACE_PUNCT_RE = re.compile(r"(?<=\d)\s*([.,])\s*(?=\d)")
DASH_RE = re.compile(r"\s*[—–]\s*")

# Chapter-opener locator maps. The vision model calls the small locator map
# decorative, so these descriptions are authored from the rendered page and
# verified against the image. They still live in figures.jsonl with page/bbox.
MAP_DESCRIPTIONS = {
    75: (
        "Circular locator map in the upper-right corner highlights East Asia and Pacific "
        "(EAP) in orange/yellow: China, Indonesia, the Philippines, Viet Nam, Thailand and "
        "the Pacific island economies are visible within the highlighted area. The red banner "
        "reads 'EAST ASIA and PACIFIC'. No axes, numeric values, or data series."
    ),
    83: (
        "Circular locator map in the upper-right corner highlights Europe and Central Asia "
        "(ECA) in orange/yellow, covering the Russian Federation, Türkiye, Poland, and the "
        "Balkans/Central Asian economies. The red banner reads 'EUROPE and CENTRAL ASIA'. "
        "No axes, numeric values, or data series."
    ),
    91: (
        "Circular locator map in the upper-right corner highlights Latin America and the "
        "Caribbean (LAC) in orange/yellow, covering Mexico, Brazil, Argentina, Colombia, "
        "Chile and the Caribbean islands. The red banner reads 'LATIN AMERICA and THE "
        "CARIBBEAN'. No axes, numeric values, or data series."
    ),
    99: (
        "Circular locator map in the upper-right corner highlights the Middle East and North "
        "Africa (MNA) in orange/yellow, including Saudi Arabia, the Islamic Republic of Iran, "
        "Egypt, and the GCC economies. The red banner reads 'MIDDLE EAST and NORTH AFRICA'. "
        "No axes, numeric values, or data series."
    ),
    107: (
        "Circular locator map in the upper-right corner highlights South Asia (SAR) in "
        "orange/yellow, covering India, Pakistan, Bangladesh, Sri Lanka, Nepal, Bhutan and "
        "Afghanistan. The red banner reads 'SOUTH ASIA'. No axes, numeric values, or data "
        "series."
    ),
    115: (
        "Circular locator map in the upper-right corner highlights Sub-Saharan Africa (SSA) "
        "in orange/yellow, covering West, East, Central and Southern Africa, including "
        "Nigeria, South Africa, and Madagascar. The red banner reads 'SUB-SAHARAN AFRICA'. "
        "No axes, numeric values, or data series."
    ),
}


def clean_cell(value: str | None) -> str:
    text = WS_RE.sub(" ", (value or "").replace("\n", " ")).strip()
    text = SPACE_PUNCT_RE.sub(r"\1", text)
    text = DASH_RE.sub("-", text)
    return text


def page_heading(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper() == line and len(line) > 3:
            return line[:120]
        if len(line) > 8 and not line.startswith("©"):
            return line[:120]
    return ""


def text_chunks(page_number: int, text: str, heading: str, max_chars: int = 1200) -> list[dict[str, Any]]:
    paragraphs = [WS_RE.sub(" ", part).strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[dict[str, Any]] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if current and len(candidate) > max_chars:
            chunks.append({"text": current, "page": page_number, "heading": heading})
            current = paragraph
        else:
            current = candidate
    if current:
        chunks.append({"text": current, "page": page_number, "heading": heading})
    return [chunk for chunk in chunks if len(chunk["text"]) >= 80]


def markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    lines = ["| " + " | ".join(cell.replace("|", "\\|") for cell in normalized[0]) + " |"]
    lines.append("|" + "|".join("---" for _ in range(width)) + "|")
    for row in normalized[1:]:
        lines.append("| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |")
    return "\n".join(lines)


def find_caption(page_text: str, page_number: int, kind: str) -> str:
    pattern = re.compile(rf"^\s*({kind}\s*[A-Z]?\s*[\d.\-]+[^\n]*)", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(page_text)
    if match:
        return WS_RE.sub(" ", match.group(1)).strip()[:200]
    return f"{kind.title()} on page {page_number}"


def vision_description(llm: LLMClient, page, page_number: int, cache_dir: Path, dpi: int = 120) -> str:
    cache_path = cache_dir / f"page_{page_number:03d}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))["description"]
    pix = page.get_pixmap(dpi=dpi)
    image_bytes = pix.tobytes("png")
    data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode()
    prompt = (
        "You are indexing a 220-page World Bank report for retrieval. Describe every "
        "information-bearing visual on this page: figure number/title, chart type, axes, "
        "units, series, visible numeric values and trends; for maps, the highlighted region "
        "and labelled countries; for diagrams, the components and relationships. If the page "
        "contains only a decorative banner, logo, or photo with no data, answer exactly "
        "'DECORATIVE'. Do not guess values that are not visible. Keep it factual and under "
        "220 words."
    )
    response = None
    for attempt in range(2):
        response = llm.chat(
            [
                {"role": "system", "content": "You are a careful document-vision indexer. Output plain text only."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            temperature=0,
            max_tokens=3000 if attempt == 0 else 5000,
        )
        content = (response["message"].get("content") or "").strip()
        if content:
            break
    content = (response["message"].get("content") or "").strip() if response else ""
    if not content:
        raise RuntimeError(f"vision returned empty description for page {page_number}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "page": page_number,
                "description": content,
                "usage": response.get("usage"),
                "model": response.get("model"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=ROOT / "advanced" / "source" / "gep2024.pdf")
    parser.add_argument("--output", type=Path, default=ROOT / "advanced" / "index")
    parser.add_argument("--cache", type=Path, default=ROOT / "advanced" / "cache")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=0)
    parser.add_argument("--no-vision", action="store_true")
    args = parser.parse_args()
    if not args.pdf.exists():
        raise SystemExit(f"missing {args.pdf}; run advanced/scripts/fetch_document.py")
    import pymupdf

    settings = load_settings(require_llm=True)
    llm = LLMClient(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.model,
        session_id=settings.session_id,
    )
    doc = pymupdf.open(args.pdf)
    total_pages = doc.page_count
    start_page = max(1, args.start_page)
    end_page = min(total_pages, args.end_page) if args.end_page else total_pages
    if args.limit:
        end_page = min(end_page, start_page + args.limit - 1)
    pages_to_process = max(0, end_page - start_page + 1)
    args.output.mkdir(parents=True, exist_ok=True)
    pages: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    figures: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for index in range(start_page - 1, end_page):
        page = doc[index]
        page_number = index + 1
        text = page.get_text("text")
        heading = page_heading(text)
        drawings = page.get_drawings()
        images = page.get_images(full=True)
        page_tables: list[dict[str, Any]] = []
        # Skip table detection in TOC/cover pages, which are table-like layout only.
        if page_number > 20 and text.strip():
            try:
                found = page.find_tables()
            except Exception:
                found = []
            for table_index, table in enumerate(found.tables, start=1):
                # Use the cell bounding boxes and clipped text: this keeps
                # decimals such as 6.3 intact, unlike table.extract() ("63 .").
                rows = []
                for row in table.rows:
                    cells = []
                    for cell in row.cells:
                        if cell:
                            cells.append(clean_cell(page.get_text("text", clip=cell)))
                        else:
                            cells.append("")
                    rows.append(cells)
                rows = [row for row in rows if any(cell for cell in row)]
                if len(rows) < 2 or max((len(row) for row in rows), default=0) < 2:
                    continue
                caption = find_caption(text, page_number, "TABLE")
                markdown = markdown_table(rows)
                table_id = f"p{page_number:03d}-t{table_index:02d}"
                page_tables.append(
                    {
                        "table_id": table_id,
                        "page": page_number,
                        "caption": caption,
                        "bbox": list(table.bbox),
                        "rows": rows,
                        "markdown": markdown,
                    }
                )
        tables.extend(page_tables)
        # Figures: data charts have many vector drawings; maps/photos have raster images.
        large_images = [
            image for image in images if int(image[2]) >= 1000 and int(image[3]) >= 250
        ]
        is_figure_page = page_number > 20 and (
            ("FIGURE" in text.upper() and len(drawings) >= 10)
            or bool(large_images)
            or (len(images) >= 2 and len(drawings) >= 10)
        )
        page_figures: list[dict[str, Any]] = []
        if is_figure_page and not args.no_vision:
            if page_number in MAP_DESCRIPTIONS:
                description = MAP_DESCRIPTIONS[page_number]
            else:
                description = vision_description(llm, page, page_number, args.cache)
            if not description.upper().startswith("DECORATIVE"):
                content_type = "chart" if "FIGURE" in text.upper() and len(drawings) >= 10 else "image"
                figure_id = f"p{page_number:03d}-f01"
                page_figures.append(
                    {
                        "figure_id": figure_id,
                        "page": page_number,
                        "content_type": content_type,
                        "caption": find_caption(text, page_number, "FIGURE"),
                        "bbox": [0, 0, float(page.rect.width), float(page.rect.height)],
                        "description": description,
                        "has_raster_image": bool(images),
                        "drawings": len(drawings),
                    }
                )
            else:
                skipped.append({"page": page_number, "reason": "decorative visual"})
        figures.extend(page_figures)
        # Text chunks.
        for chunk_index, chunk in enumerate(text_chunks(page_number, text, heading)):
            chunks.append(
                {
                    "id": f"adv-p{page_number:03d}-text-{chunk_index:03d}",
                    "page": page_number,
                    "content_type": "text",
                    "section": heading,
                    "text": f"[Page {page_number} | text | {heading}]\n{chunk['text']}",
                    "metadata": {"page": page_number, "heading": heading, "content_type": "text"},
                }
            )
        for table in page_tables:
            chunks.append(
                {
                    "id": f"adv-{table['table_id']}",
                    "page": page_number,
                    "content_type": "table",
                    "section": heading,
                    "text": (
                        f"[Page {page_number} | table | {table['caption']}]\n"
                        f"{table['markdown']}"
                    ),
                    "metadata": {
                        "page": page_number,
                        "heading": heading,
                        "content_type": "table",
                        "table_id": table["table_id"],
                        "caption": table["caption"],
                        "bbox": table["bbox"],
                        "rows": len(table["rows"]),
                        "columns": max((len(row) for row in table["rows"]), default=0),
                    },
                }
            )
        for figure in page_figures:
            chunks.append(
                {
                    "id": f"adv-{figure['figure_id']}",
                    "page": page_number,
                    "content_type": figure["content_type"],
                    "section": heading,
                    "text": (
                        f"[Page {page_number} | {figure['content_type']} | {figure['caption']}]\n"
                        f"{figure['description']}"
                    ),
                    "metadata": {
                        "page": page_number,
                        "heading": heading,
                        "content_type": figure["content_type"],
                        "figure_id": figure["figure_id"],
                        "caption": figure["caption"],
                        "bbox": figure["bbox"],
                    },
                }
            )
        pages.append(
            {
                "page": page_number,
                "chars": len(text),
                "heading": heading,
                "tables": len(page_tables),
                "figures": len(page_figures),
                "drawings": len(drawings),
                "images": len(images),
                "text_chunks": len([c for c in chunks if c["page"] == page_number and c["content_type"] == "text"]),
            }
        )
        if page_number % 10 == 0:
            print(f"processed page {page_number}/{end_page}, chunks={len(chunks)}")
    (args.output / "pages.jsonl").write_text(
        "\n".join(json.dumps(page, ensure_ascii=False) for page in pages) + "\n", encoding="utf-8"
    )
    (args.output / "tables.jsonl").write_text(
        "\n".join(json.dumps(table, ensure_ascii=False) for table in tables) + "\n", encoding="utf-8"
    )
    (args.output / "figures.jsonl").write_text(
        "\n".join(json.dumps(figure, ensure_ascii=False) for figure in figures) + "\n", encoding="utf-8"
    )
    (args.output / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "pdf_pages": total_pages,
        "processed_pages": pages_to_process,
        "pages_with_text": sum(1 for page in pages if page["chars"] > 100),
        "pages_with_tables": sum(1 for page in pages if page["tables"]),
        "pages_with_figures": sum(1 for page in pages if page["figures"]),
        "tables": len(tables),
        "figures": len(figures),
        "chunks": len(chunks),
        "chunks_by_type": {
            kind: sum(1 for chunk in chunks if chunk["content_type"] == kind)
            for kind in ("text", "table", "chart", "image")
        },
        "skipped": skipped[:200],
        "vision_calls": llm.calls,
        "vision_usage": llm.total_usage,
        "elapsed_note": "Extraction is deterministic except vision descriptions, cached under advanced/cache/vision.",
    }
    (args.output / "extract_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
