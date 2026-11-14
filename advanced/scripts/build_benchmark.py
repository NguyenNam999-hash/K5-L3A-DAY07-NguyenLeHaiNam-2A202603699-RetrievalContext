"""Author and validate the advanced benchmark (18 dev + 30 test questions)."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHUNKS = ROOT / "advanced" / "index" / "chunks.json"
OUTPUT = ROOT / "advanced" / "benchmark" / "questions.json"


def gold(page: int, content_type: str, evidence: str, **extra):
    return {"page": page, "content_type": content_type, "evidence": evidence, **extra}


def q(qid, split, group, question, answer, gold_items, answerable=True, **extra):
    item = {
        "id": qid,
        "split": split,
        "group": group,
        "question": question,
        "answerable": answerable,
    }
    if answerable:
        item["expected_answer"] = answer
        item["gold"] = gold_items
    else:
        item["expected_behavior"] = "search_and_abstain"
        item["absence_reason"] = answer
    item.update(extra)
    return item


DEV = [
    q("ad-t01", "dev", "table",
      "According to Table 1.1, what is the real GDP growth forecast for the United States in 2024?",
      "2.5 percent.",
      [gold(26, "table", "| United States | 5.8 | 1.9 | 2.5 | 2.5 | 1.8 | 1.8 |  | 0.9 | 0.1 |",
            table_id="p026-t01", row="United States", column="2024f")]),
    q("ad-t02", "dev", "table",
      "In Table 2.1.1, what is the 2024 forecast for private consumption growth in East Asia and Pacific?",
      "6.2 percent.",
      [gold(81, "table", "| Private consumption | 9.6 | 1.9 | 9.0 | 6.2 | 5.5 | 5.3 |  | 0.3 | 0.1 |",
            table_id="p081-t01", row="Private consumption", column="2024f")]),
    q("ad-t03", "dev", "table",
      "According to Table B1.2.1, what is Niger's real GDP growth forecast for 2024?",
      "9.1 percent.",
      [gold(50, "table", "| Niger | 1.4 | 11.5 | 2.0 | 9.1 | 6.2 | 5.1 |  | -3.7 | -1.2 |",
            table_id="p050-t01", row="Niger", column="2024f")]),
    q("ad-t04", "dev", "table",
      "In Table 4.1, which small state is marked both tourism reliant and 'In distress' for risk of overall debt distress?",
      "Grenada.",
      [gold(194, "table", "| Grenada | X |  |  | X | In distress | X | Blend |",
            table_id="p194-t01", row="Grenada")]),
    q("ad-i01", "dev", "image",
      "Which region's locator map highlights China, Indonesia, and Viet Nam?",
      "East Asia and Pacific.",
      [gold(75, "image", "highlights East Asia and Pacific (EAP) in orange/yellow: China, Indonesia, the Philippines, Viet Nam")]),
    q("ad-i02", "dev", "image",
      "Which region's locator map highlights the Russian Federation and Türkiye?",
      "Europe and Central Asia.",
      [gold(83, "image", "highlights Europe and Central Asia (ECA) in orange/yellow, covering the Russian Federation, Türkiye")]),
    q("ad-c01", "dev", "chart",
      "According to Figure 1.3 panel C, what are the global trade growth forecasts for 2024 and 2025?",
      "2.5 percent in 2024 and 3.4 percent in 2025.",
      [gold(31, "chart", "forecast about 2.5% (2024f) and 3.4% (2025f)")]),
    q("ad-c02", "dev", "chart",
      "In Figure 3.1 panel A, what share of GDP per year do low-income countries need for resilient and low-carbon infrastructure?",
      "About 8 percent of GDP per year.",
      [gold(128, "chart", "low income (~8%)")]),
    q("ad-x01", "dev", "text",
      "What is global inflation projected to average in 2024?",
      "3.5 percent.",
      [gold(19, "text", "Global inflation is projected to moderate—but at a slower clip than previously assumed, averaging 3.5 percent this year.")]),
    q("ad-x02", "dev", "text",
      "What period does the public investment dataset cover?",
      "1980-2019.",
      [gold(135, "text", "spanning the period 1980-2019")]),
    q("ad-x03", "dev", "text",
      "What is LIC growth forecast to rise from in 2023 to in 2024?",
      "From 3.8 percent in 2023 to 5 percent in 2024.",
      [gold(28, "text", "Growth in LICs is forecast to improve over the forecast horizon from a subdued 3.8 percent in 2023 to 5 percent this year.")]),
    q("ad-x04", "dev", "text",
      "What average global growth rate does the foreword project through 2026, and how does it compare with the pre-COVID decade?",
      "2.7 percent a year through 2026, below the 3.1 percent average in the decade before COVID-19.",
      [gold(17, "text", "2.7 percent a year on average through 2026, well below the 3.1 percent average in the decade before COVID-19")]),
    q("ad-x05", "dev", "text",
      "What slowed the recovery in the Sahel region?",
      "A surge in conflict.",
      [gold(46, "text", "particularly in the Sahel region of Africa following a surge in conflict last year")]),
    q("ad-m01", "dev", "multi_hop",
      "Compare China's 2024 growth forecast in Table 1.1 with the East Asia and Pacific aggregate in Table 2.1.1.",
      "Both are 4.8 percent.",
      [gold(26, "table", "| China | 8.4 | 3.0 | 5.2 | 4.8 | 4.1 | 4.0 |  | 0.3 | -0.2 |",
            table_id="p026-t01", row="China", column="2024f"),
       gold(81, "table", "| EMDE EAP, GDP 1 | 7.6 | 3.4 | 5.1 | 4.8 | 4.2 | 4.1 |  | 0.3 | -0.2 |",
            table_id="p081-t01", row="EMDE EAP, GDP 1", column="2024f")]),
    q("ad-m02", "dev", "multi_hop",
      "According to the Executive Summary, at what rate does global growth stabilize in 2024, and what does Table 1.1 forecast for 2025?",
      "2.6 percent in 2024 and 2.7 percent in 2025.",
      [gold(19, "text", "Global growth is projected to stabilize at 2.6 percent this year"),
       gold(26, "table", "| World | 6.3 | 3.0 | 2.6 | 2.6 | 2.7 | 2.7 |  | 0.2 | 0.0 |",
            table_id="p026-t01", row="World", column="2025f")]),
    q("au01", "dev", "unanswerable",
      "What is the World Bank's GDP growth forecast for Vietnam in 2027?",
      "The report's forecast tables stop at 2026; no 2027 forecast is provided.",
      [], answerable=False,
      absence_checks=[r"2027"]),
    q("au02", "dev", "unanswerable",
      "How many employees work at the World Bank?",
      "The report does not state a World Bank headcount.",
      [], answerable=False,
      absence_checks=[r"World Bank.{0,80}(employees|headcount)", r"(employees|headcount).{0,40}World Bank"]),
    q("au03", "dev", "unanswerable",
      "Who is the photographer of the report's cover?",
      "The report does not credit a cover photographer.",
      [], answerable=False,
      absence_checks=[r"photographer", r"cover photo"]),
]

TEST = [
    q("at-t01", "test", "table",
      "According to Table 1.1, what is the World real GDP growth forecast for 2024?",
      "2.6 percent.",
      [gold(26, "table", "| World | 6.3 | 3.0 | 2.6 | 2.6 | 2.7 | 2.7 |  | 0.2 | 0.0 |",
            table_id="p026-t01", row="World", column="2024f")]),
    q("at-t02", "test", "table",
      "According to Table 1.1, what is China's real GDP growth forecast for 2024?",
      "4.8 percent.",
      [gold(26, "table", "| China | 8.4 | 3.0 | 5.2 | 4.8 | 4.1 | 4.0 |  | 0.3 | -0.2 |",
            table_id="p026-t01", row="China", column="2024f")]),
    q("at-t03", "test", "table",
      "According to Table B1.2.1, what is Rwanda's real GDP growth forecast for 2024?",
      "7.6 percent.",
      [gold(50, "table", "| Rwanda | 10.9 | 8.2 | 8.2 | 7.6 | 7.8 | 7.5 |  | 0.1 | 0.0 |",
            table_id="p050-t01", row="Rwanda", column="2024f")]),
    q("at-t04", "test", "table",
      "In Table 2.5.1, what is the 2024 real GDP growth forecast for South Asia?",
      "6.2 percent.",
      [gold(112, "table", "| EMDE South Asia, GDP 1 | 8.6 | 5.8 | 6.6 | 6.2 | 6.2 | 6.2 |  | 0.6 | 0.3 |",
            table_id="p112-t01", row="EMDE South Asia, GDP 1", column="2024f")]),
    q("at-t05", "test", "table",
      "In Table A3.2.1, how many economies are in the sample used to estimate public investment multipliers?",
      "129 economies.",
      [gold(159, "table", "| Number of economies | 129 |", table_id="p159-t01", row="Number of economies")]),
    q("at-t06", "test", "table",
      "In Table 4.1, which small state is marked both as a commodity exporter and 'In distress' for risk of overall debt distress?",
      "São Tomé and Príncipe.",
      [gold(194, "table", "| São Tomé and Príncipe |  | X | X |  | In distress | X | IDA |",
            table_id="p194-t01", row="São Tomé and Príncipe")]),
    q("at-i01", "test", "image",
      "Which region's locator map highlights Brazil and Mexico?",
      "Latin America and the Caribbean.",
      [gold(91, "image", "highlights Latin America and the Caribbean (LAC) in orange/yellow, covering Mexico, Brazil")]),
    q("at-i02", "test", "image",
      "Which region's locator map highlights Saudi Arabia and the Islamic Republic of Iran?",
      "Middle East and North Africa.",
      [gold(99, "image", "highlights the Middle East and North Africa (MNA) in orange/yellow, including Saudi Arabia, the Islamic Republic of Iran")]),
    q("at-i03", "test", "image",
      "Which region's locator map highlights India and Pakistan?",
      "South Asia.",
      [gold(107, "image", "highlights South Asia (SAR) in orange/yellow, covering India, Pakistan")]),
    q("at-i04", "test", "image",
      "Which region's locator map highlights Nigeria and South Africa?",
      "Sub-Saharan Africa.",
      [gold(115, "image", "highlights Sub-Saharan Africa (SSA) in orange/yellow, covering West, East, Central and Southern Africa, including Nigeria, South Africa")]),
    q("at-i05", "test", "image",
      "What ISBN is printed above the barcode on the back cover?",
      "978-1-4648-2058-8.",
      [gold(220, "image", "ISBN 978-1-4648-2058-8")]),
    q("at-c01", "test", "chart",
      "In Figure 1.1 panel D, what percentage of EMDEs are vulnerable to debt-related stress?",
      "About 38 percent.",
      [gold(27, "chart", "EMDES about 38")]),
    q("at-c02", "test", "chart",
      "In Figure 1.2 panel B, which year has the highest global trade policy uncertainty index?",
      "2024.",
      [gold(29, "chart", "2024 highest near 75")]),
    q("at-c03", "test", "chart",
      "In Figure 4.15 panel A, what share of GDP do small states receive in official development assistance?",
      "About 14 percent of GDP.",
      [gold(192, "chart", "small states, about 14%")]),
    q("at-c04", "test", "chart",
      "What does Figure 1.5 panel D show about global consumer price inflation by 2026?",
      "It declines toward the inflation target by 2026.",
      [gold(34, "chart", "inflation declining toward target by 2026")]),
    q("at-c05", "test", "chart",
      "In Figure 1.6 panel A, what level do market expectations for the U.S. policy rate reach in 2024–25?",
      "About 5–6 percent.",
      [gold(35, "chart", "about 5–6 percent and flattening through 2024–25")]),
    q("at-x01", "test", "text",
      "What is global growth projected to be in 2025-26?",
      "2.7 percent.",
      [gold(25, "text", "before edging up to 2.7 percent in 2025-26")]),
    q("at-x02", "test", "text",
      "How far below the 2010-19 average is global growth projected to be in 2024?",
      "Half a percentage point below the 2010-19 average.",
      [gold(52, "text", "Global growth is projected to remain subdued at 2.6 percent in 2024—half a percentage point below the 2010-19 average")]),
    q("at-x03", "test", "text",
      "How many EMDE small states are in the chapter's sample?",
      "Up to 35.",
      [gold(173, "text", "up to 35 EMDE small states")]),
    q("at-x04", "test", "text",
      "What is the average VAT rate in small states, and how does it compare with other EMDEs?",
      "14 percent, about 2 percentage points lower than the average in other EMDEs.",
      [gold(186, "text", "VAT rates in small states average 14 percent, around 2 percentage points lower than the average in other EMDEs.")]),
    q("at-m01", "test", "multi_hop",
      "Compare world growth and East Asia and Pacific growth in 2024. Which is higher, and by how many percentage points?",
      "East Asia and Pacific is higher: 4.8 percent versus 2.6 percent for the world, a difference of 2.2 percentage points.",
      [gold(26, "table", "| World | 6.3 | 3.0 | 2.6 | 2.6 | 2.7 | 2.7 |  | 0.2 | 0.0 |",
            table_id="p026-t01", row="World", column="2024f"),
       gold(81, "table", "| EMDE EAP, GDP 1 | 7.6 | 3.4 | 5.1 | 4.8 | 4.2 | 4.1 |  | 0.3 | -0.2 |",
            table_id="p081-t01", row="EMDE EAP, GDP 1", column="2024f")]),
    q("at-m02", "test", "multi_hop",
      "What does the South Asia chapter say about the region's growth slowdown, and what does Table 2.5.1 show for 2024?",
      "Growth is projected to slow from 6.6 percent in 2023 to 6.2 percent in 2024; Table 2.5.1 also shows 6.2 percent for 2024.",
      [gold(107, "text", "Growth in the South Asia (SAR) region is projected to slow from 6.6 percent in 2023 to 6.2 percent in 2024"),
       gold(112, "table", "| EMDE South Asia, GDP 1 | 8.6 | 5.8 | 6.6 | 6.2 | 6.2 | 6.2 |  | 0.6 | 0.3 |",
            table_id="p112-t01", row="EMDE South Asia, GDP 1", column="2024f")]),
    q("at-m03", "test", "multi_hop",
      "What happened to trade policy uncertainty in 2024 while global trade growth was forecast at 2.5 percent?",
      "Trade policy uncertainty was at its highest level in 2024, while global trade growth was forecast at 2.5 percent in 2024.",
      [gold(29, "chart", "2024 highest near 75"),
       gold(31, "chart", "forecast about 2.5% (2024f) and 3.4% (2025f)")]),
    q("at-m04", "test", "multi_hop",
      "How much faster is LIC growth than world growth in 2024?",
      "LIC growth is 5.0 percent versus 2.6 percent for the world, so it is 2.4 percentage points faster.",
      [gold(50, "table", "| Low-Income Countries, GDP b | 4.1 | 5.0 | 3.8 | 5.0 | 5.3 | 5.5 |  | -0.5 | -0.3 |",
            table_id="p050-t01", row="Low-Income Countries, GDP b", column="2024f"),
       gold(26, "table", "| World | 6.3 | 3.0 | 2.6 | 2.6 | 2.7 | 2.7 |  | 0.2 | 0.0 |",
            table_id="p026-t01", row="World", column="2024f")]),
    q("au04", "test", "unanswerable",
      "Which country had the highest foreign direct investment inflow in 2023?",
      "The report discusses FDI only qualitatively and does not rank countries by 2023 inflows.",
      [], answerable=False,
      absence_checks=[r"FDI.{0,100}(highest|largest|top)", r"foreign direct investment.{0,100}(highest|largest|top)"]),
    q("au05", "test", "unanswerable",
      "What is the exact population of India in 2024?",
      "The report does not provide an exact 2024 population for India.",
      [], answerable=False,
      absence_checks=[r"population of India", r"India.{0,40}population"]),
    q("au06", "test", "unanswerable",
      "How many member countries does the World Bank have?",
      "The report does not state the number of World Bank member countries.",
      [], answerable=False,
      absence_checks=[r"\d{2,3}\s+member countries", r"member countries.{0,20}\d"]),
    q("au07", "test", "unanswerable",
      "Who is the President of the World Bank?",
      "The report names the Senior Vice President and Chief Economist but not the President.",
      [], answerable=False,
      absence_checks=[r"President of the World Bank", r"World Bank President"]),
    q("au08", "test", "unanswerable",
      "What was the World Bank's total lending in fiscal year 2024?",
      "The report does not state total World Bank lending for fiscal 2024.",
      [], answerable=False,
      absence_checks=[r"total lending.{0,80}(2024|fiscal)", r"(fiscal year 2024|FY2024).{0,80}(lending|commitments)"]),
    q("au09", "test", "unanswerable",
      "Which country had the highest inflation rate in 2023?",
      "The report does not rank countries by 2023 inflation.",
      [], answerable=False,
      absence_checks=[r"country.{0,50}(highest|largest).{0,50}inflation", r"highest inflation rate"]),
]


def main() -> None:
    chunks = json.loads(CHUNKS.read_text(encoding="utf-8"))
    by_page_type: dict[tuple[int, str], list[dict]] = {}
    for chunk in chunks:
        by_page_type.setdefault((chunk["page"], chunk["content_type"]), []).append(chunk)
    corpus = "\n".join(chunk["text"] for chunk in chunks)
    questions = DEV + TEST
    for question in questions:
        if not question.get("answerable", True):
            for pattern in question.get("absence_checks", []):
                if re.search(pattern, corpus, flags=re.IGNORECASE):
                    raise SystemExit(f"{question['id']}: absence pattern matched: {pattern}")
            continue
        for item in question["gold"]:
            candidates = by_page_type.get((int(item["page"]), item["content_type"]), [])
            evidence = " ".join(item["evidence"].split()).casefold()
            if not any(evidence in " ".join(chunk["text"].split()).casefold() for chunk in candidates):
                raise SystemExit(
                    f"{question['id']}: evidence not found on page {item['page']} "
                    f"as {item['content_type']}: {item['evidence'][:100]!r}"
                )
    counts = {
        split: {
            "answerable": sum(1 for q_ in questions if q_["split"] == split and q_.get("answerable", True)),
            "unanswerable": sum(1 for q_ in questions if q_["split"] == split and not q_.get("answerable", True)),
        }
        for split in ("dev", "test")
    }
    groups = {
        split: {
            group: sum(1 for q_ in questions if q_["split"] == split and q_["group"] == group)
            for group in ("table", "image", "chart", "multi_hop", "text", "unanswerable")
        }
        for split in ("dev", "test")
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"questions": len(questions), "counts": counts, "groups": groups}, indent=2))


if __name__ == "__main__":
    main()
