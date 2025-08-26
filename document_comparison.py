import hashlib
from pathlib import Path
from typing import List, Tuple, Dict
import html

import streamlit as st
import difflib

try:  # Graceful fallback if pypdf missing at runtime
	from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover
	PdfReader = None  # type: ignore


@st.cache_data(show_spinner=False)
def load_pdf_text(path: str) -> str:
	"""Extract text content from a PDF file.

	Returns all page texts joined with form feed separators so we can optionally split later.
	Cached for performance across reruns.
	"""
	p = Path(path)
	if not p.exists():
		raise FileNotFoundError(path)
	if PdfReader is None:
		return "<PdfReader not available>"
	try:
		reader = PdfReader(str(p))
	except Exception as e:  # pragma: no cover
		return f"<ERROR opening {p.name}: {e}>"
	pages = []
	for i, page in enumerate(reader.pages):
		try:
			txt = page.extract_text() or ""
		except Exception as e:
			txt = f"<ERROR page {i}: {e}>"
		pages.append(txt)
	return "\f".join(pages)


def normalize(text: str) -> List[str]:
	"""Produce a normalized list of lines for diffing.

	- Lowercase (configurable later)
	- Collapse internal whitespace
	- Drop empty lines
	"""
	norm = []
	for raw in text.splitlines():
		s = " ".join(raw.strip().split())
		if s:
			norm.append(s.lower())
	return norm


def hash_text(text: str) -> str:
	return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:12]


def unified_diff(a_lines: List[str], b_lines: List[str], a_name: str, b_name: str) -> str:
	return "\n".join(
		difflib.unified_diff(a_lines, b_lines, fromfile=a_name, tofile=b_name, lineterm="")
	)


def side_by_side_diff(a_lines: List[str], b_lines: List[str]) -> List[Tuple[str, str, str]]:
	"""Return tuples of (tag, left, right) using SequenceMatcher grouped opcodes.
	tag in {equal, replace, delete, insert}
	"""
	sm = difflib.SequenceMatcher(a=a_lines, b=b_lines, autojunk=False)
	rows: List[Tuple[str, str, str]] = []
	for tag, i1, i2, j1, j2 in sm.get_opcodes():
		left_block = a_lines[i1:i2]
		right_block = b_lines[j1:j2]
		max_len = max(len(left_block), len(right_block)) or 1
		for k in range(max_len):
			left_line = left_block[k] if k < len(left_block) else ""
			right_line = right_block[k] if k < len(right_block) else ""
			rows.append((tag, left_line, right_line))
	return rows


def difference_summary(base: List[str], other: List[str]) -> Dict[str, int]:
	sm = difflib.SequenceMatcher(a=base, b=other, autojunk=False)
	stats = {"equal": 0, "replace": 0, "delete": 0, "insert": 0}
	for tag, i1, i2, j1, j2 in sm.get_opcodes():
		if tag == "equal":
			stats["equal"] += (i2 - i1)
		elif tag == "replace":
			stats["replace"] += max(i2 - i1, j2 - j1)
		elif tag == "delete":
			stats["delete"] += (i2 - i1)
		elif tag == "insert":
			stats["insert"] += (j2 - j1)
	return stats


def color_for_tag(tag: str) -> str:
	return {
		"equal": "#f8f9fa",
		"replace": "#fff3cd",
		"delete": "#f8d7da",
		"insert": "#d4edda",
	}.get(tag, "#ffffff")


def render_side_by_side(rows: List[Tuple[str, str, str]]):
	st.markdown(
		"""
		<style>
		.diff-table {width:100%; border-collapse:collapse; font-family:monospace; font-size:13px;}
		.diff-table th, .diff-table td {padding:2px 6px; vertical-align:top; color:#000;}
		.diff-table tr td.line {width:50%;}
		.diff-tag {font-weight:bold; color:#000; width:60px;}
		</style>
		""",
		unsafe_allow_html=True,
	)
	table_html = ["<table class='diff-table'>", "<tr><th>Tag</th><th>Left</th><th>Right</th></tr>"]
	for tag, left, right in rows:
		bg = color_for_tag(tag)
		table_html.append(
			f"<tr style='background:{bg}'><td class='diff-tag'>{tag}</td><td class='line'>{html.escape(left)}</td><td class='line'>{html.escape(right)}</td></tr>"
		)
	table_html.append("</table>")
	st.markdown("\n".join(table_html), unsafe_allow_html=True)


def main():
	st.set_page_config(page_title="PDF Change Viewer", layout="wide")
	st.title("📑 Multi‑Document Change Viewer")
	st.caption("Select multiple PDF documents and inspect textual differences.")

	# Source directory selection
	default_dir = Path("documents/1")
	base_dir = st.text_input("Directory containing PDFs", str(default_dir.resolve()))
	base_path = Path(base_dir)
	if not base_path.exists():
		st.error("Directory does not exist.")
		return

	pdf_files = sorted([p for p in base_path.glob("*.pdf")])
	if not pdf_files:
		st.warning("No PDF files found in directory.")
		return

	multiselect = st.multiselect(
		"Choose documents to compare (2-6)",
		options=[p.name for p in pdf_files],
		default=[pdf_files[0].name] + ([pdf_files[1].name] if len(pdf_files) > 1 else []),
	)
	if len(multiselect) < 2:
		st.info("Select at least two PDFs.")
		return
	if len(multiselect) > 6:
		st.warning("Limiting to first 6 selected for performance.")
		multiselect = multiselect[:6]

	# Load texts
	texts = {}
	for name in multiselect:
		path = base_path / name
		with st.spinner(f"Extracting text: {name}"):
			raw = load_pdf_text(str(path))
		texts[name] = raw

	st.subheader("Document Hashes")
	col_count = min(4, len(texts))
	cols = st.columns(col_count)
	for i, (name, raw) in enumerate(texts.items()):
		with cols[i % col_count]:
			st.code(f"{name}\nSHA256-12: {hash_text(raw)}")

	# Choose base document
	base_doc = st.selectbox("Base (reference) document", multiselect, index=0)
	base_norm = normalize(texts[base_doc])

	st.subheader("Pairwise Difference Summaries vs Base")
	summary_rows = []
	for name, raw in texts.items():
		if name == base_doc:
			continue
		stats = difference_summary(base_norm, normalize(raw))
		total = sum(stats.values()) or 1
		changed = total - stats["equal"]
		pct = 100 * changed / total
		summary_rows.append((name, stats, pct))
	if summary_rows:
		for name, stats, pct in summary_rows:
			st.write(
				f"**{name}**: changed lines {stats['replace'] + stats['delete'] + stats['insert']} / {sum(stats.values())} (≈ {pct:.1f}% different)"
			)
			st.progress(min(1.0, pct / 100.0))

	# Detailed diff selection
	st.subheader("Detailed Diff Viewer")
	target = st.selectbox(
		"Compare base against...", [n for n in multiselect if n != base_doc]
	)
	view_mode = st.radio("View mode", ["Side-by-side", "Unified"], horizontal=True)
	a_norm = base_norm
	b_norm = normalize(texts[target])

	if view_mode == "Unified":
		diff_text = unified_diff(a_norm, b_norm, base_doc, target)
		st.code(diff_text or "<no differences>", language="diff")
	else:
		rows = side_by_side_diff(a_norm, b_norm)
		render_side_by_side(rows)

	# Raw text expander
	with st.expander("Show raw extracted text"):
		for name, raw in texts.items():
			st.markdown(f"#### {name}")
			st.text_area(f"Text {name}", raw[:100000], height=180)


if __name__ == "__main__":  # pragma: no cover
	main()

