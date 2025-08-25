import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import streamlit as st
from graphviz import Digraph
import requests

DATA_URL = "https://www.fda.gov/files/api/datatables/static/search-for-guidance.json"

DOCKET_PATTERN_PRIMARY = re.compile(r"FDA-\d{4}-[A-Z]-\d+")
DOCKET_PATTERN_FALLBACK = re.compile(r"FDA-\d{4}-[A-Z]-\d+(?:-\d+)?")
MEDIA_HREF_RE = re.compile(r'href=["\'](/media/(\d+)/download)["\']', re.IGNORECASE)
MEDIA_TITLE_RE = re.compile(r'<span[^>]*class=["\']sr-only["\'][^>]*>[^<]*? of (.*?)</span>', re.IGNORECASE)
ANCHOR_RE = re.compile(r'<a [^>]*>.*?</a>', re.IGNORECASE | re.DOTALL)


def clean_docket(raw: str):
    if not raw or not isinstance(raw, str):
        return None
    no_tags = re.sub(r"<[^>]+>", " ", raw)
    m = DOCKET_PATTERN_PRIMARY.search(no_tags)
    if m:
        return m.group(0)
    m2 = DOCKET_PATTERN_FALLBACK.search(no_tags)
    if m2:
        return m2.group(0)
    tok = no_tags.strip()
    if tok.startswith("FDA-"):
        return tok.split()[0]
    return None


def parse_media(media_html: str):
    results = []
    if not media_html or not isinstance(media_html, str):
        return results
    anchors = ANCHOR_RE.findall(media_html) or [media_html]
    existing_ids = set()
    for anchor in anchors:
        href_match = MEDIA_HREF_RE.search(anchor)
        if not href_match:
            continue
        path, file_id = href_match.group(1), href_match.group(2)
        if file_id in existing_ids:
            continue
        existing_ids.add(file_id)
        file_url = f"https://www.fda.gov{path}" if path.startswith('/') else path
        title_match = MEDIA_TITLE_RE.search(anchor)
        if title_match:
            title = title_match.group(1).strip()
        else:
            no_tags = re.sub(r"<[^>]+>", " ", anchor)
            parts = no_tags.split(" of ")
            title = parts[-1].strip() if len(parts) > 1 else no_tags.strip()
        results.append({"file_id": file_id, "title": title, "url": file_url})
    return results


def fetch_and_structure() -> Dict[str, List[dict]]:
    resp = requests.get(DATA_URL, headers={"User-Agent": "curl/7.68.0"}, timeout=60)
    resp.raise_for_status()
    raw_data = resp.json()
    docket_map = defaultdict(list)
    for entry in raw_data:
        # Filter to medical devices only similar to notebook
        if "Medical Devices" not in entry.get("field_regulated_product_field", ""):
            continue
        docket = clean_docket(entry.get("field_docket_number"))
        if not docket:
            continue
        media_html = entry.get("field_associated_media_2")
        media_items = parse_media(media_html)
        for item in media_items:
            if all(existing["file_id"] != item["file_id"] for existing in docket_map[docket]):
                docket_map[docket].append(item)
    return dict(docket_map)


def build_graph(docket_media: Dict[str, List[dict]], limit: int = 50) -> Digraph:
    g = Digraph("dockets", graph_attr={"rankdir": "LR", "splines": "true"})
    g.attr("node", shape="box", style="filled", color="#1f77b4", fontname="Helvetica", fontsize="10", fillcolor="#e8f2fb")
    count = 0
    for docket, items in sorted(docket_media.items()):
        if count >= limit:
            break
        docket_node = docket
        g.node(docket_node, docket_node)
        for itm in items:
            file_node_id = f"{docket}_{itm['file_id']}"
            label = f"{itm['file_id']}\n{itm['title'][:40]}{'...' if len(itm['title'])>40 else ''}"
            g.node(file_node_id, label, shape="note", color="#ff7f0e", fillcolor="#fff4e6")
            g.edge(docket_node, file_node_id)
        count += 1
    return g


def main():
    st.set_page_config(page_title="FDA Docket Media Graph", layout="wide")
    st.title("FDA Medical Device Guidance Dockets")
    st.write("Interactive visualization: Docket numbers connected to their associated media files.")

    with st.sidebar:
        st.header("Options")
        fetch = st.button("Fetch / Refresh Data")
        max_dockets = st.slider("Max dockets to display", 5, 200, 50, 5)
        show_json = st.checkbox("Show raw JSON mapping", False)

    if fetch or 'docket_media' not in st.session_state:
        with st.spinner("Fetching FDA data..."):
            try:
                mapping = fetch_and_structure()
                st.session_state['docket_media'] = mapping
                st.success(f"Fetched {len(mapping)} dockets.")
            except Exception as e:
                st.error(f"Failed to fetch data: {e}")
                return

    docket_media = st.session_state.get('docket_media', {})

    st.subheader("Graph View")
    graph = build_graph(docket_media, limit=max_dockets)
    st.graphviz_chart(graph)

    # Detail explorer
    st.subheader("Details")
    selected_docket = st.selectbox("Select a docket", sorted(docket_media.keys())) if docket_media else None
    if selected_docket:
        items = docket_media[selected_docket]
        st.markdown(f"### {selected_docket} ({len(items)} files)")
        for itm in items:
            st.markdown(f"- **{itm['file_id']}**: [{itm['title']}]({itm['url']})")

    if show_json and docket_media:
        st.subheader("Raw Mapping JSON")
        st.json(docket_media)

    st.caption("Data source: FDA guidance documents static JSON endpoint.")


if __name__ == "__main__":
    main()
