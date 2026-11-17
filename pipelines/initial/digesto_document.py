"""Conservative parsing of one Digesto main response; no version endpoints."""

from dataclasses import dataclass, field
import codecs
import json
import re
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup, Comment, Declaration, Doctype, NavigableString, ProcessingInstruction, Tag


EXTRACTOR_VERSION = "3"
ARTICLE_NUMBER = r"(?P<label>único|unico|\d+(?:[º°o])?(?:\s*(?:bis|ter|qu[aá]ter))?)"
ARTICLE = re.compile(
    r"^(?:art[íi]culo|arto?\.?)\s*" + ARTICLE_NUMBER + r"(?=[\s.\-–—:]|$)", re.I)
BARE_ARTICLE_NUMBER = re.compile(r"^" + ARTICLE_NUMBER + r"(?=[\s.\-–—:]|$)", re.I)
SECTION = re.compile(r"^(libro|t[íi]tulo|cap[íi]tulo|secci[óo]n|anexo)\s+(\S+)", re.I)
LEGAL_FORMULA = re.compile(
    r"(?:el\s+(?:presidente|director supremo)[^.!?]{0,180}"
    r"(?:acuerda|decreta|resuelve|hace\s+saber|a\s+sus\s+habitantes)|"
    r"ha\s+dictado|por\s+cuanto|en\s+uso\s+de\s+sus\s+facultades|"
    r"(?:acuerda|decreta|resuelve|certifica)\s*[:;])", re.I)
ERROR_PAGE = re.compile(
    r"^\s*(?:error\s+(?:interno|del\s+servidor|\d{3})|internal\s+server\s+error|"
    r"service\s+unavailable|bad\s+gateway|not\s+found|fatal\s+error|"
    r"warning\s*:|acceso\s+denegado|inicie\s+sesi[óo]n|access\s+denied)", re.I)
BLOCKS = {"p", "div", "section", "article", "main", "h1", "h2", "h3", "h4",
          "h5", "h6", "li", "ul", "ol", "blockquote", "pre", "table", "center"}


@dataclass
class DocumentExtraction:
    status: str
    reason: str
    text: str = ""
    nodes: list[dict] = field(default_factory=list)
    encoding: str | None = None


def decode_response(body: bytes, content_type: str) -> tuple[str, str]:
    """Respect explicit encodings; never silently insert replacement characters."""
    if body.startswith(codecs.BOM_UTF8):
        encoding = "utf-8-sig"
    elif body.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        encoding = "utf-16"
    else:
        declaration = re.search(
            r"(?:encoding|charset)\s*=\s*[\"']?([a-zA-Z0-9._-]+)",
            body[:1000].decode("ascii", errors="ignore"), re.I)
        header = re.search(r"charset\s*=\s*[\"']?([a-zA-Z0-9._-]+)", content_type, re.I)
        encoding = (declaration or header).group(1) if declaration or header else "utf-8"
    codecs.lookup(encoding)
    source = body.decode(encoding, errors="strict")
    if "\ufffd" in source:
        raise UnicodeError("Source contains Unicode replacement characters")
    return source, encoding


def inline_text(value) -> str:
    """Collect visible text recursively, preserving line breaks and ignoring comments."""
    if isinstance(value, (Comment, Declaration, Doctype, ProcessingInstruction)):
        return ""
    if isinstance(value, NavigableString):
        return str(value)
    if isinstance(value, Tag):
        if value.name == "br":
            return "\n"
        return "".join(inline_text(child) for child in value.children)
    return str(value)


def normalize_block(value: str) -> str:
    """Collapse whitespace within nonempty lines while preserving their boundaries."""
    return "\n".join(re.sub(r"\s+", " ", line).strip()
                     for line in value.splitlines() if line.strip())


def locator(tag: Tag) -> str:
    """Build an HTML location using tag names and sibling positions."""
    parts = []
    while isinstance(tag, Tag) and tag.name != "[document]":
        index = 1 + sum(isinstance(sibling, Tag) and sibling.name == tag.name
                        for sibling in tag.previous_siblings)
        parts.append(f"{tag.name}:nth-of-type({index})")
        tag = tag.parent
    return " > ".join(reversed(parts)) or "[document]"


def block_links(parts, document_url: str) -> list[dict]:
    """Collect HTTP links with their labels, original targets, and HTML locations."""
    links = []
    for part in parts:
        if not isinstance(part, Tag):
            continue
        anchors = ([part] if part.name == "a" else []) + list(part.find_all("a", href=True))
        for anchor in anchors:
            href = anchor.get("href", "")
            target = urljoin(document_url, href)
            if href and urlsplit(target).scheme in ("http", "https"):
                links.append({"text": normalize_block(inline_text(anchor)), "href": href,
                              "url": target, "source_locator": locator(anchor)})
    return links


def document_blocks(soup: BeautifulSoup, document_url: str) -> list[dict]:
    """Extract ordered text and table blocks with links and source article context."""
    blocks = []
    article_contexts = {}

    def source_context(tag):
        """Find the enclosing article and document region, caching article labels."""
        article = None
        region = None
        for ancestor in [tag, *tag.parents]:
            if not isinstance(ancestor, Tag):
                continue
            classes = set(ancestor.get("class", []))
            if article is None and "article" in classes:
                key = id(ancestor)
                if key not in article_contexts:
                    number = ancestor.select_one(".num")
                    raw_label = normalize_block(inline_text(number)) if number else ""
                    match = ARTICLE.match(raw_label) or BARE_ARTICLE_NUMBER.match(raw_label)
                    if not raw_label:
                        match = ARTICLE.match(normalize_block(inline_text(ancestor)))
                        raw_label = match.group(0) if match else ""
                    article_contexts[key] = {"locator": locator(ancestor),
                                             "label": match.group("label") if match else None,
                                             "source_label": raw_label,
                                             "internal_id": ancestor.get("internalid")}
                article = article_contexts[key]
            if region is None:
                region = next((name for name in ("preface", "preamble", "conclusions")
                               if name in classes), None)
        return {"_article": article, "_region": region}

    def emit(tag, parts, kind):
        """Append a nonempty block with normalized text and source metadata."""
        text = normalize_block("".join(inline_text(part) for part in parts))
        if text:
            blocks.append({"type": kind, "text": text, "source_locator": locator(tag),
                           "links": block_links(parts, document_url), **source_context(tag)})

    def walk(tag, kind="paragraph"):
        """Traverse markup, separating text blocks and distinguishing data from layout tables."""
        buffered = []

        def flush():
            """Emit the buffered inline content and clear it before the next block."""
            emit(tag, buffered, kind)
            buffered.clear()

        for child in tag.children:
            if not isinstance(child, Tag):
                buffered.append(child)
                continue
            if child.name == "table":
                flush()
                # Older legal HTML uses tables for page layout. Descend into those
                # so the whole law is not collapsed into a single table cell.
                paragraphs = child.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
                legal_layout = any(ARTICLE.match(normalize_block(inline_text(part)))
                                   or SECTION.match(normalize_block(inline_text(part)))
                                   for part in paragraphs)
                if child.find("table") or legal_layout:
                    walk(child)
                    continue
                rows = []
                cells = []
                for row in child.find_all("tr"):
                    # A nested table owns its own rows.
                    if row.find_parent("table") is child:
                        row_cells = row.find_all(["td", "th"], recursive=False)
                        rows.append([normalize_block(inline_text(cell)) for cell in row_cells])
                        cells.append([{"text": normalize_block(inline_text(cell)),
                                       "header": cell.name == "th", "rowspan": cell.get("rowspan", "1"),
                                       "colspan": cell.get("colspan", "1")}
                                      for cell in row_cells])
                rows = [row for row in rows if row]
                text = "\n".join(" | ".join(row) for row in rows)
                if text:
                    blocks.append({"type": "table", "text": text, "rows": rows, "cells": cells,
                                   "source_locator": locator(child),
                                   "links": block_links([child], document_url), **source_context(child)})
                else:
                    walk(child)
            elif child.name in BLOCKS or "article" in child.get("class", []) or child.find(list(BLOCKS)):
                flush()
                child_kind = "list_item" if child.name == "li" else (
                    "heading" if re.fullmatch(r"h[1-6]", child.name) else "paragraph")
                walk(child, child_kind)
            else:
                buffered.append(child)
        flush()

    walk(soup.body or soup)
    return blocks


def structured_nodes(blocks: list[dict]) -> tuple[str, list[dict]]:
    """Return document text and hierarchical nodes with exact character spans.

    Article containers and recognized headings establish parents. Parent text
    includes descendants, and offsets use an exclusive end position.
    """
    text = "\n".join(block["text"] for block in blocks)
    nodes = []
    stack = []
    offset = 0
    active_article_scope = None
    for block in blocks:
        content = block["text"]
        source_article = block.get("_article")
        scope = source_article["locator"] if source_article else None
        region = block.get("_region")
        if region in ("preface", "preamble", "conclusions") and source_article is None:
            stack.clear()
        if scope != active_article_scope:
            while stack and stack[-1][0] >= 5:
                stack.pop()
            active_article_scope = scope
            if source_article:
                # One node per actual article container, even when its number
                # omits "Art." or its content spans several HTML blocks/tables.
                article_node = {"id": f"n{len(nodes) + 1:06d}", "type": "article",
                                "label": source_article["label"],
                                "source_label": source_article["source_label"],
                                "source_locator": source_article["locator"],
                                "source_internal_id": source_article["internal_id"],
                                "parent_id": stack[-1][1]["id"] if stack else None,
                                "start_char": offset, "end_char": offset,
                                "links": []}
                nodes.append(article_node)
                stack.append((5, article_node))
        article = ARTICLE.match(content)
        section = SECTION.match(content)
        level = None
        kind = block["type"]
        label = None
        if article and kind != "table" and source_article is None:
            kind, level, label = "article", 5, article.group("label")
        elif section and kind != "table" and source_article is None:
            name = section.group(1).lower()
            level = {"libro": 1, "título": 2, "titulo": 2, "capítulo": 3,
                     "capitulo": 3, "sección": 4, "seccion": 4, "anexo": 1}[name]
            kind, label = ("annex" if name == "anexo" else "heading"), section.group(2)
        if level is not None:
            while stack and stack[-1][0] >= level:
                stack.pop()
        if source_article is None and region is not None and kind == "paragraph":
            kind = "conclusion" if region == "conclusions" else region
        node = {**{key: value for key, value in block.items() if not key.startswith("_")},
                "id": f"n{len(nodes) + 1:06d}", "type": kind,
                "parent_id": stack[-1][1]["id"] if stack else None,
                "start_char": offset, "end_char": offset + len(content)}
        if label is not None:
            node["label"] = label
            node["source_label"] = (article or section).group(0)
        for _, parent in stack:
            parent["end_char"] = node["end_char"]
        nodes.append(node)
        if level is not None:
            stack.append((level, node))
        offset += len(content) + 1
    for node in nodes:
        node["text"] = text[node["start_char"]:node["end_char"]]
    return text, nodes


def extract_document(body: bytes, *, content_type: str = "", expected_title: str = "",
                     document_url: str = "") -> DocumentExtraction:
    """Decode and parse a main response into a technical status, text, and nodes.

    Empty, unusable, or ambiguous content returns a rejected extraction state.
    Access challenge pages raise RuntimeError so collection stops.
    """
    if not body or not body.strip():
        return DocumentExtraction("empty_response", "The main endpoint returned no content")
    media_type = content_type.partition(";")[0].strip().lower()
    if media_type and not (media_type.startswith("text/") or media_type in (
            "application/json", "application/xml", "application/xhtml+xml", "application/octet-stream")):
        return DocumentExtraction("no_usable_text", f"Unsupported content type: {media_type}")
    try:
        source, encoding = decode_response(body, content_type)
    except (UnicodeError, LookupError) as error:
        return DocumentExtraction("needs_review", f"Cannot safely decode response: {error}")
    source = re.sub(r"^\s*<\?xml\b[^>]*\?>", "", source, count=1, flags=re.I | re.S).strip()
    # Handle a JSON string/envelope without trusting error payloads as legal text.
    for _ in range(3):
        if not source or source[0] not in '{["':
            break
        try:
            data = json.loads(source)
        except ValueError:
            return DocumentExtraction("no_usable_text", "Malformed JSON envelope", encoding=encoding)
        if isinstance(data, dict):
            if any(data.get(key) for key in ("error", "errors")) or data.get("success") is False:
                return DocumentExtraction("no_usable_text", "Source returned an error envelope", encoding=encoding)
            source = next((data[key] for key in ("html", "content", "texto", "text")
                           if isinstance(data.get(key), str)), "").strip()
        elif isinstance(data, str):
            source = data.strip()
        else:
            source = ""
    if not source or re.match(r"^\s*<\s*(?:documents|versions|error)\b", source, re.I):
        return DocumentExtraction("no_usable_text", "No main legal document in response", encoding=encoding)
    if any(token in source[:2500].lower() for token in (
            "cf-chl-", "just a moment", "attention required!", "access denied")):
        raise RuntimeError("Digesto returned an access/security page. Stopping.")
    soup = BeautifulSoup(source, "html.parser")
    for tag in list(soup(["script", "style", "noscript", "head", "nav", "form", "button", "footer"])):
        tag.decompose()
    for table in soup.find_all("table"):
        if table.find("table"):
            outer_rows = [row for row in table.find_all("tr") if row.find_parent("table") is table]
            if any(len(row.find_all(["td", "th"], recursive=False)) > 1 for row in outer_rows):
                return DocumentExtraction("needs_review", "Nested multi-column table needs structural review",
                                          encoding=encoding)
    blocks = document_blocks(soup, document_url)
    text, nodes = structured_nodes(blocks)
    if not text:
        return DocumentExtraction("no_usable_text", "Document has no text content", encoding=encoding)
    if ERROR_PAGE.match(text):
        return DocumentExtraction("no_usable_text", "Source returned a service/error page", encoding=encoding)
    has_articles = any(node["type"] == "article" for node in nodes)
    formula = bool(LEGAL_FORMULA.search(text))
    # Title overlap supports short treaties/resolutions, but never proves a title-only page is a law.
    title_words = set(re.findall(r"\w{4,}", expected_title.casefold()))
    words = set(re.findall(r"\w{4,}", text.casefold()))
    title_match = bool(title_words) and len(title_words & words) / len(title_words) >= 0.6
    operative = bool(re.search(r"\b(?:acuerdan|convienen|concede|resuelve|establece|aprueba|ratifica)\b", text, re.I))
    if not (has_articles or formula or (title_match and operative)):
        return DocumentExtraction("needs_review", "Legal-document signals could not be established", encoding=encoding)
    return DocumentExtraction("ok", "Recognized main legal document; legal currency not assessed",
                              text, nodes, encoding)
