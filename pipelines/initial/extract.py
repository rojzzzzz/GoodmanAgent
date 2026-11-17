"""Digesto Jurídico Nicaragüense: metadata + main-text extractor.

Scope:
* Catalog ALL legal instruments. No category/rank filters.
* Extract only getNormaHtmlAccordion (no historical/version endpoints).
* Two current-state JSONL files. No legal version or attempt history.
* Resumable: catalog pages and current text checkpoints are cached.
* Log empty/unusable documents instead of silently dropping them.
* Do not evade HTTP restrictions or run faster than one request per second.

Dependencies: pip install requests beautifulsoup4

Python API:
    from extract import run_extraction
    result = run_extraction(stage="texts", limit=25)

Outputs:
    digesto_metadata.jsonl
    digesto_main_text.jsonl
    digesto_cache/catalog_pages/page_000001.json ...
"""

from __future__ import annotations

import argparse
import base64
import binascii
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
from urllib.parse import quote

import requests
import pipelines.initial.config as config
from pipelines.initial.digesto_document import EXTRACTOR_VERSION, extract_document

BASE = "https://digesto.asamblea.gob.ni"
SEARCH = f"{BASE}/consultas/normas/"
API = f"{BASE}/consultas/util/ws/proxy.php"
PAGE_SIZE = 100
LOG = logging.getLogger("digesto")


@dataclass(frozen=True)
class ExtractionResult:
    """Paths for downstream loading and counts from this invocation only.

    catalog_complete is None when the catalog stage was not run. text_path is
    None if no text snapshot exists. accepted counts fetches, not corpus size.
    """

    metadata_path: Path
    text_path: Path | None
    manifest_path: Path | None
    catalog_complete: bool | None
    fetched: int = 0
    accepted: int = 0
    rejected: int = 0
    preserved: int = 0


def _validate_options(*, delay, retries, limit=None, max_pages=None, ids=None,
                      refresh_texts=False, retry_failed=False) -> None:
    """Reject invalid limits, request settings, IDs, or conflicting extraction modes."""
    if not isinstance(delay, (int, float)) or not math.isfinite(delay) or delay < 1:
        raise ValueError("delay must be a finite value >= 1.0 second")
    if type(retries) is not int or not 0 <= retries <= 5:
        raise ValueError("retries must be an integer between 0 and 5")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer or None for all records")
    if max_pages is not None and (type(max_pages) is not int or max_pages < 1):
        raise ValueError("max_pages must be a positive integer or None")
    if refresh_texts and retry_failed:
        raise ValueError("refresh_texts and retry_failed are mutually exclusive")
    if isinstance(ids, (str, bytes)):
        raise ValueError("ids must be a sequence, for example ids=['3041']")
    selected_ids([str(identifier) for identifier in ids] if ids is not None else None)


def run_extraction(
    *, stage: str = "all", metadata_path: str | Path | None = None,
    text_path: str | Path | None = None, cache_dir: str | Path | None = None,
    raw_dir: str | Path | None = None, limit: int | None = config.TEXT_BATCH_SIZE,
    ids: list[str | int] | tuple[str | int, ...] | None = None,
    refresh_catalog: bool = False, refresh_texts: bool = False,
    retry_failed: bool = False, max_pages: int | None = None,
    delay: float | None = None, retries: int | None = None,
) -> ExtractionResult:
    """Run V1 extraction from Python without parsing CLI arguments.

    stage='all' runs catalog then texts; 'catalog' or 'texts' runs one stage.
    limit=None fetches all eligible texts. Paths default to config.py and may
    be overridden per call. A limited catalog feeds its separate partial file
    into the text stage; it never substitutes an older complete catalog.

    Only the main-text endpoint is used. Failures propagate to the caller;
    completed text checkpoints remain recoverable. Importing extract does not
    start extraction, configure logging, or connect to a database.
    """
    if stage not in ("all", "catalog", "texts"):
        raise ValueError("stage must be 'all', 'catalog' or 'texts'")
    if stage == "texts" and (max_pages is not None or refresh_catalog):
        raise ValueError("max_pages and refresh_catalog require a catalog stage")
    if stage == "catalog" and (ids is not None or refresh_texts or retry_failed):
        raise ValueError("ids, refresh_texts and retry_failed require a text stage")
    delay = config.REQUEST_DELAY_SECONDS if delay is None else delay
    retries = config.REQUEST_RETRIES if retries is None else retries
    _validate_options(delay=delay, retries=retries, limit=limit, max_pages=max_pages,
                      ids=ids, refresh_texts=refresh_texts, retry_failed=retry_failed)
    metadata = Path(config.METADATA_PATH if metadata_path is None else metadata_path).expanduser().resolve()
    output = Path(config.TEXT_PATH if text_path is None else text_path).expanduser().resolve()
    cache = Path(config.CACHE_DIR if cache_dir is None else cache_dir).expanduser().resolve()
    raw = Path(config.RAW_RESPONSE_DIR if raw_dir is None else raw_dir).expanduser().resolve()
    # Validate destinations before the catalog can write anything.
    if output in (metadata, partial_path(metadata), metadata.with_suffix(".manifest.json"),
                  partial_path(metadata).with_suffix(".manifest.json")):
        raise ValueError("Metadata/catalog artifacts and text output must be different files")
    if cache in (metadata, output) or raw in (metadata, output):
        raise ValueError("Cache and raw directories must be different from output files")
    for file_path in (metadata, output):
        if file_path.is_dir():
            raise ValueError(f"Expected an output file path, got a directory: {file_path}")
    for directory in (cache, raw):
        if directory.exists() and not directory.is_dir():
            raise ValueError(f"Expected a cache/raw directory, got a file: {directory}")
    manifest = None
    if stage in ("all", "catalog"):
        manifest = catalog(argparse.Namespace(output=metadata, cache_dir=cache,
                           max_pages=max_pages, refresh=refresh_catalog, delay=delay, retries=retries))
        metadata = Path(manifest["output"])
    counts = {}
    if stage in ("all", "texts"):
        counts = texts(argparse.Namespace(metadata=metadata, output=output, raw_dir=raw,
                       limit=limit, all=limit is None, ids=[str(identifier) for identifier in ids] if ids is not None else None,
                       refresh=refresh_texts, retry_failed=retry_failed, delay=delay, retries=retries))
    return ExtractionResult(
        metadata_path=metadata,
        text_path=output if stage != "catalog" and output.is_file() else None,
        manifest_path=metadata.with_suffix(".manifest.json") if manifest is not None else None,
        catalog_complete=manifest["complete"] if manifest is not None else None,
        **counts,
    )


def utc_now() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def decode_iunpid(encoded: str) -> str:
    """Digesto uses Base64-encoded numeric IDs (e.g. Njc5Mw== -> 6793)."""
    try:
        decoded = base64.b64decode(encoded, validate=True).decode("ascii")
        return decoded if decoded.isdigit() else ""
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return ""


def is_real_record(row: object) -> bool:
    """Check whether a catalog row contains a positive Base64-encoded numeric ID."""
    if not isinstance(row, dict):
        return False
    code = row.get("iunpid")
    decoded = decode_iunpid(code.strip()) if isinstance(code, str) else ""
    return bool(decoded) and int(decoded) > 0


def source_url(identifier: str) -> str:
    """Build the public instrument URL from an encoded source ID."""
    return f"{BASE}/consultas/normas/shownorms.php?idnorm={quote(identifier, safe='')}"


def jsonl_rows(path: Path):
    """Yield JSON objects from nonblank lines, reporting malformed input with line numbers."""
    with path.open("r", encoding="utf-8-sig") as fh:
        for number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSON in {path}, line {number}: {err}") from err
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object in {path}, line {number}")
            yield row


def atomic_json(path: Path, data: object) -> None:
    """Write UTF-8 JSON through a flushed temporary file, then replace the destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp, path)


def atomic_jsonl(path: Path, rows) -> None:
    """Publish JSONL rows atomically and remove any remaining temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        with temp.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def output_lock(path: Path):
    """Prevent two collectors from publishing the same output concurrently."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError(f"Output is locked: {lock}. If a process crashed, remove its stale lock after checking it has stopped.") from error
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(f"pid={os.getpid()} started={utc_now()}\n")
        yield
    finally:
        lock.unlink(missing_ok=True)


def is_access_challenge(data: bytes) -> bool:
    """Look for known access or security challenge markers near the response start."""
    preview = data[:2500].decode("utf-8", errors="ignore").lower()
    return any(token in preview for token in (
        "cf-chl-", "just a moment", "attention required!", "access denied",
    ))


def html_document_text(body: bytes) -> str:
    """Compatibility helper: return only accepted main-document text."""
    if is_access_challenge(body):
        raise RuntimeError("An access/security page was returned instead of legal text. Stop and inspect site access.")
    return extract_document(body).text


class DigestoClient:
    def __init__(self, delay: float, retries: int = 2):
        """Create an HTTP session with request spacing and transient retry settings."""
        self.delay = delay
        self.retries = retries
        self.last_request: float | None = None
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "DigestoDataResearch/1.0",
            "Accept": "application/json, text/html, */*",
            "Accept-Language": "es-NI,es;q=0.9",
        })

    def _throttle(self) -> None:
        """Wait until the configured interval since the previous request has elapsed."""
        if self.last_request is not None:
            wait = self.delay - (time.monotonic() - self.last_request)
            if wait > 0:
                time.sleep(wait)

    def _request(self, method: str, url: str, **kwargs) -> requests.Response:
        """Send a throttled request, retry transient failures, and stop on access restrictions."""
        for attempt in range(self.retries + 1):
            self._throttle()
            try:
                try:
                    response = self.session.request(method, url, timeout=60, **kwargs)
                finally:
                    self.last_request = time.monotonic()
            except (requests.ConnectionError, requests.Timeout):
                if attempt >= self.retries:
                    raise
                LOG.warning("Transient connection failure; retry %d/%d", attempt + 1, self.retries)
            else:
                if response.status_code in (403, 429):
                    raise RuntimeError(
                        f"Digesto returned HTTP {response.status_code}. "
                        "Stop this run and respect the server's access/rate restrictions.")
                if is_access_challenge(response.content):
                    raise RuntimeError("Digesto returned an access/security page. Stopping.")
                if response.status_code not in (500, 502, 503, 504) or attempt >= self.retries:
                    response.raise_for_status()
                    return response
                LOG.warning("HTTP %d; retry %d/%d", response.status_code, attempt + 1, self.retries)
            time.sleep(min(2 ** attempt, 30))
        raise RuntimeError("Request retries exhausted")

    def initialize(self) -> None:
        """Visit the search page to initialize the session before API requests."""
        response = self._request("GET", SEARCH)
        LOG.info("Session initialized (GET %s)", response.status_code)

    def post(self, data: dict[str, str]) -> requests.Response:
        """Submit an API form with the site's expected AJAX headers."""
        return self._request("POST", API, data=data, headers={
            "X-Requested-With": "XMLHttpRequest",
            "Origin": BASE,
            "Referer": SEARCH,
        })

    def list_page(self, page: int) -> tuple[list[dict], int | None]:
        """Fetch one unfiltered catalog page and its reported total, when available."""
        form = {
            "hddQueryType": "getJuridicNorms",
            "slcRange": "0", "slcCategory": "0",
            "MateriaPrin": "0", "MateriaRel": "0",
            "txtDatePublishFrom": "", "txtDatePublishTo": "",
            "infoNorm": "", "txtContentSearch": "",
            "isavancedsearch": "true",
            "hddPageSize": str(PAGE_SIZE),
            "hddCurrentPage": str(page),
            "hddFieldSort": "",
        }
        response = self.post(form)
        try:
            result = response.json()  # Content-Type may still say text/html.
            if isinstance(result, str):
                result = json.loads(result)
        except ValueError as err:
            raise RuntimeError(f"Page {page}: expected JSON, got {response.text[:250]!r}") from err
        if not isinstance(result, dict) or not isinstance(result.get("iunps"), list):
            raise RuntimeError(f"Page {page}: unrecognized listing structure")
        items = result["iunps"]
        total = None
        for item in items:
            if isinstance(item, dict):
                raw_total = item.get("totalRegistros")
                if str(raw_total or "").isdigit():
                    total = int(raw_total)
                    break
        return items, total

    def main_text(self, iunpid: str) -> requests.Response:
        """Fetch the main document response for an encoded instrument ID."""
        return self.post({"hddQueryType": "getNormaHtmlAccordion", "iunp": iunpid})


def partial_path(path: Path) -> Path:
    """Derive a separate output path for a partial or unverified catalog."""
    return path.with_name(path.stem + ".partial" + path.suffix)


def catalog(args: argparse.Namespace) -> dict:
    """Collect catalog metadata while locking both its output and shared cache."""
    with output_lock(args.output), output_lock(args.cache_dir / "catalog"):
        return _catalog(args)


def _catalog(args: argparse.Namespace) -> dict:
    """Resume or refresh catalog pages, reconcile completeness, and return the manifest.

    Partial or unverified results are written separately so they cannot replace
    a complete catalog. An unbounded run raises if reconciliation fails.
    """
    cache_state = args.cache_dir / "catalog_cache.json"
    active_directory = "catalog_pages"
    if cache_state.is_file():
        active_directory = json.loads(cache_state.read_text(encoding="utf-8"))["directory"]
        if active_directory not in ("catalog_pages", "catalog_pages_staging"):
            raise ValueError(f"Invalid catalog cache pointer: {cache_state}")
    directory = active_directory
    if args.refresh and args.max_pages is None:
        directory = "catalog_pages_staging" if active_directory == "catalog_pages" else "catalog_pages"
    cache = args.cache_dir / directory
    if args.max_pages is not None and args.refresh:
        cache = args.cache_dir / "catalog_sample_pages"
    cache.mkdir(parents=True, exist_ok=True)
    client: DigestoClient | None = None
    page = 1
    signatures: set[tuple[str, ...]] = set()
    rows: dict[str, dict] = {}
    totals: set[int] = set()
    observations = []
    occurrences = duplicates = invalid = conflicts = 0
    terminal = False

    while args.max_pages is None or page <= args.max_pages:
        page_file = cache / f"page_{page:06d}.json"
        if page_file.is_file() and not args.refresh:
            saved = json.loads(page_file.read_text(encoding="utf-8"))
            LOG.info("Page %d: using checkpoint", page)
        else:
            if client is None:
                client = DigestoClient(args.delay, retries=getattr(args, "retries", 2))
                client.initialize()
            items, total = client.list_page(page)
            saved = {"page": page, "fetched_at_utc": utc_now(), "total_reported": total, "items": items}
            atomic_json(page_file, saved)
            LOG.info("Page %d: downloaded %d response rows", page, len(items))
        items = saved.get("items")
        if not isinstance(items, list) or saved.get("page") != page:
            raise ValueError(f"Invalid cached API page: {page_file}")
        if saved.get("total_reported") is not None:
            totals.add(int(saved["total_reported"]))
        observations.append({"page": page, "fetched_at_utc": saved["fetched_at_utc"]})
        real_ids = tuple(str(row["iunpid"]).strip() for row in items if is_real_record(row))
        if real_ids and real_ids in signatures:
            raise RuntimeError(f"Repeated result page at {page}; pagination may be looping. Existing output unchanged.")
        signatures.add(real_ids)
        for item in items:
            if not is_real_record(item):
                code = item.get("iunpid", "") if isinstance(item, dict) else ""
                summary_row = isinstance(code, str) and decode_iunpid(code.strip()) == "0"
                if not summary_row:
                    invalid += 1
                continue
            occurrences += 1
            identifier = item["iunpid"].strip()
            if identifier in rows:
                duplicates += 1
                conflicts += rows[identifier]["raw_metadata"] != item
                continue
            url = source_url(identifier)
            rows[identifier] = {**item, "iunpid": identifier,
                                "numeric_id": decode_iunpid(identifier), "url": url,
                                "source_url": url, "raw_metadata": item,
                                "metadata_fetched_at": saved["fetched_at_utc"]}
        if not real_ids:
            terminal = True
            break
        page += 1

    # Reconcile source row occurrences separately from deduplicated IDs.
    reconciled = len(totals) == 1 and occurrences == next(iter(totals))
    complete = bool(rows) and terminal and reconciled and not invalid and not conflicts and args.max_pages is None
    destination = args.output if complete else partial_path(args.output)
    if complete and args.refresh:
        # Promote the fully reconciled cache before publishing its export. A crash
        # during export can then resume from this cache, never a mixture of pages.
        atomic_json(cache_state, {"directory": directory, "promoted_at_utc": utc_now()})
    atomic_jsonl(destination, rows.values())
    manifest = {"complete": complete, "terminal_page_reached": terminal,
                "listing_rows_reconciled": reconciled, "source_completeness": "not independently verified",
                "reported_totals": sorted(totals), "row_occurrences": occurrences,
                "unique_ids": len(rows), "duplicate_occurrences": duplicates,
                "conflicting_duplicates": conflicts, "invalid_rows": invalid,
                "pages": observations, "generated_at_utc": utc_now(),
                "extractor_version": EXTRACTOR_VERSION,
                "query": {"operation": "getJuridicNorms", "page_size": PAGE_SIZE,
                          "category": "all", "rank": "all", "sort": "source default"},
                "output": str(destination.resolve()),
                "output_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
    atomic_json(destination.with_suffix(".manifest.json"), manifest)
    LOG.info("Catalog exported: %s | %d unique IDs | %d duplicate occurrences",
             destination.resolve(), len(rows), duplicates)
    if not complete:
        LOG.warning("Partial/unverified catalog written separately. Existing canonical output unchanged.")
        if args.max_pages is None:
            raise RuntimeError("Catalog reconciliation failed; inspect the partial output and manifest")
    return manifest


def selected_ids(values: list[str] | None) -> set[str] | None:
    """Normalize numeric or encoded IDs to a validated set, or None for no filter."""
    if not values:
        return None
    identifiers = set()
    for value in values:
        value = value.strip()
        identifier = base64.b64encode(value.encode("ascii")).decode("ascii") if value.isascii() and value.isdigit() else value
        if not is_real_record({"iunpid": identifier}):
            raise ValueError(f"Invalid instrument ID: {value!r}")
        identifiers.add(identifier)
    return identifiers


def save_raw_response(directory: Path, raw: bytes) -> tuple[str, str]:
    """Raw evidence files, not legal versions or a source-artifacts database."""
    digest = hashlib.sha256(raw).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.html"
    if not path.exists():
        temp = path.with_suffix(f".{os.getpid()}.tmp")
        with temp.open("wb") as file:
            file.write(raw)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp, path)
    return str(path.resolve()), digest


def texts(args: argparse.Namespace) -> dict[str, int]:
    """Check input paths and extract current text while holding the output lock."""
    if not args.metadata.is_file():
        raise FileNotFoundError(f"Missing metadata catalog: {args.metadata}. Run 'catalog' first.")
    if args.metadata.resolve() == args.output.resolve():
        raise ValueError("Metadata and text output must be different files")
    with output_lock(args.output):
        return _texts(args)


def _texts(args: argparse.Namespace) -> dict[str, int]:
    """Resume selected text extractions and return counts for this invocation.

    Recover pending records, save raw responses, and publish one current row
    per instrument. A failed refresh preserves previously accepted content.
    """
    # One current row per ID. Pending files contain only unpublished current rows,
    # allowing recovery after interruption without an attempt/version history.
    current = {}
    if args.output.is_file():
        for row in jsonl_rows(args.output):
            if isinstance(row.get("iunpid"), str):
                current[row["iunpid"]] = row
    pending = args.output.with_name(args.output.name + ".pending")
    pending_files = list(pending.glob("*.json")) if pending.is_dir() else []
    for path in sorted(pending_files):
        row = json.loads(path.read_text(encoding="utf-8"))
        if not is_real_record(row):
            raise ValueError(f"Invalid current-record checkpoint: {path}")
        current[row["iunpid"]] = row
    requested = selected_ids(getattr(args, "ids", None))
    retry_failed = getattr(args, "retry_failed", False)
    refresh = getattr(args, "refresh", False)
    raw_dir = args.raw_dir if args.raw_dir is not None else Path(config.RAW_RESPONSE_DIR)
    seen_catalog = set()
    count = successes = rejected = preserved = 0
    client: DigestoClient | None = None
    try:
        for record in jsonl_rows(args.metadata):
            if not is_real_record(record):
                continue
            identifier = record["iunpid"].strip()
            if identifier in seen_catalog:
                continue
            seen_catalog.add(identifier)
            if requested is not None and identifier not in requested:
                continue
            prior = current.get(identifier)
            if retry_failed:
                eligible = prior is not None and prior.get("status") != "ok"
            elif refresh:
                eligible = True
            else:
                eligible = prior is None
            if not eligible:
                continue
            if not args.all and count >= args.limit:
                break
            if client is None:
                client = DigestoClient(args.delay, retries=getattr(args, "retries", 2))
                client.initialize()
            response = None
            raw = b""
            try:
                response = client.main_text(identifier)
                raw = response.content
                extracted = extract_document(raw, content_type=response.headers.get("Content-Type", ""),
                                             expected_title=str(record.get("titulo") or ""),
                                             document_url=source_url(identifier))
                status, reason = extracted.status, extracted.reason
                text, nodes, encoding = extracted.text, extracted.nodes, extracted.encoding
            except requests.RequestException as error:
                response = error.response
                raw = response.content if response is not None else b""
                status, reason = "http_error", str(error)
                text, nodes, encoding = "", [], None
            raw_file = raw_digest = None
            if response is not None:
                raw_file, raw_digest = save_raw_response(raw_dir, raw)
            result = {"iunpid": identifier, "numeric_id": decode_iunpid(identifier),
                      "source_url": source_url(identifier), "fetched_at_utc": utc_now(),
                      "status": status, "status_reason": reason,
                      "main_response_bytes": len(raw), "text_chars": len(text),
                      "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None,
                      "text": text, "document_nodes": nodes, "encoding": encoding,
                      "extractor_version": EXTRACTOR_VERSION,
                      "http_status": response.status_code if response is not None else None,
                      "request": {"url": API, "method": "POST", "operation": "getNormaHtmlAccordion", "iunp": identifier}}
            if raw_file:
                result.update(raw_html_file=raw_file, raw_sha256=raw_digest,
                              content_type=response.headers.get("Content-Type", ""))
            count += 1
            successes += status == "ok"
            rejected += status != "ok"
            if prior is not None and prior.get("status") == "ok" and status != "ok":
                preserved += 1
                LOG.warning("%s | refresh %s: %s. Keeping previous usable text. Raw response: %s",
                            result["numeric_id"], status, reason, raw_file)
                continue
            pending.mkdir(parents=True, exist_ok=True)
            path = pending / f"{result['numeric_id']}.json"
            atomic_json(path, result)
            pending_files.append(path)
            current[identifier] = result
            LOG.info("%d | %s | %s | %d chars | %s", count, result["numeric_id"],
                     status, len(text), str(record.get("titulo", ""))[:65])
    finally:
        # Even a later block/error publishes already completed records atomically.
        # If publication itself fails, pending checkpoints are left for recovery.
        if pending_files:
            atomic_jsonl(args.output, current.values())
            for path in set(pending_files):
                path.unlink(missing_ok=True)
            if not any(pending.iterdir()):
                pending.rmdir()
    if requested is not None and count < args.limit:
        unknown = requested - seen_catalog
        if unknown:
            LOG.warning("Requested IDs not found in metadata: %s", ", ".join(sorted(unknown)))
    LOG.info("Batch completed: %d fetched | %d accepted | %d empty/rejected/failed", count, successes, rejected)
    LOG.info("Current text output: %s", args.output.resolve())
    return {"fetched": count, "accepted": successes, "rejected": rejected, "preserved": preserved}


def main(argv: list[str] | None = None) -> ExtractionResult:
    """Parse CLI options, configure logging, and run the selected extraction stage."""
    cli = argparse.ArgumentParser(description="Digesto: all metadata + progressive main legal text")
    commands = cli.add_subparsers(dest="command", required=True)
    cat = commands.add_parser("catalog", help="Download all legal instrument metadata; checkpoint per page")
    cat.add_argument("--output", type=Path, default=Path(config.METADATA_PATH))
    cat.add_argument("--cache-dir", type=Path, default=Path(config.CACHE_DIR))
    cat.add_argument("--refresh", action="store_true", help="Refetch listing pages (rebuild catalog)")
    cat.add_argument("--max-pages", type=int, default=None, help="Optional short listing test")
    cat.add_argument("--delay", type=float, default=config.REQUEST_DELAY_SECONDS)
    cat.add_argument("--retries", type=int, default=config.REQUEST_RETRIES, help="Retries for connection failures and transient 5xx responses")

    doc = commands.add_parser("texts", help="Extract main legal text for the next unattempted records")
    doc.add_argument("--metadata", type=Path, default=Path(config.METADATA_PATH))
    doc.add_argument("--output", type=Path, default=Path(config.TEXT_PATH))
    selection = doc.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int, default=config.TEXT_BATCH_SIZE, help="Documents to fetch in this batch")
    selection.add_argument("--all", action="store_true", help="Process all eligible catalog records")
    mode = doc.add_mutually_exclusive_group()
    mode.add_argument("--retry-failed", action="store_true", help="Reattempt only previous empty, rejected or failed records")
    mode.add_argument("--refresh", action="store_true", help="Refetch selected IDs, including successful records")
    doc.add_argument("--ids", nargs="+", help="Restrict to numeric or encoded instrument IDs")
    doc.add_argument("--raw-dir", type=Path, default=Path(config.RAW_RESPONSE_DIR), help="Raw response backup directory (default: config.RAW_RESPONSE_DIR)")
    doc.add_argument("--delay", type=float, default=config.REQUEST_DELAY_SECONDS)
    doc.add_argument("--retries", type=int, default=config.REQUEST_RETRIES, help="Retries for connection failures and transient 5xx responses")

    args = cli.parse_args(argv)
    try:
        _validate_options(delay=args.delay, retries=args.retries,
                          max_pages=args.max_pages if args.command == "catalog" else None,
                          limit=(None if args.all else args.limit) if args.command == "texts" else None,
                          ids=args.ids if args.command == "texts" else None)
    except ValueError as error:
        cli.error(str(error))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.command == "catalog":
        return run_extraction(stage="catalog", metadata_path=args.output, cache_dir=args.cache_dir,
                              max_pages=args.max_pages, refresh_catalog=args.refresh,
                              delay=args.delay, retries=args.retries)
    return run_extraction(stage="texts", metadata_path=args.metadata, text_path=args.output,
                          raw_dir=args.raw_dir, limit=None if args.all else args.limit,
                          refresh_texts=args.refresh, retry_failed=args.retry_failed,
                          ids=args.ids, delay=args.delay, retries=args.retries)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, requests.RequestException, RuntimeError) as exc:
        LOG.error("Extraction stopped: %s", exc)
        sys.exit(1)
