# Initial pipeline

![Initial pipeline](initial-pipeline.png)

[Editable Excalidraw](initial-pipeline.excalidraw) · [SVG](initial-pipeline.svg)

The arrows across the top show data dependencies. `pipelines/initial/main.py` chains extraction, transformation, quality checks, and loading. Run it from the project root after applying `database/schema.sql`:

```powershell
python -m pipelines.initial.main
```

This fetches all eligible texts. For a smaller first run, use `python -m pipelines.initial.main --limit 25`; the catalog is still collected in full and all catalog rows are loaded, including those whose text has not been attempted. Paths and credentials come from `pipelines/initial/config.py`. Importing the runner starts no work.

The runner stops before loading when the quality report is failed, incomplete, or has no recognized status. Warnings permit loading. It uses the exact input paths returned by extraction, including when the configured artifacts are in different directories. Embedding generation is not part of this runner.

The separate stage functions remain available:

```python
from pipelines.initial.config import OUTPUT_DIR
from pipelines.initial.extract import run_extraction
from pipelines.initial.transform import extract_data
from pipelines.initial.load import upload_to_postgres

# Fetch the catalog and the next 25 unattempted documents.
result = run_extraction(stage="all")

# Read both JSONL snapshots and write quality_report.json.
tables = extract_data(OUTPUT_DIR)

# Review the quality report before choosing to load.
# The database schema must already be applied.
# upload_to_postgres(tables["legal_instruments"], tables["document_nodes"])
```

Default artifacts live under `pipelines/initial/output/`. A limited catalog is published to a separate partial file. `extract_data()` reads configured filenames unless explicit input paths are passed. The initial runner requires a complete, reconciled catalog.

Extraction can resume from cached pages, current text rows, and pending checkpoints. A failed text refresh preserves previous accepted content. Quality measurement functions only report errors; the runner enforces the report status before calling the loader. Calling the loader directly bypasses that gate. The loader checks a subset of the contract and relies on SQL constraints for additional enforcement; it does not enforce every measured rule.

The current database loader is insert-only. A duplicate `(source, source_id)` or another transaction error rolls back the batch. The contract describes broader refresh and repeated-load policies that this loader does not implement.

The example node is taken from `output/node_example_3041/document.json`: `n000004` is a paragraph under article `n000003`, spanning characters `[129:736]`. The green data excerpt shows its transformed field names. Extraction status measures technical acceptance, not whether an instrument is legally in force.

Sources: `pipelines/initial/extract.py`, `digesto_document.py`, `transform.py`, `quality.py`, `load.py`, `config.py`, `data_contract.yaml`, and `database/schema.sql`. The diagram documents the code as inspected on November 15, 2026. No extraction or database loading was run to create it.

The PNG and SVG previews use an offline renderer because the standard renderer's remote Excalidraw dependency was unavailable. The `.excalidraw` file retains native editable elements.
