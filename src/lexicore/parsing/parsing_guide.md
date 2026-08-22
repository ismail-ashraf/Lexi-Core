# LexiCore Parsing Layer Guide

## Overview

The `parsing` package is responsible for extracting structured content from PDF documents.

PDF files are not always represented in the same way. A document may contain:

* Native machine-readable text
* Scanned images
* Multiple-column layouts
* Tables
* Mixed content

Because of this, LexiCore does not process every PDF using a single parser.

Instead, the system provides multiple parsing tools and a separate **Parser Agent** that decides which parser should handle each page.

---

## Architecture

```text
PDF
 │
 ▼
Parser Agent
 │
 ├── Document Inspection
 │
 └── Page Detection
       │
       ├── No text
       │      ↓
       │     OCR
       │
       ├── Text + multiple columns
       │      ↓
       │     Unstructured
       │
       └── Text + simple layout
              ↓
            PyMuPDF
```

The important design principle is:

> The Agent is responsible for routing. The individual parsers are responsible for parsing.

This keeps the parsing tools independent from the routing logic.

---

# Package Structure

```text
lexicore/
│
├── agents/
│   └── parser_agent.py
│
└── parsing/
    ├── ocr_parser.py
    ├── parsing_guide.md
    ├── pymupdf_parser.py
    ├── pymupdf_tools.py
    ├── unstructured_parser.py
    └── __init__.py
```

---

# 1. `pymupdf_tools.py`

This module contains reusable low-level PDF analysis and reconstruction utilities built on top of PyMuPDF.

It does not decide which parser should be used.

## `inspect_results()`

Performs document-level inspection.

It calculates:

* Total number of pages
* Number of pages containing text
* Total number of extracted words
* Total number of extracted characters

Example result:

```python
{
    "total_pages": 10,
    "pages_with_text": 8,
    "total_words": 4200,
    "total_characters": 25000
}
```

The Parser Agent uses this information to understand the general structure of the document.

---

## `detect_columns()`

Detects possible text columns on a page using the X positions of extracted text blocks.

It returns:

```python
{
    "column_count": 2,
    "columns": [...]
}
```

The result is used by the Parser Agent to determine whether a page has a simple or multi-column layout.

---

## `reconstruct_reading_order()`

Attempts to reconstruct the logical reading order of text blocks.

This is important for layouts such as:

```text
Column 1        Column 2

Text            Text
Text            Text
Text            Text
```

Instead of simply following the raw PDF block order, the function groups blocks by columns and orders them according to their position.

It also handles blocks that span most of the page width, such as headings or full-width sections.

---

## `reconstruct_table()`

Attempts to reconstruct table-like structures from extracted words.

The process is approximately:

```text
Words
 ↓
Group by Y position
 ↓
Create rows
 ↓
Sort cells by X position
 ↓
Return table rows
```

Example:

```python
[
    ["Name", "Age", "Department"],
    ["Ahmed", "25", "Engineering"],
    ["Sara", "27", "Finance"]
]
```

This is a heuristic reconstruction utility. It should not be considered a general-purpose table detection algorithm.

---

# 2. `pymupdf_parser.py`

This is the parser used for pages with a relatively simple native text layout.

It receives a single PDF page:

```python
parse_with_pymupdf(page)
```

Its workflow is:

```text
Page
 ↓
detect_columns()
 ↓
reconstruct_reading_order()
 ↓
Structured page result
```

Example output:

```python
{
    "page_number": 1,
    "column_count": 1,
    "blocks": [...]
}
```

The parser works at the **page level**.

It does not reopen or iterate through the entire PDF.

---

# 3. `ocr_parser.py`

This parser handles pages that do not contain a usable text layer.

Its workflow is:

```text
PDF Page
 ↓
Render page to Pixmap
 ↓
Convert Pixmap → PIL Image
 ↓
Tesseract OCR
 ↓
Extract text + bounding boxes + confidence
```

Tesseract is configured for both Arabic and English:

```text
ara + eng
```

The parser returns OCR elements containing:

* Extracted text
* Bounding box
* Confidence
* Page number

Example:

```python
{
    "text": "Hello",
    "bbox": (100, 120, 180, 150),
    "confidence": "96.4",
    "page_number": 1
}
```

The bounding box is important because OCR is not only extracting text; it also preserves the approximate location of that text on the page.

---

# 4. `unstructured_parser.py`

This parser is used for pages where the layout is more complex, particularly multi-column or structurally rich pages.

The main library used is:

```python
unstructured
```

The Parser Agent passes it a single page.

Because `partition_pdf()` expects a PDF file rather than a PyMuPDF `Page` object, the adapter performs the following:

```text
PyMuPDF Page
 ↓
Create temporary PDF
 ↓
Copy only the selected page
 ↓
partition_pdf()
 ↓
Unstructured Elements
```

The resulting elements are separated into:

```python
{
    "tables": [...],
    "texts": [...]
}
```

This adapter hides the implementation detail that Unstructured requires a PDF file.

---

# 5. Parser Agent

The Parser Agent is located outside the `parsing` package:

```text
lexicore/agents/parser_agent.py
```

Its responsibility is **routing**, not low-level parsing.

The Agent first performs document-level inspection:

```python
inspection = inspect_results(file_path)
```

Then it evaluates pages individually.

## Routing logic

### Case 1 — Scanned document

If the document contains no pages with extractable text:

```text
pages_with_text == 0
```

the pages are routed to:

```text
OCR
```

---

### Case 2 — Individual scanned page

If the document contains text overall, but a specific page has no text:

```text
page.get_text() == ""
```

that page is routed to:

```text
OCR
```

This supports mixed PDFs containing both native and scanned pages.

---

### Case 3 — Multi-column page

If a page contains text and multiple columns are detected:

```text
column_count > 1
```

the page is routed to:

```text
Unstructured
```

---

### Case 4 — Simple native page

If the page contains text and does not appear to have multiple columns:

```text
column_count <= 1
```

the page is routed to:

```text
PyMuPDF
```

---

# Parser Interface

All three parsers follow the same page-level interface:

```python
parse_with_pymupdf(page)

parse_with_ocr(page)

parse_with_unstructured(page)
```

This is intentional.

The Parser Agent owns the PDF and determines which page should go to which parser.

The individual parser should not reopen and process the entire document.

This creates a clean separation:

```text
Agent
  │
  ├── Decision
  │
  └── Page
        │
        ├── PyMuPDF Parser
        ├── OCR Parser
        └── Unstructured Parser
```

---

# Why Multiple Parsers?

A single parser is not reliable for every PDF type.

For example:

| PDF characteristic            | Parser       |
| ----------------------------- | ------------ |
| Native text + simple layout   | PyMuPDF      |
| Scanned page                  | OCR          |
| Multi-column / complex layout | Unstructured |

This routing strategy allows LexiCore to use a specialized tool for each document structure instead of forcing every PDF through the same pipeline.

---

# Testing

The Parser Agent has routing tests covering the three main paths:

```text
Simple native PDF
    → PyMuPDF

Scanned PDF
    → OCR

Multi-column PDF
    → Unstructured
```

The tests focus on verifying the Agent's routing decisions rather than testing the internal implementation of external parsing libraries.

Run the parsing tests with:

```bash
pytest tests/test_parsing/test_parser_agent.py -v
```

Run the complete test suite with:

```bash
pytest -v
```

---

# Design Principle

The parsing layer follows a simple responsibility boundary:

```text
Parser Agent
    ↓
"Which tool should handle this page?"

Parser
    ↓
"How do I extract the content from this page?"
```

Keeping these responsibilities separate makes the system easier to test, extend, and maintain.

For example, a future parser can be added without rewriting the existing parsers:

```text
New Parser
    ↓
register it with the Agent
    ↓
add routing rule
    ↓
add routing tests
```

The current implementation is intended as the foundation for a more advanced document ingestion pipeline that can later add stronger layout detection, table extraction, metadata normalization, chunking, and downstream embedding.
