
from pathlib import Path

import pymupdf as fitz

from .ocr_parser import parse_with_ocr
from .pymupdf_parser import parse_with_pymupdf
from .pymupdf_tools import analyze_page


class DocumentParser:
    """Extract readable text from TXT and PDF documents."""

    @staticmethod
    def parse(storage_path: str) -> str:
        path = Path(storage_path)
        suffix = path.suffix.lower()

        if suffix == ".txt":
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="cp1252")

            if not text.strip():
                raise ValueError("Document contains no text")

            return text

        if suffix != ".pdf":
            raise ValueError(f"Unsupported document format: {suffix}")

        text_parts = []

        with fitz.open(str(path)) as pdf:
            if pdf.is_encrypted:
                raise ValueError("Encrypted PDF files are not supported")

            for page in pdf:
                page_blocks = []

                # 1. Extract native text using PyMuPDF
                native_result = parse_with_pymupdf(page)

                for block in native_result["blocks"]:
                    block_text = block["text"].strip()

                    if not block_text:
                        continue

                    page_blocks.append({
                        "text": block_text,
                        "bbox": (
                            block["x0"],
                            block["y0"],
                            block["x1"],
                            block["y1"],
                        ),
                        "source": "native",
                    })

                # 2. Analyze the page to find image regions
                page_analysis = analyze_page(page)
                image_blocks = page_analysis["image_blocks"]

                # 3. Extract text from image regions using OCR
                if image_blocks:
                    for image_block in image_blocks:
                        clip = fitz.Rect(image_block["bbox"])

                        ocr_result = parse_with_ocr(
                            page,
                            clip=clip,
                        )

                        for row in ocr_result["rows"]:
                            if not row["text"].strip():
                                continue

                            elements = row["elements"]

                            page_blocks.append({
                                "text": row["text"],
                                "bbox": (
                                    min(e["bbox"][0] for e in elements),
                                    min(e["bbox"][1] for e in elements),
                                    max(e["bbox"][2] for e in elements),
                                    max(e["bbox"][3] for e in elements),
                                ),
                                "source": "ocr",
                            })

                # 4. Fallback: OCR the entire page if no content was found
                elif not page_blocks:
                    ocr_result = parse_with_ocr(page)

                    for row in ocr_result["rows"]:
                        if not row["text"].strip():
                            continue

                        elements = row["elements"]

                        page_blocks.append({
                            "text": row["text"],
                            "bbox": (
                                min(e["bbox"][0] for e in elements),
                                min(e["bbox"][1] for e in elements),
                                max(e["bbox"][2] for e in elements),
                                max(e["bbox"][3] for e in elements),
                            ),
                            "source": "ocr",
                        })

                # 5. Sort extracted content by its page position
                page_blocks.sort(
                    key=lambda block: (
                        block["bbox"][1],
                        block["bbox"][0],
                    )
                )

                page_text = "\n".join(
                    block["text"] for block in page_blocks
                )

                if page_text.strip():
                    text_parts.append(page_text)

        text = "\n\n".join(text_parts)

        if not text.strip():
            raise ValueError(
                "No extractable text found in PDF. OCR may be required."
            )

        return text
