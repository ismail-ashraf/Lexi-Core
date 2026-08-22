import os
import tempfile

import pymupdf as fitz
from unstructured.partition.pdf import partition_pdf


def parse_with_unstructured(page):

    if page is None:
        raise ValueError("Page is Empty")

    temp_path = None
    temp_pdf = None

    try:

        temp_pdf = fitz.open()

        temp_pdf.insert_pdf(
            page.parent,
            from_page=page.number,
            to_page=page.number,
        )

        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as temp_file:

            temp_path = temp_file.name

        temp_pdf.save(temp_path)

        elements = partition_pdf(
            filename=temp_path,
            strategy="hi_res",
        )

        tables = []
        texts = []

        for element in elements:

            metadata = {
                "page_number": page.number + 1,
            }

            if element.category == "Table":

                tables.append({
                    "text": element.text,
                    "metadata": metadata,
                })

            else:

                texts.append({
                    "text": element.text,
                    "metadata": metadata,
                })

        return {
            "tables": tables,
            "texts": texts,
        }

    finally:

        if temp_pdf is not None:
            temp_pdf.close()

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)