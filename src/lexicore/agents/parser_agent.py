from lexicore.parsing.pymupdf_tools import inspect_results , detect_columns
from lexicore.parsing.ocr_parser import parse_with_ocr
from lexicore.parsing.unstructured_parser import parse_with_unstructured
from lexicore.parsing.pymupdf_parser import parse_with_pymupdf
import pymupdf as fitz


def parse_document(file_path):

    inspection = inspect_results(file_path)

    pdf = fitz.open(file_path)

    results = []

    try:

        pages_with_text = inspection["pages_with_text"]

        for page in pdf:

            if pages_with_text == 0:

                parser = "ocr"
                result = parse_with_ocr(page)

            else:

                text = page.get_text().strip()

                if not text:

                    parser = "ocr"
                    result = parse_with_ocr(page)

                else:

                    column_result = detect_columns(page)

                    column_count = column_result["column_count"]

                    if column_count > 1:

                        parser = "unstructured"
                        result = parse_with_unstructured(page)

                    else:

                        parser = "pymupdf"
                        result = parse_with_pymupdf(page)

            results.append({
                "page_number": page.number + 1,
                "parser": parser,
                "result": result,
            })

        return {
            "inspection": inspection,
            "pages": results,
        }

    finally:

        pdf.close()