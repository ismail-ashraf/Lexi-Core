from .pymupdf_tools import (
    detect_columns,
    reconstruct_reading_order,
)


def parse_with_pymupdf(page):

    if page is None:
        raise ValueError("Page is Empty")

    column_result = detect_columns(page)

    columns = column_result["columns"]
    column_count = column_result["column_count"]

    ordered_blocks = reconstruct_reading_order(
        page=page,
        columns=columns,
    )

    return {
        "page_number": page.number + 1,
        "column_count": column_count,
        "blocks": ordered_blocks,
    }