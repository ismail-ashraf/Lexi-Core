import pymupdf as fitz


def inspect_results(file_path):

    pdf = fitz.open(file_path)

    try:
        total_pages = pdf.page_count
        total_characters = 0
        total_words = 0
        pages_with_text = 0

        for page in pdf:

            text = page.get_text()

            if text:

                cleaned_text = (
                    text
                    .replace(" ", "")
                    .replace("\n", "")
                    .replace("\r", "")
                )

                if cleaned_text:
                    total_characters += len(cleaned_text)
                    pages_with_text += 1

            page_words = page.get_text("words")

            if page_words:
                total_words += len(page_words)

        return {
            "total_pages": total_pages,
            "pages_with_text": pages_with_text,
            "total_words": total_words,
            "total_characters": total_characters,
        }

    finally:
        pdf.close()


def detect_columns(page, tolerance_ratio=0.05):

    if page is None:
        raise ValueError("Page is Empty")

    blocks = page.get_text("blocks")

    if not blocks:
        return {
            "column_count": 0,
            "columns": [],
        }

    x_positions = []

    for block in blocks:

        x0, y0, x1, y1 = block[:4]

        x_positions.append(x0)

    page_width = page.rect.width
    tolerance = page_width * tolerance_ratio

    x_positions.sort()

    columns = []

    current_group = [x_positions[0]]

    for i in range(len(x_positions) - 1):

        current_x = x_positions[i]
        next_x = x_positions[i + 1]

        difference = next_x - current_x

        if difference <= tolerance:

            current_group.append(next_x)

        else:

            columns.append(current_group)
            current_group = [next_x]

    columns.append(current_group)

    return {
        "column_count": len(columns),
        "columns": columns,
    }


def reconstruct_reading_order(
    page,
    columns,
    full_width_ratio=0.8,
):

    blocks = page.get_text("blocks")

    if not blocks:
        return []

    if not columns:
        return []

    page_width = page.rect.width

    column_centers = [
        sum(column) / len(column)
        for column in columns
    ]

    classified_blocks = []

    for block in blocks:

        x0, y0, x1, y1, text = block[:5]

        if not text.strip():
            continue

        block_width = x1 - x0

        is_full_width = (
            block_width >= page_width * full_width_ratio
        )

        if is_full_width:

            classified_blocks.append({
                "text": text,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "column": -1,
                "full_width": True,
            })

            continue

        column_index = min(
            range(len(column_centers)),
            key=lambda i: abs(x0 - column_centers[i]),
        )

        classified_blocks.append({
            "text": text,
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "column": column_index,
            "full_width": False,
        })

    normal_blocks = [
        block
        for block in classified_blocks
        if not block["full_width"]
    ]

    normal_blocks.sort(
        key=lambda block: (
            block["column"],
            block["y0"],
        )
    )

    full_width_blocks = [
        block
        for block in classified_blocks
        if block["full_width"]
    ]

    full_width_blocks.sort(
        key=lambda block: block["y0"]
    )

    ordered_blocks = []

    normal_index = 0
    full_width_index = 0

    while (
        normal_index < len(normal_blocks)
        or full_width_index < len(full_width_blocks)
    ):

        if full_width_index >= len(full_width_blocks):

            ordered_blocks.extend(
                normal_blocks[normal_index:]
            )

            break

        if normal_index >= len(normal_blocks):

            ordered_blocks.extend(
                full_width_blocks[full_width_index:]
            )

            break

        full_width_block = full_width_blocks[full_width_index]
        normal_block = normal_blocks[normal_index]

        if full_width_block["y0"] < normal_block["y0"]:

            ordered_blocks.append(full_width_block)
            full_width_index += 1

        else:

            ordered_blocks.append(normal_block)
            normal_index += 1

    return ordered_blocks


def reconstruct_table(page, y_tolerance=5):

    words = page.get_text("words")

    if not words:
        return []

    cells = []

    for word in words:

        x0, y0, x1, y1, text = word[:5]

        if not text.strip():
            continue

        cells.append({
            "text": text,
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
        })

    cells.sort(
        key=lambda cell: cell["y0"]
    )

    rows = []

    for cell in cells:

        added_to_row = False

        for row in rows:

            row_y = sum(
                item["y0"]
                for item in row
            ) / len(row)

            if abs(cell["y0"] - row_y) <= y_tolerance:

                row.append(cell)
                added_to_row = True
                break

        if not added_to_row:
            rows.append([cell])

    for row in rows:

        row.sort(
            key=lambda cell: cell["x0"]
        )

    table = []

    for row in rows:

        row_data = [
            cell["text"]
            for cell in row
        ]

        table.append(row_data)

    return table

def analyze_page(page):
    if page is None:
        raise ValueError("Page is Empty")

    data = page.get_text("dict")

    text_blocks = []
    image_blocks = []

    for block in data["blocks"]:
        if block["type"] == 0:
            lines = []

            for line in block["lines"]:
                spans = []

                for span in line["spans"]:
                    spans.append({
                        "text": span["text"],
                        "font": span["font"],
                        "size": span["size"],
                        "bbox": span["bbox"],
                    })

                lines.append(spans)

            text_blocks.append({
                "bbox": block["bbox"],
                "lines": lines,
            })


        elif block["type"] == 1:

            x0, y0, x1, y1 = block["bbox"]

            image_width = x1 - x0

            image_height = y1 - y0

            image_blocks.append({

                "bbox": block["bbox"],

                "width": block["width"],

                "height": block["height"],

                "area_ratio": (
                image_width * image_height
                ) / (page.rect.width * page.rect.height),
            })
        has_native_text = len(text_blocks) > 0
        has_images = len(image_blocks) > 0

        if has_native_text and has_images:
            page_type = "mixed"
        elif has_native_text:
            page_type = "native"
        elif has_images:
            page_type = "image_based"
        else:
            page_type = "empty"

    return {
        "page_number": page.number + 1,
        "page_size": {
            "width": page.rect.width,
            "height": page.rect.height,
        },
        "text_blocks": text_blocks,
        "image_blocks": image_blocks,
        "text_blocks_count": len(text_blocks),
        "image_blocks_count": len(image_blocks),
        "page_type": page_type,
    }