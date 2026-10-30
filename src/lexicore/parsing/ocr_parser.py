
import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def parse_with_ocr(page, clip=None):
    if page is None:
        raise ValueError("Page is Empty")

    dpi = 300
    pix = page.get_pixmap(dpi=dpi, clip=clip)

    scale = 72 / dpi

    origin_x = pix.x * scale
    origin_y = pix.y * scale

    if pix.n == 1:
        mode = "L"
    elif pix.n == 3:
        mode = "RGB"
    elif pix.n == 4:
        mode = "RGBA"
    else:
        raise ValueError("Unsupported Pixmap Format")

    image = Image.frombytes(
        mode,
        (pix.width, pix.height),
        pix.samples,
    )

    data = pytesseract.image_to_data(
        image,
        lang="ara+eng",
        output_type=pytesseract.Output.DICT,
    )

    elements = []
    grouped_rows = {}

    for i in range(len(data["text"])):
        text = data["text"][i].strip()

        if not text:
            continue

        x0 = data["left"][i] * scale + origin_x
        y0 = data["top"][i] * scale + origin_y

        x1 = x0 + data["width"][i] * scale
        y1 = y0 + data["height"][i] * scale

        element = {
            "text": text,
            "bbox": (x0, y0, x1, y1),
            "center_y": (y0 + y1) / 2,
            "confidence": float(data["conf"][i]),
            "page_number": page.number + 1,
            "block_num": data["block_num"][i],
            "par_num": data["par_num"][i],
            "line_num": data["line_num"][i],
            "word_num": data["word_num"][i],
        }

        elements.append(element)

        row_key = (
            element["block_num"],
            element["par_num"],
            element["line_num"],
        )

        grouped_rows.setdefault(row_key, []).append(element)

    structured_rows = []

    for row in grouped_rows.values():
        row.sort(key=lambda element: element["bbox"][0])

        structured_rows.append({
            "text": " ".join(
                element["text"] for element in row
            ),
            "elements": row,
            "page_number": page.number + 1,
        })

    structured_rows.sort(
        key=lambda row: min(
            element["bbox"][1]
            for element in row["elements"]
        )
    )

    return {
        "elements": elements,
        "rows": structured_rows,
    }
