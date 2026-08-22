import pytesseract
from PIL import Image


def parse_with_ocr(page):

    if page is None:
        raise ValueError("Page is Empty")

    pix = page.get_pixmap(dpi=300)

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

    for i in range(len(data["text"])):

        text = data["text"][i].strip()

        if not text:
            continue

        x0 = data["left"][i]
        y0 = data["top"][i]

        width = data["width"][i]
        height = data["height"][i]

        x1 = x0 + width
        y1 = y0 + height

        elements.append({
            "text": text,
            "bbox": (x0, y0, x1, y1),
            "confidence": data["conf"][i],
            "page_number": page.number + 1,
        })

    return {
        "elements": elements,
    }