from gliner import GLiNER
import time
from lexicore.configrations.config import settings

onnx_folder_path = settings.ONNX_FOLDER_PATH

model = GLiNER.from_pretrained(
    onnx_folder_path,
    load_onnx_model=True,
    onnx_model_file="model.onnx",
    local_files_only=True
)

print("Model type:", type(model))
print("Runtime:", getattr(model, "runtime", "Not exposed"))

text = """
This Employment Agreement is effective as of January 1, 2026,
between Acme Corporation and John Smith.
The Company shall pay the Employee an annual salary of $120,000.
The Agreement shall automatically renew for successive one-year terms
unless either party provides 30 days written notice of termination.
"""

labels = [
    "Act",
    "Address",
    "Court",
    "EffectiveDate",
    "PII_Ref",
    "Parties",
    "Percentage",
    "Price",
    "Principal",
    "Ratio",
    "Regulation",
    "RenewalTerm",
    "Rent",
    "Role",
    "Salary",
    "Shares",
    "TerminationDate",
    "Title",
]
start = time.perf_counter()
end = time.perf_counter()

elapsed = end - start
print(elapsed)
entities = model.predict_entities(text, labels, threshold=0.5)

for entity in entities:
    print(
        f"Text: {entity['text']} | "
        f"Label: {entity['label']} | "
        f"Score: {entity['score']:.2f}"
    )