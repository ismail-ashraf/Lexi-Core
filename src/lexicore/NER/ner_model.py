from gliner import GLiNER
import time
from lexicore.configrations.config import settings

model_dir = settings.NER_MODEL_PATH


model = GLiNER.from_pretrained(
    model_dir,
    local_files_only=True,
)
if model is None:
    raise ValueError("Model not found")

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
entities = model.predict_entities(
    text,
    labels,
    threshold=0.5,
)
end = time.perf_counter()
elapsed = end - start
print(elapsed)

for entity in entities:
    print(
        f"{entity['text']:<35} "
        f"Label: {entity['label']:<18} "
        f"Score: {entity['score']:.4f}"
    )
