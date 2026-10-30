from pydantic_settings import BaseSettings,SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
env_file = BASE_DIR / '.env'

class Settings(BaseSettings):
    NER_ONNX_MODEL_PATH : str
    BERT_ONNX_MODEL_PATH : str
    MAX_LENGTH : int
    legal_ner_labels: list[str] = [
        "Act", "Address", "Court", "EffectiveDate", "PII_Ref",
        "Parties", "Percentage", "Price", "Principal", "Ratio",
        "Regulation", "RenewalTerm", "Rent", "Role", "Salary",
        "Shares", "TerminationDate", "Title"
    ]
    ner_threshold: float = 0.5

    DATABASE_URL : str

    model_config = SettingsConfigDict(
        env_file=env_file,
        env_file_encoding='utf-8',
        extra="ignore"
    )


settings = Settings()
