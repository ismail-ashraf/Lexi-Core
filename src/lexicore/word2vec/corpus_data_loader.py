from pathlib import Path


def load_corpus(file_path: str | Path) -> list[str]:
    path = Path(file_path)

    with path.open("r", encoding="utf-8") as file:
        corpus = [
            line.strip()
            for line in file
            if line.strip()
        ]

    return corpus