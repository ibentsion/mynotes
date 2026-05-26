"""Build data/words.txt from all text files in data/text/.

Keeps any whitespace-delimited token where:
  - every character is in the project charset (outputs/model/charset.json), AND
  - the token contains at least one Hebrew letter or digit
    (excludes pure-punctuation and pure-Latin noise)
"""
import json
import sys
import unicodedata
from pathlib import Path

TEXT_DIR = Path("data/text")
OUT_FILE = Path("data/words.txt")
CHARSET_FILE = Path("outputs/model/charset.json")

_HEBREW_RANGE = (ord("א"), ord("ת"))


def _has_hebrew_or_digit(token: str) -> bool:
    return any(
        _HEBREW_RANGE[0] <= ord(c) <= _HEBREW_RANGE[1] or c.isdigit() for c in token
    )


def main() -> None:
    txt_files = sorted(TEXT_DIR.glob("*.txt"))
    if not txt_files:
        sys.exit(f"No .txt files found in {TEXT_DIR}")
    if not CHARSET_FILE.exists():
        sys.exit(f"Charset file not found: {CHARSET_FILE} — run train-ctc first")

    charset = set(json.loads(CHARSET_FILE.read_text(encoding="utf-8")))

    words: set[str] = set()
    for path in txt_files:
        text = unicodedata.normalize("NFC", path.read_text(encoding="utf-8"))
        for token in text.split():
            if all(c in charset for c in token) and _has_hebrew_or_digit(token):
                words.add(token)

    OUT_FILE.write_text("\n".join(sorted(words)) + "\n", encoding="utf-8")
    print(f"{len(words)} unique tokens written to {OUT_FILE}")


if __name__ == "__main__":
    main()
