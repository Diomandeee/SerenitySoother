from typing import List
import re


def split_text(text):
    # Use regular expressions to split text based on numbering patterns
    lines = re.split(r"(\d+\.\d+\.\d+\s)", text)

    # Filter out empty and None lines
    lines = [line.strip() for line in lines if line is not None and line.strip()]

    return lines


def split_by_markdown_delimiters(s: str) -> List[str]:
    # Split by headers from h1 to h6, italic/bold delimiters, block quotes, list items, and colons
    patterns = [
        r"\#{1,6} ",  # Headers (e.g., # Header1, ## Header2, ...)
        r"\*{1,2}",  # Italic and bold
        r"\>",  # Block quotes
        r"\- ",  # List items
        r"\* ",  # List items
        r"\+ ",  # List items
        r"\d+\. ",  # Numbered list items
        r"\:",  # Colons
    ]

    combined_pattern = "|".join(patterns)

    # Split by the combined pattern and remove empty strings
    return [
        segment.strip() for segment in re.split(combined_pattern, s) if segment.strip()
    ]


def split_by_multiple_delimiters(s: str, delimiters: List[str] = None) -> List[str]:
    if delimiters is None:
        delimiters = [";", ",", "|"]
    delimiter_pattern = "|".join(map(re.escape, delimiters))
    return re.split(delimiter_pattern, s)


def split_by_consecutive_spaces(s: str) -> List[str]:
    return re.split(r"\s{2,}", s)


def split_by_capital_letters(s: str) -> List[str]:
    return re.findall(r"[A-Z][a-z]*", s)


def split_string_to_parts(raw: str, delimiter: str = "\n") -> List[str]:
    return raw.split(delimiter)
