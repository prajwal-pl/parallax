from __future__ import annotations


def truncate_output(text: str, max_chars: int = 12_000) -> str:
    """
    Bound a tool output to roughly `max_chars`, keeping the head (what ran) and
    the tail (summary, errors, exit status) and noting how much was omitted.
    """
    if len(text) <= max_chars:
        return text

    keep = max_chars // 2
    omitted = len(text) - 2 * keep
    return (
        f"{text[:keep]}\n\n"
        f"... [{omitted} characters omitted. Narrow the command, search a smaller "
        f"path, or view a line range] ...\n\n"
        f"{text[-keep:]}"
    )
