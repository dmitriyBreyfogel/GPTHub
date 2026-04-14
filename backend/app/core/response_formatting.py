from __future__ import annotations

TECHNICAL_FORMATTING_GUIDANCE = "\n".join(
    [
        "Response formatting for technical content:",
        "- Use GitHub-flavored Markdown when it improves readability.",
        "- When you write source code, always wrap it in fenced code blocks with an explicit language tag when the language is known.",
        "- Keep prose explanations outside the code fence unless the user explicitly asks for inline comments inside the code.",
        "- Use inline backticks for commands, file paths, environment variables, identifiers and short snippets.",
        "- Use LaTeX for formulas. Use inline math with `$...$` inside a sentence.",
        "- Use display math with `$$...$$` for standalone formulas, derivations, matrices, systems of equations and multi-line transformations.",
        "- Do not place formulas inside code fences unless the user explicitly asks for raw LaTeX source.",
        "- When an answer mixes prose, formulas and code, separate them cleanly with short lead-in lines and blank lines.",
    ]
)


def append_technical_formatting_guidance(system_prompt: str) -> str:
    normalized_prompt = (system_prompt or "").strip()
    if not normalized_prompt:
        return TECHNICAL_FORMATTING_GUIDANCE
    if TECHNICAL_FORMATTING_GUIDANCE in normalized_prompt:
        return normalized_prompt
    return f"{normalized_prompt}\n\n{TECHNICAL_FORMATTING_GUIDANCE}"
