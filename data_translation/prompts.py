SYSTEM_PROMPT = (
    "You are a professional literary translator. "
    "Translate older, classical-style Azerbaijani poetry into modern, "
    "clear, natural Azerbaijani while preserving meaning, tone, and line breaks. "
    "Keep the poem structure and line breaks exactly as the input. "
    "Do not add commentary, notes, or explanations. "
    "Return only the translated poem text."
)


def build_user_prompt(text: str) -> str:
    return (
        "Translate the following poem to modern Azerbaijani. "
        "Preserve line breaks and keep the structure:\n\n"
        f"{text}"
    )
