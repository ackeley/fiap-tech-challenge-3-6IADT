import re

# regras simples (você pode expandir depois)
FORBIDDEN_PATTERNS = [
    r"\bmg\b", r"\bml\b", r"\bcomprimid", r"\bcápsul", r"\bdose\b",
    r"\bprescrev", r"\breceit", r"\btomar\b.*\b\d+\b", r"\badministre\b",
    r"\bposologia\b"
]

def safety_check(text: str) -> dict:
    hits = []
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            hits.append(pat)

    ok = len(hits) == 0
    return {"ok": ok, "hits": hits}

def apply_guardrail(answer: str) -> str:
    """
    Se detectar conteúdo de prescrição, substitui por resposta segura.
    """
    check = safety_check(answer)
    if check["ok"]:
        return answer

    return (
        "⚠️ Resposta bloqueada por regra de segurança.\n\n"
        "O assistente detectou conteúdo potencialmente prescritivo (ex.: dose/medicação).\n"
        "Por segurança, não é permitido prescrever ou indicar posologia.\n\n"
        "Sugestão segura:\n"
        "- Reforçar avaliação clínica e critérios de gravidade\n"
        "- Consultar protocolo interno e discutir com médico responsável\n"
        "- Registrar a decisão no prontuário\n"
    )
