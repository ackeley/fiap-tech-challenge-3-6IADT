from langchain_core.prompts import PromptTemplate
from src.rag.retriever import get_retriever
from src.assistant.llm_local import load_llm_cpu
from src.safety.logger import log_event
from src.safety.validator import apply_guardrail, safety_check

PROMPT = PromptTemplate(
    input_variables=["context", "question", "patient"],
    template="""
Você é um assistente clínico do hospital (pt-BR).

REGRAS DE SEGURANÇA (OBRIGATÓRIAS):
- NÃO prescreva medicamentos, doses ou receitas.
- NÃO substitua decisão clínica.
- Sempre solicitar validação de um médico responsável.
- Se houver sinais de gravidade, orientar escalonamento imediato.

DADOS DO PACIENTE:
{patient}

TRECHOS DE PROTOCOLOS INTERNOS (use como base e cite as fontes):
{context}

PERGUNTA:
{question}

FORMATO DA RESPOSTA:
1) Conduta sugerida (passos objetivos)
2) Alertas
3) Exames a verificar
4) Validação humana
5) Fontes (listar nomes dos arquivos consultados)
"""
)

def format_context(docs):
    blocks = []
    sources = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "desconhecido")
        sources.append(src)
        blocks.append(f"[Trecho {i} | Fonte: {src}]\n{d.page_content}\n")
    return "\n".join(blocks), sorted(set(sources))

def ask(question: str, patient: dict):
    retriever = get_retriever(k=3)
    docs = retriever.invoke(question)

    context, sources = format_context(docs)
    prompt = PROMPT.format(context=context, question=question, patient=patient)

    llm = load_llm_cpu()
    raw = llm(prompt, max_new_tokens=240, do_sample=True, temperature=0.7)[0]["generated_text"]

    guarded = apply_guardrail(raw)
    check = safety_check(raw)

    log_event({
        "question": question,
        "patient": patient,
        "sources": sources,
        "retrieved_chunks": [
            {"source": d.metadata.get("source"), "preview": d.page_content[:250]}
            for d in docs
        ],
        "safety": check,
        "answer_preview": guarded[:600]
    })

    return guarded, sources

if __name__ == "__main__":
    patient = {
        "idade": 63,
        "sexo": "M",
        "satO2": 91,
        "temperatura": 38.2,
        "sintomas": "dispneia leve e febre",
        "comorbidades": "DPOC"
    }
    question = "Quais sinais de gravidade e próximos passos neste caso de pneumonia?"
    answer, sources = ask(question, patient)
    print(answer)
    print("\nFONTES:", sources)
