from typing import TypedDict, Optional, List, Dict, Any
from langgraph.graph import StateGraph, END

from src.assistant.rag_assistant import ask  # usa seu RAG pronto (com fontes)
from src.safety.logger import log_event


# ---------
# 1) Estado do fluxo (o "JSON" que vai passando de nó em nó)
# ---------
class ClinicalState(TypedDict, total=False):
    patient: Dict[str, Any]
    question: str

    pending_exams: List[str]
    severity: str            # "low" | "moderate" | "high"
    alert: bool

    answer: str
    sources: List[str]


# ---------
# 2) Nós do fluxo (funções pequenas e claras)
# ---------
def node_check_pending_exams(state: ClinicalState) -> ClinicalState:
    """
    Simula checagem de exames pendentes.
    Em produção: consultaria banco/prontuário (SQL/NoSQL/API).
    """
    # regra simples para demo
    pending = []
    symptoms = (state["patient"].get("sintomas") or "").lower()
    if "dispneia" in symptoms or "febre" in symptoms:
        pending = ["Radiografia de tórax", "Hemograma", "PCR"]

    state["pending_exams"] = pending
    return state


def node_assess_severity(state: ClinicalState) -> ClinicalState:
    """
    Classificação simples (regra) só para demonstrar fluxo seguro.
    """
    p = state["patient"]
    sat = p.get("satO2")
    temp = p.get("temperatura")

    severity = "low"
    if sat is not None and sat < 92:
        severity = "high"
    elif temp is not None and temp >= 39.0:
        severity = "moderate"

    state["severity"] = severity
    state["alert"] = severity == "high"
    return state


def node_rag_answer(state: ClinicalState) -> ClinicalState:
    """
    Gera resposta contextualizada (RAG + LLM com LoRA).
    """
    answer, sources = ask(state["question"], state["patient"])
    state["answer"] = answer
    state["sources"] = sources
    return state


def node_emit_alert(state: ClinicalState) -> ClinicalState:
    """
    Em produção: enviaria alerta para equipe (email/sms/pager etc).
    Aqui: só registra log.
    """
    log_event({
        "type": "ALERT",
        "severity": state.get("severity"),
        "patient": state.get("patient"),
        "question": state.get("question"),
        "pending_exams": state.get("pending_exams"),
        "sources": state.get("sources", []),
    })
    return state


def node_no_alert(state: ClinicalState) -> ClinicalState:
    """
    Caminho sem alerta — também registra.
    """
    log_event({
        "type": "NO_ALERT",
        "severity": state.get("severity"),
        "patient": state.get("patient"),
        "question": state.get("question"),
        "pending_exams": state.get("pending_exams"),
        "sources": state.get("sources", []),
    })
    return state


# ---------
# 3) Função de roteamento (decisão)
# ---------
def route_alert(state: ClinicalState) -> str:
    return "emit_alert" if state.get("alert") else "no_alert"


# ---------
# 4) Montar o grafo
# ---------
def build_graph():
    g = StateGraph(ClinicalState)

    g.add_node("check_pending_exams", node_check_pending_exams)
    g.add_node("assess_severity", node_assess_severity)
    g.add_node("rag_answer", node_rag_answer)
    g.add_node("emit_alert", node_emit_alert)
    g.add_node("no_alert", node_no_alert)

    # ordem principal do fluxo
    g.set_entry_point("check_pending_exams")
    g.add_edge("check_pending_exams", "assess_severity")
    g.add_edge("assess_severity", "rag_answer")

    # decisão: alerta ou não
    g.add_conditional_edges(
        "rag_answer",
        route_alert,
        {"emit_alert": "emit_alert", "no_alert": "no_alert"}
    )

    # fim
    g.add_edge("emit_alert", END)
    g.add_edge("no_alert", END)

    return g.compile()


# ---------
# 5) Execução de exemplo
# ---------
if __name__ == "__main__":
    app = build_graph()

    state: ClinicalState = {
        "patient": {
            "idade": 63,
            "sexo": "M",
            "satO2": 91,
            "temperatura": 38.2,
            "sintomas": "dispneia leve e febre",
            "comorbidades": "DPOC"
        },
        "question": "Quais sinais de gravidade e próximos passos neste caso de pneumonia?"
    }

    result = app.invoke(state)

    print("\n=== RESULTADO FINAL DO FLUXO ===")
    print("Severidade:", result.get("severity"))
    print("Alerta:", result.get("alert"))
    print("Exames pendentes:", result.get("pending_exams"))
    print("Fontes:", result.get("sources"))
    print("\nResposta:\n", result.get("answer"))
