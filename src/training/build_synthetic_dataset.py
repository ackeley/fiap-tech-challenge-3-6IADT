import json
import random
from pathlib import Path

random.seed(42)

BASE_DIR = Path(__file__).resolve().parents[2]
PROTOCOL_DIR = BASE_DIR / "data" / "synthetic" / "protocols"
OUT_TRAIN = BASE_DIR / "data" / "synthetic" / "train.jsonl"
OUT_VALID = BASE_DIR / "data" / "synthetic" / "valid.jsonl"


def read_protocols():
    protocols = []
    for p in PROTOCOL_DIR.glob("*.md"):
        protocols.append({"name": p.stem, "text": p.read_text(encoding="utf-8")})
    return protocols


def make_patient_case():
    return {
        "idade": random.choice([18, 25, 33, 41, 52, 63, 74]),
        "sexo": random.choice(["F", "M"]),
        "satO2": random.choice([89, 91, 93, 95, 97]),
        "temperatura": random.choice([36.8, 37.5, 38.2, 39.0]),
        "sintomas": random.choice([
            "tosse e febre",
            "dispneia leve e febre",
            "fraqueza e confusão",
            "náuseas e vômitos",
            "dor torácica atípica e mal-estar",
            "poliúria e polidipsia"
        ]),
        "comorbidades": random.choice([
            "sem comorbidades conhecidas",
            "hipertensão",
            "diabetes tipo 2",
            "asma",
            "DPOC",
            "insuficiência renal crônica"
        ])
    }


def make_examples(protocols, n=300):
    examples = []

    for _ in range(n):
        protocol = random.choice(protocols)
        patient = make_patient_case()
        example_type = random.choice(["protocol", "faq", "laudo"])

        if example_type == "protocol":
            instruction = (
                f"Você é um assistente clínico do hospital. "
                f"Use o protocolo interno {protocol['name']} e NÃO prescreva."
            )
            input_text = f"Paciente: {patient}"
            output = (
                f"Com base no protocolo {protocol['name']}, sugere-se:\n"
                f"- Avaliar sinais de gravidade.\n"
                f"- Sugerir exames conforme protocolo.\n"
                f"- Indicar alertas e necessidade de escalonamento.\n"
                f"Decisão final deve ser validada por médico responsável.\n"
                f"Fonte: protocolo {protocol['name']}."
            )

        elif example_type == "faq":
            instruction = "Responda de forma objetiva, segura e sem prescrição direta."
            input_text = "Quando devo acionar a equipe médica imediatamente?"
            output = (
                "Deve-se acionar a equipe médica imediatamente na presença de sinais "
                "de instabilidade clínica, deterioração do estado geral ou critérios "
                "de gravidade definidos em protocolos internos. "
                "A decisão deve ser validada por médico responsável."
            )

        else:  # laudo
            instruction = "Gere um rascunho de laudo clínico estruturado."
            input_text = f"Dados do paciente: {patient}"
            output = (
                "LAUDO (RASCUNHO)\n"
                "Queixa: ...\n"
                "Achados clínicos: ...\n"
                "Hipóteses diagnósticas: ...\n"
                "Exames sugeridos: ...\n"
                "Alertas: ...\n"
                "Plano: encaminhar para avaliação médica."
            )

        examples.append({
            "instruction": instruction,
            "input": input_text,
            "output": output
        })

    return examples


def write_jsonl(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    protocols = read_protocols()

    if not protocols:
        raise RuntimeError(
            "Nenhum protocolo encontrado. "
            "Crie arquivos .md em data/synthetic/protocols/"
        )

    data = make_examples(protocols, n=400)
    split = int(len(data) * 0.9)

    write_jsonl(OUT_TRAIN, data[:split])
    write_jsonl(OUT_VALID, data[split:])

    print("Arquivos gerados com sucesso:")
    print(OUT_TRAIN)
    print(OUT_VALID)


if __name__ == "__main__":
    main()
