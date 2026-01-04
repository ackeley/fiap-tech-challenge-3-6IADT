# 🏥 Assistente Médico Inteligente com LLM, LangChain e LangGraph

Este projeto implementa um **assistente médico virtual em português (pt-BR)**, treinado com **dados internos (sintéticos)** de um hospital, capaz de **auxiliar condutas clínicas**, **responder dúvidas médicas** e **orquestrar fluxos automatizados**, sempre respeitando **regras rígidas de segurança, validação humana e auditabilidade**.

> ⚠️ O sistema **NÃO prescreve medicamentos**, **NÃO define doses** e **NÃO substitui decisões médicas**.

---

## 🎯 Objetivo

Demonstrar como **modelos de linguagem (LLMs)** podem ser usados de forma **ética, segura e explicável** na área da saúde, integrando:

- Fine-tuning com dados internos
- Recuperação de conhecimento (RAG)
- Orquestração de decisões clínicas
- Logging e auditoria
- Validação humana obrigatória

---

## 🧠 Arquitetura Geral

- **LLM Base:** Qwen/Qwen2.5-3B-Instruct  
- **Fine-tuning:** LoRA (PEFT)  
- **RAG:** LangChain + FAISS  
- **Orquestração:** LangGraph  
- **Segurança:** Guardrails + validação humana  
- **Logs:** JSONL para auditoria  

---

## 📂 Estrutura do Projeto

```text
medical-assistant/
├── data/
│   ├── synthetic/
│   │   ├── train.jsonl
│   │   ├── valid.jsonl
│   │   └── protocols/
│   │       └── protocolo_pneumonia.md
│   └── vectorstore/
│       └── protocols_faiss/
│
├── lora_adapter/
│   ├── adapter_config.json
│   └── adapter_model.safetensors
│
├── logs/
│   └── assistant_audit.jsonl
│
├── src/
│   ├── training/
│   │   └── build_synthetic_dataset.py
│   │
│   ├── rag/
│   │   ├── ingest_protocols.py
│   │   ├── retriever.py
│   │   └── test_retriever.py
│   │
│   ├── assistant/
│   │   ├── llm_local.py
│   │   └── rag_assistant.py
│   │
│   ├── safety/
│   │   ├── logger.py
│   │   └── validator.py
│   │
│   └── flows/
│       └── clinical_flow.py
│
└── README.md
```

## Instalação e Execução

Siga os passos abaixo para configurar e executar o projeto localmente.

### Pré-requisitos

-   Python 3.12.9 ou superior.

### Passo 1: Clonar o Repositório

```bash
git clone https://github.com/ackeley/fiap-tech-challenge-3-6IADT
cd fiap-tech-challenge-3-6IADT
```

### Passo 2: Configurar o Ambiente Virtual

É altamente recomendável usar um ambiente virtual para isolar as dependências.

```bash
# Criar o ambiente
python -m venv venv

# Ativar o ambiente
# No Windows:
# venv\Scripts\activate
# No macOS/Linux:
source venv/bin/activate
```

### Passo 3: Instalar as Dependências

```bash
pip install -r requirements.txt
```

* requirements.txt → dependências do projeto
* requirements.lock.txt → ambiente testado 

### Passo 4: Verificar estrutura mínima esperada
Você precisa ter pelo menos:
```text
data/synthetic/protocols/protocolo_pneumonia.md
data/synthetic/train.jsonl
data/synthetic/valid.jsonl
```

### Passo 5: Indexar protocolos (criar FAISS)
```bash
python -m src.rag.ingest_protocols
```

### Passo 6: Testar Retriever (deve imprimir trecho + fonte)
```bash
python -m src.rag.test_retriever
```

Você deve ver algo como:

- “Trecho 1 | Fonte: protocolo_pneumonia.md”

### Passo 7: Preparar o LoRA adapter (obrigatório para rodar a LLM personalizada)
```text
lora_adapter/adapter_config.json
lora_adapter/adapter_model.safetensors
```
** Essa pasta vem do Colab (lora_adapter.zip) e deve ser descompactada na raiz do projeto, pois foi onde foi treinado o modelo.


### Passo 8: Rodar o Assistente RAG (LangChain + LLM + fontes)
```bash
python -m src.assistant.rag_assistant

```

Resultado esperado:

- resposta em pt-BR

- sem prescrição

- com “Fontes: …”

### Passo 9: Rodar o fluxo LangGraph (pipeline automatizado)
```bash
python -m src.flows.clinical_flow
```

Resultado esperado:

- severidade

- exames pendentes

- alerta (true/false)

- resposta

- logs gravados

### Passo 10: Conferir auditoria (logs)
```text
logs/assistant_audit.jsonl
```

Cada execução registra:

- pergunta

- paciente

- trechos recuperados

- fontes

- resultado do safety check


## 🧪 Dataset e Fine-tuning

### 🔹 Dataset

O dataset utilizado neste projeto é **100% sintético e anonimizado**, desenvolvido exclusivamente para fins de pesquisa, experimentação e demonstração técnica.

Ele é baseado em:

- Protocolos clínicos internos
- Perguntas frequentes realizadas por médicos
- Exemplos de laudos, triagens e orientações clínicas
- Simulações de cenários hospitalares reais

**Formato dos dados:**
- Arquivos `.jsonl` (JSON Lines)
- Cada linha representa uma interação independente

**Estrutura dos registros:**
- `instruction`: instrução principal para o modelo
- `input`: contexto clínico ou dados do paciente (anonimizados)
- `output`: resposta esperada do assistente

---

### 🔹 Fine-tuning

O fine-tuning do modelo foi realizado utilizando a técnica **LoRA (Low-Rank Adaptation)**, que permite adaptar modelos de grande porte com **baixo custo computacional** e **mínima alteração do modelo base**.

**Características do processo:**

- Execução em ambiente com GPU (ex: Google Colab)
- Uso de PEFT (Parameter-Efficient Fine-Tuning)
- Preservação completa do modelo original
- Ajustes focados exclusivamente no domínio clínico

**Vantagens da abordagem:**

- Redução significativa de custo
- Facilidade de versionamento
- Adequado para ambientes regulados
- Menor risco de degradação do modelo base

**Resultado do treinamento:**

- Adaptadores treinados armazenados na pasta:
  ```text
  lora_adapter/
  ``` 

  **Arquivos principais:**

- `adapter_config.json`  
- `adapter_model.safetensors`  

Esses adaptadores são **carregados dinamicamente durante a execução do assistente**, permitindo alternar ou versionar ajustes do modelo sem necessidade de novo treinamento.

---

## 🔎 RAG — Retrieval Augmented Generation

O assistente **não responde apenas com conhecimento do modelo**.

**Fluxo RAG:**

- Protocolos clínicos em arquivos `.md` são indexados  
- Embeddings gerados com `sentence-transformers`  
- Vetores armazenados em **FAISS**  

A cada pergunta:

- Trechos relevantes são recuperados  
- O contexto é injetado no prompt  
- A resposta é gerada com base nas fontes  

Isso garante **respostas mais seguras, rastreáveis e facilmente atualizáveis**.

---

## 🔐 Segurança e Validação

### 🛡️ Guardrails

Bloqueio automático de conteúdo sensível, como:

- Prescrição de medicamentos  
- Dosagens e posologia  
- Receitas e orientações terapêuticas diretas  

Implementado por meio de:

- Expressões regulares (Regex)  
- Validação pós-geração  

---

### 👨‍⚕️ Validação Humana

- **Toda resposta exige validação de um médico responsável**  
- Nenhuma decisão clínica é automatizada ou tomada pela IA  

---

## 🧾 Logging e Auditoria

Cada execução gera um registro estruturado em:

```bash
logs/assistant_audit.jsonl
```

Inclui:

- Timestamp  
- Pergunta realizada  
- Dados do paciente (anonimizados)  
- Trechos recuperados via RAG  
- Fontes utilizadas  
- Resultado do safety check  
- Tipo de evento (`ALERT` / `NO_ALERT`)  

---

## 🔁 LangGraph — Fluxo Clínico Automatizado

O **LangGraph** coordena todo o fluxo clínico:

1. Recebe dados do paciente + pergunta  
2. Verifica exames pendentes  
3. Avalia a gravidade do caso  
4. Executa recuperação via RAG  
5. Aplica validações de segurança  
6. Decide se gera alerta  
7. Registra auditoria  
8. Finaliza o fluxo  

---

## ▶️ Como Executar

### 1️⃣ Indexar protocolos clínicos
```bash
python -m src.rag.ingest_protocols
```

### 2️⃣ Testar recuperação de contexto
```bash
python -m src.rag.test_retriever
```

### 3️⃣ Executar assistente RAG
```bash
python -m src.assistant.rag_assistant
```
### 4️⃣ Executar fluxo completo com LangGraph
```bash
python -m src.flows.clinical_flow