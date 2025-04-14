import gradio as gr
import json
import numpy as np
import faiss
import datetime
import os
from collections import deque
import requests
from openai import OpenAI

# ✅ API keys and clients
HUGGINGFACE_API_KEY = os.getenv("HF_API_KEY")
ALIYUN_API_KEY = os.getenv("ALIYUN_API_KEY")

client = OpenAI(
    api_key=ALIYUN_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

HF_HEADERS = {
    "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
    "Content-Type": "application/json"
}

# ✅ Global constants
RESULTS_DIR = "results"
global_chat_history = deque(maxlen=50)

# Prompt segments
DIRECT_SYSTEM_PREFIX = (
    "You are a clinical oncology expert. Review the patient information below and generate a structured, evidence-based treatment plan aligned with current NCCN guidelines and recent literature. "
)

RAG_SYSTEM_PREFIX = (
    "You are a clinical oncology expert. Use the following retrieved references and patient details to generate a structured, evidence-based treatment plan aligned with NCCN guidelines and recent studies. "
)

SYSTEM_PROMPT_SHARED = (
    "You are a clinical oncology expert. Your task is to generate a personalized, mutation-guided treatment plan for a patient with Non–Small Cell Lung Cancer (NSCLC), following the NCCN Clinical Practice Guidelines in Oncology (Version 4.2024) and supported by peer-reviewed studies.\n\n"

    "🧬 **Molecular Biomarker-Driven Personalization**:\n"
    "- Perform clinical interpretation of all listed mutations (e.g., EGFR, ALK, TP53, CDKN2A, RET, MET, KRAS G12C, ERBB2, ROS1, SMAD4).\n"
    "- Identify therapeutic relevance: Is it directly targetable? A co-mutation influencing prognosis/resistance? Actionable only via clinical trials?\n\n"

    "📋 **Required Output Format:**\n"
    "1. **Title** – e.g., 'Treatment Recommendation for EGFR Exon 19del + TP53 + CDKN2A Mutations in Advanced NSCLC'\n"
    "2. **Primary Treatment** – First-line treatment with drug name, dosage, and justification.\n"
    "3. **Rationale** – Explain biomarker implications, resistance risk, and supportive trial data.\n"
    "4. **Subsequent Therapy Options** – Guidance based on progression type: resistance mutation (e.g., T790M), histologic transformation (e.g., SCLC), CNS involvement, etc.\n"
    "5. **Adjunctive Therapies** – Radiation, bone-modifying agents, prophylactic anticoagulation if relevant.\n"
    "6. **Monitoring & Follow-up** – Imaging frequency, re-biopsy recommendation, pneumonitis surveillance, and germline testing triggers.\n"
    "7. **Final Drug Plan Summary** – Output a concise regimen: drug(s), dose, route, cycle, and timing (e.g., Osimertinib 80 mg PO QD + Carboplatin AUC 5 IV q3w x 4 cycles).\n\n"

    "📚 **Mandatory References to Use Where Appropriate**:\n"
    "- FLAURA: Osimertinib vs. 1st-gen EGFR TKIs (N Engl J Med. 2018; DOI: 10.1056/NEJMoa1713137)\n"
    "- FLAURA2: Osimertinib + chemo vs. Osimertinib (NEJM. 2023; DOI: 10.1056/NEJMoa2301385)\n"
    "- MARIPOSA-2: Amivantamab-vmjw + chemo post-Osimertinib (Ann Oncol. 2024; 35:77–90)\n"
    "- NCCN Guidelines: NSCLC Version 4.2024 (DOI: 10.6004/jnccn.2204.0023)\n"
    "- CHRYSALIS: Amivantamab in EGFR exon 20 insertions (JCO. 2021; 39:3391–3402)\n"
    "- IMMUNOTARGET: Poor ICI response in EGFR/ALK-positive NSCLC (Ann Oncol. 2019; 30:1321–1328)\n"
    "- CROWN: Lorlatinib vs. Crizotinib in ALK+ NSCLC (N Engl J Med. 2020; 383:2018–2029)\n\n"

    "⚠️ **Key Rules from Guidelines:**\n"
    "- Targeted therapy is preferred first-line when driver mutations are present — regardless of PD-L1 status.\n"
    "- Avoid ICIs (e.g., pembrolizumab, nivolumab) in EGFR- or ALK-positive NSCLC unless no targeted options exist.\n"
    "- If osimertinib follows recent ICI use, monitor closely for pneumonitis (*PMID: 31079805*).\n"
    "- Oligoprogression: Favor local therapy (SABR/surgery) over switching systemic regimens.\n"
    "- Rebiopsy at progression is essential to detect histologic transformation (e.g., SCLC) or resistance (T790M, C797S).\n"
    "- Category 1 recommendations reflect highest-level evidence and consensus.\n\n"

    "🎯 **Final Goal:**\n"
    "Deliver a clinically actionable, evidence-backed, mutation-aware treatment plan — as if intended for oncologists or tumor board review."
)
# ✅ Load FAISS + chunks
def load_index_and_chunks():
    index = faiss.read_index("faiss_index.bin")
    with open("chunks.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta["chunks"], meta["sources"]

# ✅ Alibaba Embedding
def embed_query(text):
    response = client.embeddings.create(
        model="text-embedding-v3",
        input=[text],
        dimensions=1024,
        encoding_format="float"
    )
    embedding = response.data[0].embedding
    return np.array(embedding, dtype='float32')

# ✅ RAG retrieval
def retrieve_chunks(query_embedding, index, chunks, sources, k=5):
    D, I = index.search(query_embedding.reshape(1, -1), k)
    return [f"{chunks[i]} (Source: {sources[i]})" for i in I[0]]

# ✅ Query chat model
def query_chat_model(messages):
    prompt = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages]) + "\nAssistant:"
    url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    response = requests.post(url, headers=HF_HEADERS, json={
        "inputs": prompt,
        "parameters": {"max_new_tokens": 512}
    })
    response.raise_for_status()
    return response.json()[0]["generated_text"].split("Assistant:")[-1].strip()

# ✅ Save .md transcript
def save_chat_markdown(history):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filename = os.path.join(RESULTS_DIR, f"chat_{datetime.datetime.now():%Y%m%d_%H%M%S}.md")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# NSCLC Chat Session\n**Timestamp:** {datetime.datetime.now()}\n\n")
        for turn in history:
            role = "🧑 User" if turn['role'] == 'user' else "🤖 Assistant"
            f.write(f"## {role}\n{turn['content']}\n\n")
    return filename

# ✅ Direct chat
def direct_chat(user_input, chat_history):
    chat_history.append({"role": "user", "content": user_input})
    messages = [{"role": "system", "content": DIRECT_SYSTEM_PREFIX + SYSTEM_PROMPT_SHARED}] + list(chat_history)
    reply = query_chat_model(messages)
    chat_history.append({"role": "assistant", "content": reply})
    global_chat_history.clear()
    global_chat_history.extend(chat_history)
    return chat_history, chat_history

# ✅ RAG chat
def rag_chat(user_input, chat_history):
    index, chunks, sources = load_index_and_chunks()
    embedding = embed_query(user_input)
    context = "\n".join(retrieve_chunks(embedding, index, chunks, sources))
    chat_history.append({"role": "user", "content": user_input})
    messages = [{"role": "system", "content": RAG_SYSTEM_PREFIX + SYSTEM_PROMPT_SHARED + "\n\nContext:\n" + context}] + list(chat_history)
    reply = query_chat_model(messages)
    chat_history.append({"role": "assistant", "content": reply})
    global_chat_history.clear()
    global_chat_history.extend(chat_history)
    return chat_history, chat_history

# ✅ Chat reset
def new_chat():
    global_chat_history.clear()
    return [], []

# ✅ Save chat + download
def save_chat():
    filepath = save_chat_markdown(global_chat_history)
    return "", filepath

# ✅ Gradio UI
def create_gradio_app():
    with gr.Blocks(title="NSCLC Chat with RAG and Memory") as demo:
        gr.Markdown("### 🧠 NSCLC Cancer Treatment Chat Assistant")
        new_btn = gr.Button("🔄 New Chat")
        chatbot = gr.Chatbot(type="messages")
        user_box = gr.Textbox(placeholder="Type your question here...", show_label=False)
        with gr.Row():
            direct_btn = gr.Button("⬆️ Direct")
            rag_btn = gr.Button("🔍 RAG")
        save_btn = gr.Button("💾 Save & Download Chat")
        download_btn = gr.File(label="📥 Markdown Output")
        state = gr.State([])

        direct_btn.click(fn=direct_chat, inputs=[user_box, state], outputs=[state, chatbot])
        rag_btn.click(fn=rag_chat, inputs=[user_box, state], outputs=[state, chatbot])
        new_btn.click(fn=new_chat, outputs=[state, chatbot])
        save_btn.click(fn=save_chat, outputs=[gr.Textbox(visible=False), download_btn])

    return demo
