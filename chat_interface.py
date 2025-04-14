import gradio as gr
import json
import numpy as np
import faiss
import datetime
import os
from collections import deque
import requests
import openai

# ✅ API Keys from environment variables
HUGGINGFACE_API_KEY = os.getenv("HF_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# ✅ Headers for Hugging Face
HEADERS = {
    "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
    "Content-Type": "application/json"
}

# Global variables
RESULTS_DIR = "results"
global_chat_history = deque(maxlen=50)

# ✅ Prompt segments
DIRECT_SYSTEM_PREFIX = (
    "You are a clinical oncology expert. Review the patient information below and generate a structured, evidence-based treatment plan aligned with current NCCN guidelines and recent literature. "
)

RAG_SYSTEM_PREFIX = (
    "You are a clinical oncology expert. Use the following retrieved references and patient details to generate a structured, evidence-based treatment plan aligned with NCCN guidelines and recent studies. "
)

SYSTEM_PROMPT_SHARED = (
    "You are a clinical oncology expert. Your task is to generate a personalized, mutation-guided treatment plan for a patient with Non–Small Cell Lung Cancer (NSCLC), following the NCCN Clinical Practice Guidelines in Oncology (Version 4.2024) and supported by peer-reviewed studies.\n\n"
    "🧬 Molecular Biomarker-Driven Personalization:\n"
    "- Perform clinical interpretation of all listed mutations (e.g., EGFR, ALK, TP53, CDKN2A, RET, MET, KRAS G12C, ERBB2, ROS1, SMAD4).\n"
    "- Identify therapeutic relevance: Is it directly targetable? A co-mutation influencing prognosis/resistance? Actionable only via clinical trials?\n\n"
    "📋 Required Output Format:\n"
    "1. Title\n2. Primary Treatment\n3. Rationale\n4. Subsequent Therapy Options\n5. Adjunctive Therapies\n6. Monitoring & Follow-up\n7. Final Drug Plan Summary\n\n"
    "📚 Mandatory References:\n"
    "- FLAURA, FLAURA2, MARIPOSA-2, NCCN 2024, CHRYSALIS, IMMUNOTARGET, CROWN.\n\n"
    "⚠️ Key Rules:\n"
    "- Targeted therapy > ICIs when mutations exist.\n"
    "- Avoid immunotherapy in EGFR/ALK+ unless no targeted option remains.\n"
    "- Local therapy for oligoprogression preferred.\n"
    "- Rebiopsy essential at progression.\n\n"
    "🎯 Final Goal: Create an actionable treatment plan for oncologists or tumor boards."
)

# ✅ Load FAISS index and chunks
def load_index_and_chunks():
    index = faiss.read_index("faiss_index.bin")
    with open("chunks.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta["chunks"], meta["sources"]

# ✅ Embedding via OpenAI (compatible with SDK ≥ 1.0.0)
def embed_query(text):
    response = openai.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    embedding = response.data[0].embedding
    return np.array(embedding, dtype='float32')

# ✅ RAG chunk retrieval
def retrieve_chunks(query_embedding, index, chunks, sources, k=5):
    D, I = index.search(query_embedding.reshape(1, -1), k)
    return [f"{chunks[i]} (Source: {sources[i]})" for i in I[0]]

# ✅ Query Hugging Face Mistral model
def query_chat_model(messages):
    prompt = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in messages]) + "\nAssistant:"
    url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    response = requests.post(url, headers=HEADERS, json={"inputs": prompt, "parameters": {"max_new_tokens": 512}})
    response.raise_for_status()
    return response.json()[0]["generated_text"].split("Assistant:")[-1].strip()

# ✅ Save chat
def save_chat_markdown(history):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filename = os.path.join(RESULTS_DIR, f"chat_{datetime.datetime.now():%Y%m%d_%H%M%S}.md")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# NSCLC Chat Session\n**Timestamp:** {datetime.datetime.now()}\n\n")
        for turn in history:
            role = "🧑 User" if turn['role'] == 'user' else "🤖 Assistant"
            f.write(f"## {role}\n{turn['content']}\n\n")
    return filename

# ✅ Direct Mode
def direct_chat(user_input, chat_history):
    chat_history.append({"role": "user", "content": user_input})
    messages = [{"role": "system", "content": DIRECT_SYSTEM_PREFIX + SYSTEM_PROMPT_SHARED}] + list(chat_history)
    reply = query_chat_model(messages)
    chat_history.append({"role": "assistant", "content": reply})
    global_chat_history.clear()
    global_chat_history.extend(chat_history)
    return chat_history, chat_history

# ✅ RAG Mode
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

def new_chat():
    global_chat_history.clear()
    return [], []

def save_chat():
    filepath = save_chat_markdown(global_chat_history)
    return "", filepath

# ✅ Gradio interface
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
        download_btn = gr.File(label="📥 Download Markdown File")
        state = gr.State([])

        direct_btn.click(fn=direct_chat, inputs=[user_box, state], outputs=[state, chatbot])
        rag_btn.click(fn=rag_chat, inputs=[user_box, state], outputs=[state, chatbot])
        new_btn.click(fn=new_chat, outputs=[state, chatbot])
        save_btn.click(fn=save_chat, outputs=[gr.Textbox(visible=False), download_btn])

    return demo
