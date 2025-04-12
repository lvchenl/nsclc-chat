import gradio as gr
import json
import numpy as np
import faiss
import datetime
import os
from collections import deque
from openai import OpenAI
import requests

# Configuration
EMBED_API_URL = "http://172.175.0.175:11434/api/embed"
RESULTS_DIR = "results"

client = OpenAI(
    base_url="http://172.175.0.175:8099",
    api_key="5dab9eca-64ca-592e-a5cb-72cdc327cb7b"
)

# Prompt segments
DIRECT_SYSTEM_PREFIX = "You are a highly knowledgeable and trustworthy medical assistant. Use the information below to "
RAG_SYSTEM_PREFIX = "You are a highly knowledgeable and trustworthy medical assistant. Use the following retrieved information to "

SYSTEM_PROMPT_SHARED = (
    "create a personalized cancer treatment plan. Provide accurate, evidence-based recommendations tailored to the patient. \n\n"
    "You need to search new and clinical scientific studies about the information of patient from the Internet to finish this task successfully. \n\n"
    "When referencing scientific studies, include citation markers such as DOI or PMID where available.\n\n"
    "Also incorporate key principles from the NCCN Clinical Practice Guidelines in Oncology: Non–Small Cell Lung Cancer, "
    "Version 4.2024, including:\n\n"
    "🔬 Molecular Testing & Targeted Therapy:\n"
    "- Mandatory broad molecular profiling: EGFR, ALK, ROS1, BRAF, MET, RET, KRAS, NTRK, ERBB2, PD-L1.\n"
    "- ALK-positive: First-line options include alectinib, brigatinib, ceritinib; lorlatinib for resistance mutations.\n"
    "- Oligoprogression: Local therapies (SABR/surgery) are preferred before switching systemic therapy.\n"
    "- PD-1 monotherapy has limited efficacy in ALK-positive NSCLC.\n\n"
    "🔁 ROS1 Rearrangement:\n"
    "- Treat with crizotinib, ceritinib, or lorlatinib (similar pathway to ALK).\n\n"
    "🧠 CNS Progression:\n"
    "- Use SRS +/- surgery for symptomatic brain lesions; SRS also for high-risk asymptomatic lesions.\n\n"
    "🧪 Biomarker-Based Treatment:\n"
    "- EGFR: Osimertinib (including adjuvant in Stage IB–IIIA).\n"
    "- PD-L1 ≥50%: Pembrolizumab monotherapy (if no driver mutation).\n"
    "- PD-L1 <50%: Chemo-immunotherapy combos recommended.\n"
    "- Avoid immunotherapy in EGFR/ALK-positive patients unless no targeted options remain.\n\n"
    "⚡ Advanced Disease Strategy:\n"
    "- Consider local therapy (SABR, IGTA) in oligometastatic/oligoprogressive disease.\n"
    "- Use genotyping after progression to identify resistance and guide next-line treatment.\n"
    "\nUse the information provided below, along with these clinical principles, to generate a complete, patient-specific treatment plan with a short summary including specific treatments or drugs."
)

# Shared memory
global_chat_history = deque(maxlen=50)

def load_index_and_chunks():
    index = faiss.read_index("faiss_index.bin")
    with open("chunks.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return index, meta["chunks"], meta["sources"]

def embed_query(text):
    payload = {"model": "nomic-embed-text", "input": [text]}
    response = requests.post(EMBED_API_URL, json=payload)
    response.raise_for_status()
    return np.array(response.json()["embeddings"][0], dtype='float32')

def retrieve_chunks(query_embedding, index, chunks, sources, k=5):
    D, I = index.search(query_embedding.reshape(1, -1), k)
    return [f"{chunks[i]} (Source: {sources[i]})" for i in I[0]]

def save_chat_markdown(history):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(RESULTS_DIR, f"chat_{timestamp}.md")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# NSCLC Chat Session\n**Timestamp:** {timestamp}\n\n")
        for turn in history:
            role = "🧑 User" if turn['role'] == 'user' else "🤖 Assistant"
            f.write(f"## {role}\n{turn['content']}\n\n")
    return filename

def query_chat_model(messages):
    response = client.chat.completions.create(
        model="ds-r1-qwen-14b",
        messages=messages,
        temperature=0.3,
        stream=False
    )
    return response.choices[0].message.content

def direct_chat(user_input, chat_history):
    chat_history.append({"role": "user", "content": user_input})
    messages = [{"role": "system", "content": DIRECT_SYSTEM_PREFIX + SYSTEM_PROMPT_SHARED}] + list(chat_history)
    reply = query_chat_model(messages)
    chat_history.append({"role": "assistant", "content": reply})
    global_chat_history.clear()
    global_chat_history.extend(chat_history)
    return chat_history, [(x['content'], y['content']) for x, y in zip(chat_history[::2], chat_history[1::2])]

def rag_chat(user_input, chat_history):
    index, chunks, sources = load_index_and_chunks()
    embedding = embed_query(user_input)
    retrieved = retrieve_chunks(embedding, index, chunks, sources, k=5)
    context = "\n".join(retrieved)
    chat_history.append({"role": "user", "content": user_input})
    messages = [
        {"role": "system", "content": RAG_SYSTEM_PREFIX + SYSTEM_PROMPT_SHARED + "\n\nContext:\n" + context}
    ] + list(chat_history)
    reply = query_chat_model(messages)
    chat_history.append({"role": "assistant", "content": reply})
    global_chat_history.clear()
    global_chat_history.extend(chat_history)
    return chat_history, [(x['content'], y['content']) for x, y in zip(chat_history[::2], chat_history[1::2])]

def new_chat():
    global_chat_history.clear()
    return [], []

def save_chat():
    return f"✅ Saved to `{save_chat_markdown(global_chat_history)}`"

# ✅ Exported Gradio app creator (for FastAPI mounting)
def create_gradio_app():
    with gr.Blocks(title="NSCLC Chat with RAG and Memory") as demo:
        gr.Markdown("### 🧠 NSCLC Cancer Treatment Chat Assistant")
        new_btn = gr.Button("🔄 New Chat")
        chatbot = gr.Chatbot()
        user_box = gr.Textbox(placeholder="Type your question here...", show_label=False)
        with gr.Row():
            direct_btn = gr.Button("⬆️")
            rag_btn = gr.Button("RAG")
        status_box = gr.Textbox(label="Status")
        save_btn = gr.Button("💾 Save Chat")
        state = gr.State([])

        direct_btn.click(fn=direct_chat, inputs=[user_box, state], outputs=[state, chatbot])
        rag_btn.click(fn=rag_chat, inputs=[user_box, state], outputs=[state, chatbot])
        new_btn.click(fn=new_chat, outputs=[state, chatbot])
        save_btn.click(fn=save_chat, outputs=status_box)

    return demo
