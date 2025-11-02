import os
import base64
import numpy as np
from PIL import Image
from pathlib import Path
from openai import OpenAI
from utils import embed_image
from faiss_utils import load_faiss_index, normalize

def search_image_by_question(question, co, top_k=1):
    # Embed the question correctly
    response = co.embed(
        texts=[question],
        input_type="search_query",
        model="embed-v4.0"
    )
    query_emb = response.embeddings.float[0]  # ✅ Correct access

    index, filenames = load_faiss_index()
    norm_query = normalize(np.array(query_emb)).astype("float32")
    
    D, I = index.search(norm_query[np.newaxis, :], top_k)
    matched_paths = [str(Path("images") / filenames[i]) for i in I[0] if i < len(filenames)]
    print("📂 matched_paths:", matched_paths)
    return matched_paths

def encode_image_to_base64(img_path: str) -> str:
    """Encodes an image to base64 for embedding in a prompt."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def answer_question_about_images(question: str, matched_paths: list, client: OpenAI,
                                 model="gpt-4.1-mini", retrieved_text="", verbose=True) -> str:
    """
    Sends a multimodal prompt (text + images) to the LLM and returns the answer.
    """

    SYSTEM_PROMPT = """
You are an AI assistant inside a Retrieval-Augmented Generation (RAG) system.

You must answer questions ONLY using the provided retrieved text and images.
Do NOT use outside knowledge. Do NOT hallucinate.

If the answer is not clearly supported by the retrieved text or images, respond with:
"I don’t have enough information in the provided documents to answer that."

If the user's question is unclear, missing context, or overly broad, ask a clarifying question before answering.

Response Requirements:
- Be concise, factual, and specific.
- Reference page numbers from the retrieved text when possible.
- Describe only what is visually observable in images — no assumptions.
""".strip()

    # Build USER prompt
    USER_PROMPT = f"""
User Question:
{question}

Retrieved Text Context:
{retrieved_text if retrieved_text else "No text context available."}

If the answer cannot be determined from the above context, ask for clarification.
""".strip()

    try:
        # Prepare multimodal message content
        message_content = [{"type": "text", "text": USER_PROMPT}]

        for img_path in matched_paths:
            with open(img_path, "rb") as img:
                img_bytes = img.read()
            message_content.append({
                "type": "image",
                "image": img_bytes,
                "mime_type": "image/png"
            })

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": message_content},
            ],
            max_tokens=800,
        )

        answer_text = response.choices[0].message.content.strip()

        if verbose:
            print("🧠 LLM Response:", answer_text)

        return answer_text

    except Exception as e:
        print(f"❌ Error processing images or getting response: {e}")
        return "An error occurred while processing the request."
