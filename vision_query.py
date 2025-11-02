import os
import base64
import numpy as np
from PIL import Image
from pathlib import Path
from openai import OpenAI
from utils import embed_image
from faiss_utils import load_faiss_index, normalize

def search_image_by_question(question, co, top_k=1):
    response = co.embed(
        texts=[question],
        input_type="search_query",
        model="embed-v4.0"
    )
    query_emb = response.embeddings.float[0]

    index, filenames = load_faiss_index()
    norm_query = normalize(np.array(query_emb)).astype("float32")
    
    D, I = index.search(norm_query[np.newaxis, :], top_k)
    matched_paths = [str(Path("images") / filenames[i]) for i in I[0] if i < len(filenames)]
    print("📂 matched_paths:", matched_paths)
    return matched_paths


def encode_image_to_base64(img_path: str) -> str:
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def answer_question_about_images(question: str, matched_paths: list, client: OpenAI,
                                 model="gpt-4o-mini", retrieved_text="", verbose=True) -> str:
    """
    Sends a multimodal prompt (text + images) to the LLM and returns the answer.
    NOTE: Model changed to gpt-4o-mini (supports image input).
    """

    SYSTEM_PROMPT = """
You are an AI assistant inside a Retrieval-Augmented Generation (RAG) system.

Answer questions ONLY using the provided text and images.
If the answer is not supported by the retrieved information, respond:
"I don’t have enough information in the provided documents to answer that."

If the question is unclear or ambiguous, ask a clarifying question first.

- Be concise, factual, and specific.
- Reference page numbers where possible.
- Describe only what is visually observable — no assumptions.
""".strip()

    USER_PROMPT = f"""
User Question:
{question}

Retrieved Text Context:
{retrieved_text if retrieved_text else "No text context available."}
""".strip()

    try:
        # Build combined content (text + images)
        content_blocks = [{"type": "text", "text": USER_PROMPT}]

        for img_path in matched_paths:
            with open(img_path, "rb") as img:
                img_bytes = img.read()
                content_blocks.append({
                    "type": "image",
                    "image": img_bytes,
                    "mime_type": "image/png"
                })

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content_blocks},
            ],
            max_tokens=800,
        )

        answer_text = response.choices[0].message.content.strip()

        if verbose:
            print("🧠 LLM Response:", answer_text)

        return answer_text

    except Exception as e:
        print("\n❌ ERROR DETAILS →", type(e).__name__, str(e), "\n")
        return "An error occurred while processing the request."
