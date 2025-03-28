import json
import numpy as np
from faiss import IndexFlatL2
from openai import OpenAI

# conda install -c conda-forge faiss-cpu
# faiss install: https://github.com/facebookresearch/faiss/blob/main/INSTALL.md

def load_data(chunk_path, emb_path):
    with open(chunk_path, "rt", encoding="UTF-8") as fp:
        chunks = json.load(fp)
        key_emb: np.ndarray = np.load(emb_path)
        vectors = IndexFlatL2(key_emb.shape[1])
        vectors.add(key_emb)

        return chunks, vectors
        
def get_query_emb(client: OpenAI, query_text: str):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query_text]
    )

    query_emb = resp.data[0].embedding
    return np.array([query_emb])

def build_prompt(q_emb, q_text, vectors: IndexFlatL2, v_chunks):
    dist, index = vectors.search(q_emb, k=5)
    print(f"Distance: {dist}")
    print(f"Indices: {index}")
    prompts = [v_chunks[i] for i in reversed(index[0])]
    prompts.append(f"問題:{q_text}")
    return "\n\n".join(prompts)

def create_chat(client: OpenAI, prompt):
    sys_prompt = "你現在是個專業的文件檢索問答機器人，︀請根據文件內容的資訊回答問題。"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ],
        stream=True
    )

    def stream_response():
        for resp in response:
            if not resp.choices:
                continue
            token = resp.choices[0].delta.content
            print(end=token, flush=True)
        print()

    return stream_response


openAI_apikey = "[apikey]"
client = OpenAI(api_key=openAI_apikey)

#query_text = "請介紹這篇論文"
#query_text = "GPT-4能夠理解圖片嗎?"
#query_text = "GPT-4的模型參數量多少?"
#query_text = "請問第二頁的內容是什麼?"
query_text = "請摘要第四章節的內容"

chunks, vectors = load_data("src/chunks.json", "src/embs.npy")
query_emb = get_query_emb(client, query_text)
prompt = build_prompt(query_emb, query_text, vectors, chunks)
stream_response = create_chat(client, prompt)

print(f"問題：{query_text}\n回答：")
stream_response()