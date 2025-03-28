#索引階段Index Phase
#將文章切成一塊一塊(chunk), 並索引起來, 使用階段可以方便系統進行搜尋

import os, re, tiktoken, json
import numpy as np

print("目前所在的資料夾:", os.getcwd())

def iter_tex(data_dir):
    for dir_path, _, file_list in os.walk(data_dir):
        for file_name in file_list:
            if not file_name.endswith('.tex'):
                continue
            full_path = os.path.join(dir_path, file_name)
            yield full_path

def get_segments(full_path):
    with open(full_path,"rt",encoding="UTF-8") as fp:
        text = fp.read().strip()
        text = re.sub(" +", " ", text)
        return text.split("\n")

def calc_tokens(tk: tiktoken.Encoding, seg: str):
    tokens = tk.encode(seg, disallowed_special=())
    return len(tokens)

def process_segments(segments: list[tuple[int, str]], chunk_size=300):
    print(f'Original Segments: {len(segments)}')
    i = 0
    while i + 1 < len(segments):
        #取得當前區塊與下個區塊的長度內容
        seg1_len, seg1_txt = segments[i]
        seg2_len, seg2_txt = segments[i + 1]

        #若兩個區塊長度相加小於上限則合併
        if seg1_len + seg2_len < chunk_size:
            segments[i][0] = seg1_len + seg2_len
            segments[i][1] = f"{seg1_txt}\n{seg2_txt}"
            segments.pop(i + 1) #移除已經被合併的Chunk
        #若區塊大小超過上限則開始處理下一個
        else:
            i += 1
    print(f'Processed Segments: {len(segments)}')
    return [seg[1].strip() for seg in segments]

from tempfile import NamedTemporaryFile as NTF

def dump_segments(segments):
    with NTF("wt", dir=".", delete=False) as fp:
        print(fp.name)
        for i, seg in enumerate(segments):
            fp.write(f"=== Chunk {i} Begin ===\n")
            fp.write(f"{seg}\n")
            fp.write(f"=== Chunk {i} End ===\n\n")

def create_embeddings(openAI_client, chunks):
    resp = openAI_client.embeddings.create(
        model = "text-embedding-3-small",
        input = chunks
    )

    embs = [item.embedding for item in resp.data]
    embs = np.array(embs)

    print(f"Embedding Shape: {embs.shape}")
    return embs

def dump_data(chunks, embs, data_dir):
    chunk_path = f"{data_dir}/chunks.json"
    with open(chunk_path, "wt", encoding="UTF-8") as fp:
        json.dump(chunks, fp , ensure_ascii=False)
    np.save(f"{data_dir}/embs.npy", embs)

tk = tiktoken.get_encoding("cl100k_base")

chunks = list()
for path in iter_tex("src/arXiv-2303.08774v6"):
    segments = get_segments(path)
    segments = [[calc_tokens(tk, seg), seg] for seg in segments]
    segments = process_segments(segments)
    print(segments)
    chunks.extend(segments)
    dump_segments(segments)

from openai import OpenAI
import openai

print(openai.__file__)
print(openai.__version__)

openAI_apikey = "[apikey]"
client = OpenAI(api_key=openAI_apikey)

embs = create_embeddings(client, chunks)
dump_data(chunks, embs, "src")


"""
try:
    print(tk.encode("<|endofprompt|>"))
except Exception as e:
    print(e)
# ValueError: Encountered text corresponding to disallowed special token.

print(tk.encode("<|endofprompt|>", disallowed_special=()))
# 當作一般文字來編碼 - [27, 91, 408, 1073, 41681, 91, 29]

print(tk.encode("<|endofprompt|>", allowed_special="all"))
# 當作特殊 Token 來編碼 - [100276]
"""
