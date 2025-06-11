if __name__ == '__main__':
    import faiss
    import json
    import numpy as np
    from sentence_transformers import SentenceTransformer

    # 加载 SentenceTransformer 模型
    model_embed = SentenceTransformer("all-MiniLM-L6-v2")

    # 读取塔罗牌数据
    with open("tarot-images.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # 提取塔罗牌文本信息（你可以选择其他字段，如 'meanings' 或 'fortune_telling'）
    # 在这里我们将使用 'fortune_telling' 和 'meanings' 中的 'light' 和 'shadow' 部分
    card_texts = []
    for card in data["cards"]:
        # 提取'fortune_telling'和'meanings'中的'light'和'shadow'文本
        fortune_telling_text = " ".join(card["fortune_telling"])
        meanings_text = " ".join(card["meanings"]["light"] + card["meanings"]["shadow"])

        # 合并文本以提供更多的上下文
        combined_text = fortune_telling_text + " " + meanings_text
        card_texts.append(combined_text)

    # 使用 SentenceTransformer 将文本转换为向量
    card_vectors = model_embed.encode(card_texts).astype("float32")

    # 创建 FAISS 索引
    dimension = card_vectors.shape[1]  # 向量的维度
    index = faiss.IndexFlatL2(dimension)  # 使用 L2 距离度量来创建索引

    # 将向量添加到索引中
    index.add(card_vectors)

    # 保存索引到文件
    faiss.write_index(index, "tarot.index")

    print("FAISS 索引已创建并保存至 tarot.index")