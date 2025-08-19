import os
import torch
from transformers import AutoTokenizer, AutoModel

#TODO: add chunking based on phrase rule and chunking based on headedness
def chunk(text: str, rules_per_chunk: int) -> list[str]:
    """
    Break each section of chunk_size rules into a chunk.

    Args:
        text: the grammar rules in a str
        chunk_size: the number of rules in a chunk

    Returns:
        A list of chunks (each a string).
    """

    lines = text.split("\n")

    chunks = [
        "\n".join(lines[i:i + rules_per_chunk])
        for i in range(0, len(lines), rules_per_chunk)
    ]
    return chunks

def embed_texts(texts: list[str], emb_model: str) -> list[list[float]]:
    """
    Get embeddings for a list of text strings.

    Args:
        texts (list[str]): A list of text strings.
        emb_model (str): The name of an embedding model.

    Returns:
        list[list[float]]: A list of embedding vectors.
    """

    tokenizer = AutoTokenizer.from_pretrained(emb_model)
    model = AutoModel.from_pretrained(emb_model)

    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    # Get embeddings
    last_hidden_state = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())

    sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    mean_pooled = sum_embeddings / sum_mask

    return mean_pooled.tolist()

