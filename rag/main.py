import fire
import os
import json
import numpy as np
import logging
import pandas as pd
import openai
import pyrootutils
from dotenv import load_dotenv
from transformers import AutoModel

# Set up project root and add to Python path
PROJECT_ROOT = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)
import sys
sys.path.append(str(PROJECT_ROOT))

from rag import embed_texts, chunk
from vector_db import VectorDB
from scfg.prompt import ChatCompletionResponse, basic_prompt
from scfg.utils import get_logger


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%d-%m %H:%M:%S",
    level=logging.INFO,
)

log = get_logger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
BATCH_DIR = PROJECT_ROOT / "batches"

"""
Runs an experiment with retrieval-augmented generation on a given grammar.
Allows for selecting the following parameters:
1. Embedding Model
2. Chunking
    - Chunks based on headed vs. non-headed
    - Chunks based on phrase type
"""

load_dotenv()

def retrieve_context(
        db: VectorDB,
        emb_model: str,
        sample: str,
        num_results: int,
    ) -> str:
    """
    Embeds the query (sample) and then retrieves the closest 
    """
    
    # Embed the query 
    # TODO: Should query be embedded in parts?
    query_embedding = embed_texts(texts=[sample], emb_model=emb_model)[0]
    query_embedding = np.asarray(query_embedding, dtype=float)

    # Search and return N results
    context = db.search(query_vector=query_embedding, num_results=num_results)

    return context

def generate_batchfile_rag(
        grammar_name: str,
        db: VectorDB,
        emb_model: str,
        prompt_type: str = "basic",
        model: str = "o4-mini",
        max_new_tokens: int | None = None,
        num_results: int = 5,
    ):
    """
    Args:
        grammar_name: the id of the grammar
        db: the VectorDB object holding the SCFG rule embeddings 
        emb_model: the model to do embeddings with
        prompt_type:
        model:
        max_new_tokens: 
        num_results: how many results to return as context
    """
    samples = []
    with open(DATA_DIR / f"samples_{grammar_name}.jsonl", "r") as f:
        for line in f:
            sample = json.loads(line)
            samples.append(sample)

    grammar_path = DATA_DIR / f"grammar_{grammar_name}.json"
    with open(grammar_path, "r") as f:
        grammar = json.load(f)
    n_words = grammar["n_words"]
    n_rules = grammar["n_rules"]

    df = pd.DataFrame(samples)

    if prompt_type == "basic":
        prompt_func = basic_prompt
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    df["prompt"] = df.apply(
        lambda row: prompt_func(
            grammar_str=retrieve_context(db, emb_model, row["left_phonetic"], num_results), 
            sample=row["left_phonetic"]
        ),
        axis=1,
    )
    df["json"] = df.apply(
        lambda row: ChatCompletionResponse(
            user_prompt=row["prompt"],
            max_new_tokens=max_new_tokens,
            metadata={
                "input_sentence": row["left_phonetic"],
                "output_sentence": row["right_phonetic"],
                "grammar_name": grammar_name,
                "n_words": str(n_words),
                "n_rules": str(n_rules),
                "model": model,
                "depth": str(row["depth"]),
            },
        ).to_openai_batched_json(model=model, custom_id=f"request-{row.name}"),
        axis=1,
    )

    model_pathsafe_name = model.replace("/", "_")
    batch_jsonl_filename = f"inputs_{grammar_name}_{model_pathsafe_name}.jsonl"
    batch_jsonl_path = BATCH_DIR / batch_jsonl_filename
    log.info(f"Writing batch job to {batch_jsonl_path}")

    with open(batch_jsonl_path, "w") as f:
        for j in df["json"]:
            f.write(f"{j}\n")


def run_experiment(
        grammar_name: str,
        emb_model: str, 
        rules_per_chunk: int,
        model_name: str = "o4-mini",
        max_new_tokens: int = 1000
        ) -> None:
    '''
    Embeds an SCFG using the specified embedding model from huggingface.
    Runs an experiment with translation using RAG.

    Args:
        params_name: The name of a SCFGParams object
        emb_model: The name of an embedding model from huggingface's library
        chunking: The types of chunking
        model: the model that will be queried
    '''
    grammar_path = DATA_DIR / f"grammar_{grammar_name}.json"
    with open(grammar_path, "r") as f:
        scfg = json.load(f)

    # Break up rules into chunks and store as embeddings in a vector db
    chunks = chunk(scfg["grammar_str"], rules_per_chunk)
    
    # Get embedding dimension from the model
    embedding_model = AutoModel.from_pretrained(emb_model)
    embedding_dim = embedding_model.config.hidden_size
    db = VectorDB(embedding_dim)
    db.add_chunks(chunks, emb_model) # FIXME: Pass the actual embedding model

    # Generate a batchfile with just retrieved context in the prompt
    generate_batchfile_rag(grammar_name=grammar_name, emb_model=emb_model, db=db, model=model_name, max_new_tokens=max_new_tokens)

    # Batch run 
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)

    input_path = f"batches/inputs_{grammar_name}_{model_name}.jsonl"
    output_path = f"batches/responses_{grammar_name}_{model_name}.jsonl"

    # Opens inputs file and sends requests to OpenAI for processing
    with open(input_path, "r") as infile, open(output_path, "w") as outfile:
        for line in infile:
            request = json.loads(line)
            custom_id = request.get("custom_id")

            try:
                response = client.chat.completions.create(**request["body"])
            except Exception as e:
                print(f"Error processing request {request.get('custom_id')}: {e}")
                continue

            # Save request + response in JSONL
            result = {
                "custom_id": custom_id,
                "request": request,
                "response": response.model_dump()  # converts to serializable dict
            }
            outfile.write(json.dumps(result) + "\n")
            outfile.flush()

    print(f"Done. Responses saved to {output_path}")

    return

if __name__ == "__main__":
    fire.Fire({
        "run_experiment": run_experiment
    })