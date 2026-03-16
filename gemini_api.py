# Upload and download to Gemini API

import logging
import os
from typing import cast

import dotenv
import fire
import pyrootutils
from google import genai
from openai import OpenAI

from scfg.utils import get_logger

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%d-%m %H:%M:%S",
    level=logging.INFO,
)

log = get_logger(__name__)

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)
DATA_DIR = PROJECT_ROOT / "data"
BATCH_DIR = PROJECT_ROOT / "batches"

dotenv.load_dotenv(PROJECT_ROOT / ".env")


def upload_batch(
    fname: str,
    fpath=BATCH_DIR,
):
    client = genai.Client()
    openai_client = OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    batchfile_path = str(fpath / fname)
    display_name = fname.split(".")[0]

    uploaded_file = client.files.upload(
        file=batchfile_path,
        config=genai.types.UploadFileConfig(
            display_name=display_name, mime_type="jsonl"
        ),
    )
    print(uploaded_file)
    input_file_id = cast(str, uploaded_file.name)
    batch = openai_client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(batch)


def download_batch(
    batch_id: str,
    outpath=BATCH_DIR,
):
    client = genai.Client()
    batch = client.batches.get(name=batch_id)

    print(batch)

    batch_display_name = cast(str, batch.display_name)
    print(client.files.get(name=batch_display_name))

    if batch.state == genai.types.JobState.JOB_STATE_SUCCEEDED:
        batch_dest = cast(genai.types.BatchJobDestination, batch.dest)
        batch_dest_fname = cast(str, batch_dest.file_name)
        output_fname = BATCH_DIR / (
            "batch_" + batch_dest_fname.split("-")[-1] + "_output.jsonl"
        )

        file_content = (
            client.files.download(file=batch_dest_fname).decode("utf-8").splitlines()
        )
        with open(outpath / output_fname, "w") as f:
            for line in file_content:
                f.write(line + "\n")
    else:
        print(f"Current state: {batch.state}")


if __name__ == "__main__":
    fire.Fire()
