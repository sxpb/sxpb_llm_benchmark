from huggingface_hub import hf_hub_download
import os

model_dir = "/tmp/models"
os.makedirs(model_dir, exist_ok=True)

hf_hub_download(repo_id="TheBloke/Llama-2-7B-Chat-GGUF", filename="llama-2-7b-chat.Q2_K.gguf", local_dir=model_dir)
