from llama import Llama
from typing import List

import os
import gradio as gr
import torch
import torch.distributed as dist


ckpt_dir = "/path/to/Meta-Llama-3-8B-Instruct/"
tokenizer_path = "/path/to/Meta-Llama-3-8B-Instruct/tokenizer.model"

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'

temperature: float = 0.6
top_p: float = 0.9
max_seq_len: int = 128
max_gen_len: int = 64
max_batch_size: int = 4

generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
)


def generate_text_completion(prompt):
    result = generator.text_completion(
        [prompt],
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    return result[0]['generation']


iface = gr.Interface(
    fn=generate_text_completion, 
    inputs=gr.Textbox(placeholder="Enter prompt..."),
    outputs="text",
    title="LLaMA 3 8B Text Generation"
)

iface.launch(server_name="0.0.0.0", server_port=8888)
