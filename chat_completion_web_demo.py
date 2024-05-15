from llama import Llama
from typing import List, Optional

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
max_seq_len: int = 512
max_batch_size: int = 4
max_gen_len: Optional[int] = None

generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
)


def generate_chat_completion(user_input):
    dialog = [{"role": "user", "content": user_input}]
    result = generator.chat_completion(
        [dialog],
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    return result[0]['generation']['content']


iface = gr.Interface(
    fn=generate_chat_completion, 
    inputs=gr.Textbox(placeholder="Enter prompt..."),
    outputs="text",
    title="LLaMA 3 8B Chat Generation"
)

iface.launch(server_name="0.0.0.0", server_port=9999)
