import gradio as gr
from vector_chat import chat_interface

def ui():
    with gr.Blocks["VectorChat"]:
        enabled = gr.Checkbox(label="Enable VectorChat", name="enable", default=True)
        
    enabled.change(fn=chat_interface.set_enabled, inputs=["enable"])
    
def custom_generate_chat_prompt(text, state, **kwargs):
    # TODO: return prompt
    pass
    
def state_modifier(state, **kwargs):
    chat_context = chat_interface.get_chat_context(state, state["index"])
    return chat_context