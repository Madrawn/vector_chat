import gradio as gr
import extensions.vector_chat.vc.vector_chat as vector_chat
from modules import shared

chat_interface = vector_chat.ChatInterface()


def ui():
    with gr.Row():
        with gr.Column():
            enabled = gr.Checkbox(label="Enable VectorChat", name="enable", default=True)
            clear = gr.Button("Clear Chat History", name="clear")
            dropdown = gr.Dropdown(label="Distance Function", name="dist", choices=["l2", "cosine", "ip"], default="cosine", value="cosine")
        with gr.Column():
            gr.Label("Stats:")
            stats = gr.Textbox(name="stats", type="text", default="", rows=10, cols=10, placeholder="DB stats will appear here")
            refresh = gr.Button("Refresh", name="refresh")

    refresh.click(fn=chat_interface.refresh_db, outputs=stats)
    enabled.change(fn=chat_interface.set_enabled, inputs=[enabled])
    clear.click(fn=chat_interface.clear)
    dropdown.change(fn=chat_interface.set_distance, inputs=[dropdown])


def custom_generate_chat_prompt(text, state, **kwargs):
    global chat_interface
    if not chat_interface.enabled:
        return
    _continue = kwargs.get('_continue', False)
    chat_interface.init(shared)
    msgs = [[user, ai] for user,ai in state["history"]["visible"]]
    chat_interface.add_multiple_messages(msgs, state)

    
    prompt = chat_interface.get_chat_context(text, len(state["history"]["visible"]), state, _continue)
    return prompt 
