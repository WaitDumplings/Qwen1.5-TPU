import gradio as gr
import mdtex2html
import os
from models import *
from path import Config_Path

bmodel_path = "../bmodels"
config_path = Config_Path(bmodel_path)
config_path.update_all_model_names()

def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y
    
def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text

def predict(input, chatbot, history):
    global model
    chatbot.append((parse_text(input), ""))
    for response, history in model.stream_predict(input, history):
        chatbot[-1] = (parse_text(input), parse_text(response))
        yield chatbot, history

def reset_user_input():
    return gr.update(value='')

def reset_state():
    return [], []

gr.Chatbot.postprocess = postprocess

with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">LLM-TPU</h1>""")
    with gr.Column(scale=0.5, visible=True) as right_col:         
        with gr.Tab(label='Model'):
            with gr.Row():
                llm_model = gr.Dropdown(label='LLM Model', choices= ["None"] + config_path.model_filenames, value = "None", show_label=True, interactive=True)
            with gr.Row():
                load_model_Btn = gr.Button(label='load model', value="Load Bmodel")
                model_refresh = gr.Button(label='Refresh', value='\U0001f504 Refresh All Files', variant='secondary')
                ResetBtn = gr.Button(label='Reset', value='Reset')
            def model_refresh_clicked():
                config_path.update_all_model_names()
                if config_path.model_filenames:
                    return [gr.update(choices= config_path.model_filenames)]
                else:
                    return [gr.update(choices=["None"])]

            def load_bmodel(module):
                global model
                model_path = os.path.join(bmodel_path, module)

                # tokenizer
                tokenizer_name = module.split("_")[0] + "_tokenizer"
                tokenizer_path = os.path.join(bmodel_path, tokenizer_name)
                dev_id = 1
                if 'model' in locals():
                    del model

                if tokenizer_name.lower().startswith("qwen1.5"):
                    model = Qwen1_5(model_path, tokenizer_path, dev_id)
                elif tokenizer_name.lower().startswith("qwen"):
                    pass
                elif tokenizer_name.lower().startswith("chatglm"):
                    model = GLM(model_path, tokenizer_path, dev_id)
                else:
                    raise ValueError("No Model Exists")
            
            load_model_Btn.click(load_bmodel, [llm_model], [], queue=True)
            model_refresh.click(model_refresh_clicked, [], [llm_model], queue=False)
            
    history = gr.State([])
    chatbot = gr.Chatbot()

    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=3).style(
                    container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")

    submitBtn.click(predict, [user_input, chatbot, history],
                    [chatbot, history], show_progress=True, queue=True)
    submitBtn.click(reset_user_input, [], [user_input])

    ResetBtn.click(reset_state, outputs=[chatbot, history], show_progress=True)

demo.queue().launch(share=True, server_name="0.0.0.0", inbrowser=True)

