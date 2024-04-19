import gradio as gr

with gr.Blocks("COMP3340 Group Project Demo") as demo:
  with gr.Tab("Model"):
    pass

if __name__ == "__main__":
  demo.launch(server_name="0.0.0.0")