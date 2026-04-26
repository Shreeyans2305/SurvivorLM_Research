import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "YOUR_USERNAME/survival-model"

# Load model (CPU-friendly settings)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
    device_map="cpu",
    low_cpu_mem_usage=True
)

# Optional system prompt
SYSTEM_PROMPT = """You are a practical survival expert AI.
Give concise, actionable advice.
Prioritize safety and realism.
Answer in a clear, step-by-step manner when possible.
If you don't know the answer, say you don't know instead of guessing.
Provide empathy and encouragement.
"""

def generate(prompt, max_tokens, temperature):
    full_prompt = f"{SYSTEM_PROMPT}\nUser: {prompt}\nAssistant:"

    inputs = tokenizer(full_prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            do_sample=True,
            top_p=0.9
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean output (remove prompt part)
    return response.split("Assistant:")[-1].strip()

# UI
with gr.Blocks() as demo:
    gr.Markdown("# 🧭 Survival AI")
    gr.Markdown("Ask survival-related questions.")

    with gr.Row():
        prompt = gr.Textbox(label="Your question", lines=3)

    with gr.Row():
        max_tokens = gr.Slider(50, 300, value=150, label="Max tokens")
        temperature = gr.Slider(0.1, 1.0, value=0.7, label="Temperature")

    output = gr.Textbox(label="Response")

    submit = gr.Button("Generate")
    submit.click(generate, inputs=[prompt, max_tokens, temperature], outputs=output)

demo.launch()