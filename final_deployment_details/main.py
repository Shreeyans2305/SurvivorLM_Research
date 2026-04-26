from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os, json

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/survival-q4_k_m.gguf")
HF_REPO    = os.environ.get("HF_REPO",   "Shreyy2305/survival-gguf")
GGUF_FILE  = os.environ.get("GGUF_FILE", "survival-q4_k_m.gguf")
HF_TOKEN   = os.environ.get("HF_TOKEN",  None)

SYSTEM_PROMPT = """You are SurvivalGuide, an expert survival assistant trained to help \
people in emergency and disaster situations. You provide clear, practical, actionable \
advice on power outages, water shortages, natural disasters, wilderness survival, \
first aid, and emergency preparedness. If you don't know the answer, say you don't know instead of guessing. \
Provide empathy and encouragement. 

Rules:
1. Only answers questions related to survival, emergencies, and disasters — do not provide general advice or information on unrelated topics.
2. Lead every response with the single most critical action first
3. Be very concise and direct — people in emergencies need fast answers
4. Avoid mult-step responses — give the one thing they should do right now, then explain next steps if relevant or explicitly asked for more detail
5. Flag anything life-threatening with WARNING at the start
6. If someone needs emergency services, say so in the first sentence
7. Never give specific medication dosage advice — direct to medical professionals
8. When uncertain, say so and give the safest conservative option
9. Only provide answers based on widely accepted best practices from reputable sources like the Red Cross, CDC, WHO, FEMA, and established survival experts — do not give advice based on fringe theories or unproven techniques.
"""

llm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm
    try:
        from llama_cpp import Llama

        if os.path.exists(MODEL_PATH):
            model_path = MODEL_PATH
        else:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=GGUF_FILE,
                token=HF_TOKEN,
                local_dir="/tmp/models",
            )

        llm = Llama(
            model_path=model_path,
            n_ctx=1024,
            n_threads=int(os.environ.get("N_THREADS", "4")),
            n_gpu_layers=0,
            verbose=False,
            chat_format="gemma",
        )
        print("✅ Model ready")

    except Exception as e:
        print(f"❌ Model load failed: {e}")

    yield
    llm = None


app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ready" if llm else "loading"}

@app.get("/favicon.ico")
def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)


class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.post("/chat")
def chat(req: ChatRequest):
    if llm is None:
        def not_ready():
            yield f"data: {json.dumps({'token': 'Model loading...'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(not_ready(), media_type="text/event-stream")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in req.history:
        if len(turn) == 2:
            messages.append({"role": "user", "content": turn[0]})
            messages.append({"role": "assistant", "content": turn[1]})
    messages.append({"role": "user", "content": req.message})

    def generate():
        try:
            stream = llm.create_chat_completion(
                messages=messages,
                temperature=0.3,
                max_tokens=512,
                stream=True,
            )
            for chunk in stream:
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    yield f"data: {json.dumps({'token': delta})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'token': f'Error: {str(e)}'})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>SurvivorLM</title>

<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500&family=Noto+Serif:wght@400;600&family=Public+Sans:wght@400&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>

<style>
:root {
  --primary: #094cb2;
  --gold: #6d5e00;

  --surface-1: #f8fafc;
  --surface-2: #eef2f7;
  --surface-3: #e3e8ef;
  --surface-4: #d8dee8;

  --text: #0f172a;
}

body {
  margin: 0;
  font-family: Inter, sans-serif;
  background: var(--surface-2);
  color: var(--text);
  display: flex;
  flex-direction: column;
  height: 100vh;
}

/* HEADER */
header {
  backdrop-filter: blur(20px);
  background: rgba(255,255,255,0.8);
  padding: 18px 32px;
  display: flex;
  align-items: center;
  gap: 12px;
  font-family: "Public Sans", sans-serif;
}

header h1 {
  font-family: "Noto Serif", serif;
  font-size: 1.4rem;
  margin: 0;
}

#limit {
  margin-left: auto;
  font-size: 0.75rem;
  opacity: 0.6;
}

/* CHAT */
#chat {
  flex: 1;
  overflow-y: auto;
  padding: 40px;
  display: flex;
  flex-direction: column;
  gap: 22px;
}

.msg {
  max-width: 760px;
  padding: 18px 22px;
  border-radius: 16px;
  line-height: 1.75;
}

/* USER */
.user {
  align-self: flex-end;
  background: linear-gradient(135deg, #094cb2, #3b82f6);
  color: white;
}

/* BOT */
.bot {
  background: var(--surface-1);
}

/* TYPOGRAPHY */
.bot h1, .bot h2, .bot h3 {
  font-family: "Noto Serif", serif;
}

.bot p {
  margin: 10px 0;
}

.bot ul {
  padding-left: 20px;
}

/* INPUT */
footer {
  padding: 24px;
  background: var(--surface-3);
  display: flex;
  gap: 12px;
}

textarea {
  flex: 1;
  border-radius: 12px;
  padding: 14px;
  font-size: 0.95rem;
  border: none;
  outline: none;
  font-family: Inter;
}

textarea:focus {
  box-shadow: 0 0 0 2px var(--primary);
}

/* BUTTON */
button {
  background: linear-gradient(135deg, #094cb2, #2563eb);
  border: none;
  color: white;
  padding: 12px 20px;
  border-radius: 12px;
  cursor: pointer;
  font-weight: 500;
}

button:disabled {
  opacity: 0.5;
}
</style>
</head>

<body>

<header>
  <h1>SurvivalLM</h1>
  <span id="limit"></span>
</header>

<div id="chat"></div>

<footer>
<textarea id="input" rows="2" placeholder="Describe your situation..."></textarea>
<button onclick="send()">Ask</button>
</footer>

<script>
let history = [];

/* LIMIT */
function key(){ return new Date().toISOString().slice(0,10); }
function usage(){ return JSON.parse(localStorage.getItem(key())||"0"); }
function inc(){ localStorage.setItem(key(), usage()+1); update(); }
function update(){ document.getElementById("limit").textContent = usage()+"/10"; }
update();

/* UI */
function add(text, role){
  const d=document.createElement("div");
  d.className="msg "+role;
  d.innerHTML = role==="bot" ? marked.parse(text) : text;
  chat.appendChild(d);
  chat.scrollTop=999999;
  return d;
}

/* SEND */
async function send(){
  if(usage()>=10){ alert("Daily limit reached"); return; }

  const i=document.getElementById("input");
  const msg=i.value.trim();
  if(!msg) return;
  i.value="";

  add(msg,"user");
  const bot=add("...","bot");

  inc();

  const res=await fetch("/chat",{
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({message:msg,history})
  });

  const reader=res.body.getReader();
  const decoder=new TextDecoder();
  let full="";

  while(true){
    const {done,value}=await reader.read();
    if(done) break;

    for(const line of decoder.decode(value).split("\\n")){
      if(line.startsWith("data: ") && line!=="data: [DONE]"){
        const {token}=JSON.parse(line.slice(6));
        full+=token;
        bot.innerHTML = marked.parse(full);
      }
    }
  }

  history.push([msg,full]);
}
</script>

</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML