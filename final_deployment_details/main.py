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
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SurvivorLM</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Lora:ital,wght@0,400;0,600;1,400&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --sand:    #f5f0e8;
    --sand-2:  #ede7d9;
    --sand-3:  #e0d8c8;
    --bark:    #3a2e20;
    --bark-2:  #5c4a32;
    --moss:    #3d5c3a;
    --moss-2:  #527a4e;
    --ember:   #c0392b;
    --amber:   #c97a2a;
    --text:    #1e1810;
    --text-2:  #4a3f30;
    --text-3:  #7a6a55;
    --white:   #fdfaf5;
    --radius:  14px;
    --mono:    'IBM Plex Mono', monospace;
    --serif:   'Lora', Georgia, serif;
    --sans:    'DM Sans', sans-serif;
  }

  html, body {
    height: 100%;
    background: var(--sand);
    color: var(--text);
    font-family: var(--sans);
    font-size: 15px;
    line-height: 1.6;
  }

  /* ── LAYOUT ────────────────────────────── */
  #app {
    display: grid;
    grid-template-rows: auto 1fr auto;
    height: 100vh;
    max-width: 860px;
    margin: 0 auto;
  }

  /* ── HEADER ────────────────────────────── */
  header {
    padding: 20px 28px 18px;
    border-bottom: 1.5px solid var(--sand-3);
    background: var(--white);
    display: flex;
    align-items: center;
    gap: 14px;
  }

  .logo-mark {
    width: 36px;
    height: 36px;
    background: var(--bark);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }

  .logo-mark svg {
    width: 20px;
    height: 20px;
    fill: var(--sand);
  }

  .header-text {
    flex: 1;
  }

  .header-text h1 {
    font-family: var(--serif);
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--bark);
    letter-spacing: -0.01em;
    line-height: 1.2;
  }

  .header-text p {
    font-size: 0.72rem;
    color: var(--text-3);
    font-family: var(--mono);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 2px;
  }

  .header-actions {
    display: flex;
    align-items: center;
    gap: 14px;
  }

  #limit-badge {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--text-3);
    background: var(--sand-2);
    border: 1px solid var(--sand-3);
    padding: 4px 10px;
    border-radius: 20px;
  }

  .hf-link {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--moss);
    text-decoration: none;
    border: 1.5px solid var(--moss-2);
    padding: 5px 11px;
    border-radius: 20px;
    display: flex;
    align-items: center;
    gap: 6px;
    transition: background 0.15s, color 0.15s;
    white-space: nowrap;
  }

  .hf-link:hover {
    background: var(--moss);
    color: var(--white);
  }

  .hf-link svg {
    width: 13px;
    height: 13px;
  }

  /* ── CHAT AREA ─────────────────────────── */
  #chat {
    overflow-y: auto;
    padding: 32px 28px;
    display: flex;
    flex-direction: column;
    gap: 28px;
    background: var(--sand);
  }

  /* Scrollbar */
  #chat::-webkit-scrollbar { width: 5px; }
  #chat::-webkit-scrollbar-track { background: transparent; }
  #chat::-webkit-scrollbar-thumb { background: var(--sand-3); border-radius: 10px; }

  /* Empty state */
  #empty {
    margin: auto;
    text-align: center;
    padding: 40px 20px;
    max-width: 440px;
    animation: fadeUp 0.5s ease both;
  }

  #empty .empty-icon {
    width: 56px;
    height: 56px;
    background: var(--bark);
    border-radius: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 20px;
  }

  #empty .empty-icon svg {
    width: 28px;
    height: 28px;
    fill: var(--sand);
  }

  #empty h2 {
    font-family: var(--serif);
    font-size: 1.35rem;
    color: var(--bark);
    margin-bottom: 10px;
  }

  #empty p {
    font-size: 0.88rem;
    color: var(--text-3);
    line-height: 1.7;
    margin-bottom: 24px;
  }

  .starters {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    text-align: left;
  }

  .starter {
    background: var(--white);
    border: 1.5px solid var(--sand-3);
    border-radius: 10px;
    padding: 12px 14px;
    font-size: 0.82rem;
    color: var(--text-2);
    cursor: pointer;
    transition: border-color 0.15s, box-shadow 0.15s;
    font-family: var(--sans);
    text-align: left;
  }

  .starter:hover {
    border-color: var(--bark-2);
    box-shadow: 0 2px 8px rgba(58,46,32,0.1);
  }

  .starter strong {
    display: block;
    color: var(--bark);
    font-weight: 600;
    margin-bottom: 2px;
    font-size: 0.8rem;
  }

  /* ── MESSAGES ──────────────────────────── */
  .message-row {
    display: flex;
    gap: 14px;
    align-items: flex-start;
    animation: fadeUp 0.3s ease both;
  }

  .message-row.user {
    flex-direction: row-reverse;
  }

  .avatar {
    width: 32px;
    height: 32px;
    border-radius: 9px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 2px;
  }

  .avatar.bot-avatar {
    background: var(--bark);
  }

  .avatar.bot-avatar svg {
    width: 18px;
    height: 18px;
    fill: var(--sand);
  }

  .avatar.user-avatar {
    background: var(--moss);
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--white);
    font-weight: 500;
  }

  .bubble {
    max-width: 680px;
    padding: 16px 20px;
    border-radius: var(--radius);
    line-height: 1.75;
    font-size: 0.93rem;
  }

  .message-row.user .bubble {
    background: var(--bark);
    color: var(--sand);
    border-bottom-right-radius: 4px;
  }

  .message-row.bot .bubble {
    background: var(--white);
    color: var(--text);
    border: 1px solid var(--sand-3);
    border-bottom-left-radius: 4px;
  }

  /* Markdown inside bot bubble */
  .bubble h1, .bubble h2, .bubble h3 {
    font-family: var(--serif);
    color: var(--bark);
    margin: 14px 0 6px;
  }

  .bubble h1 { font-size: 1.1rem; }
  .bubble h2 { font-size: 1rem; }
  .bubble h3 { font-size: 0.95rem; }

  .bubble p { margin: 8px 0; }
  .bubble p:first-child { margin-top: 0; }
  .bubble p:last-child { margin-bottom: 0; }

  .bubble ul, .bubble ol {
    padding-left: 18px;
    margin: 8px 0;
  }

  .bubble li { margin: 4px 0; }

  .bubble strong { color: var(--bark); }

  .bubble code {
    font-family: var(--mono);
    font-size: 0.83em;
    background: var(--sand-2);
    padding: 1px 5px;
    border-radius: 4px;
  }

  /* WARNING highlight */
  .bubble p:has(> strong:first-child) {
    padding: 10px 14px;
    background: #fff4f3;
    border-left: 3px solid var(--ember);
    border-radius: 0 8px 8px 0;
    margin: 10px 0;
  }

  /* Thinking dots */
  .thinking {
    display: flex;
    gap: 5px;
    padding: 6px 4px;
    align-items: center;
  }

  .thinking span {
    width: 7px;
    height: 7px;
    background: var(--sand-3);
    border-radius: 50%;
    animation: pulse 1.2s ease-in-out infinite;
  }

  .thinking span:nth-child(2) { animation-delay: 0.2s; }
  .thinking span:nth-child(3) { animation-delay: 0.4s; }

  @keyframes pulse {
    0%, 80%, 100% { transform: scale(0.8); opacity: 0.4; }
    40% { transform: scale(1.2); opacity: 1; }
  }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  /* ── INPUT AREA ────────────────────────── */
  footer {
    background: var(--white);
    border-top: 1.5px solid var(--sand-3);
    padding: 16px 24px 20px;
  }

  .input-wrap {
    display: flex;
    align-items: flex-end;
    gap: 10px;
    background: var(--sand);
    border: 1.5px solid var(--sand-3);
    border-radius: var(--radius);
    padding: 10px 12px 10px 16px;
    transition: border-color 0.15s, box-shadow 0.15s;
  }

  .input-wrap:focus-within {
    border-color: var(--bark-2);
    box-shadow: 0 0 0 3px rgba(58,46,32,0.08);
  }

  textarea {
    flex: 1;
    border: none;
    background: transparent;
    font-family: var(--sans);
    font-size: 0.93rem;
    color: var(--text);
    resize: none;
    outline: none;
    max-height: 140px;
    line-height: 1.6;
  }

  textarea::placeholder { color: var(--text-3); }

  #send-btn {
    background: var(--bark);
    border: none;
    color: var(--sand);
    width: 36px;
    height: 36px;
    border-radius: 9px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    transition: background 0.15s, transform 0.1s;
  }

  #send-btn:hover:not(:disabled) { background: var(--bark-2); }
  #send-btn:active:not(:disabled) { transform: scale(0.93); }
  #send-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  #send-btn svg { width: 17px; height: 17px; fill: currentColor; }

  .footer-note {
    text-align: center;
    font-size: 0.7rem;
    color: var(--text-3);
    margin-top: 10px;
    font-family: var(--mono);
  }

  /* ── RESPONSIVE ────────────────────────── */
  @media (max-width: 600px) {
    header { padding: 14px 16px; }
    #chat { padding: 20px 16px; }
    footer { padding: 12px 16px 16px; }
    .starters { grid-template-columns: 1fr; }
    .hf-link span { display: none; }
    .header-text p { display: none; }
  }
</style>
</head>
<body>
<div id="app">

  <!-- HEADER -->
  <header>
    <div class="logo-mark">
      <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 2L2 7v10l10 5 10-5V7L12 2zm0 2.18L20 8.5v7L12 19.82 4 15.5v-7L12 4.18zM12 6l-5 2.5v5L12 16l5-2.5v-5L12 6zm0 2l3 1.5v3L12 14l-3-1.5v-3L12 8z"/>
      </svg>
    </div>
    <div class="header-text">
      <h1>SurvivorLM</h1>
      <p>Emergency Field Assistant · Fine-tuned</p>
    </div>
    <div class="header-actions">
      <span id="limit-badge">0 / 10 today</span>
      <a class="hf-link" href="https://huggingface.co/Shreyy2305/survival-gguf/tree/main" target="_blank" rel="noopener">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
          <polyline points="7 10 12 15 17 10"/>
          <line x1="12" y1="15" x2="12" y2="3"/>
        </svg>
        <span>Run offline</span>
      </a>
    </div>
  </header>

  <!-- CHAT -->
  <div id="chat">
    <div id="empty">
      <div class="empty-icon">
        <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 2L2 7v10l10 5 10-5V7L12 2zm0 2.18L20 8.5v7L12 19.82 4 15.5v-7L12 4.18zM12 6l-5 2.5v5L12 16l5-2.5v-5L12 6zm0 2l3 1.5v3L12 14l-3-1.5v-3L12 8z"/>
        </svg>
      </div>
      <h2>What's your situation?</h2>
      <p>Describe your emergency or preparedness question. I'll give you the single most important action first — fast and clear.</p>
      <div class="starters">
        <button class="starter" onclick="fillAndSend(this)">
          <strong>Power outage</strong>
          We've lost power for 3+ hours. What should I do first?
        </button>
        <button class="starter" onclick="fillAndSend(this)">
          <strong>Wilderness lost</strong>
          I'm lost in the woods without cell signal. Help.
        </button>
        <button class="starter" onclick="fillAndSend(this)">
          <strong>Water shortage</strong>
          No running water. How do I make water safe to drink?
        </button>
        <button class="starter" onclick="fillAndSend(this)">
          <strong>Earthquake prep</strong>
          What should be in a 72-hour emergency kit?
        </button>
      </div>
    </div>
  </div>

  <!-- INPUT -->
  <footer>
    <div class="input-wrap">
      <textarea id="input" rows="1" placeholder="Describe your situation or ask a survival question…"
        onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
      <button id="send-btn" onclick="send()" title="Send">
        <svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg>
      </button>
    </div>
    <p class="footer-note">For life-threatening emergencies, call 112 / 911 immediately. &nbsp;·&nbsp; Daily limit: 10 messages</p>
  </footer>

</div>

<script>
marked.setOptions({ breaks: true, gfm: true });

let history = [];
let busy = false;

/* ── USAGE ──────────────────────────────── */
function storageKey() { return 'usage_' + new Date().toISOString().slice(0, 10); }
function getUsage()   { return parseInt(localStorage.getItem(storageKey()) || '0', 10); }
function incUsage()   { localStorage.setItem(storageKey(), getUsage() + 1); updateBadge(); }
function updateBadge(){ document.getElementById('limit-badge').textContent = getUsage() + ' / 10 today'; }
updateBadge();

/* ── AUTO-RESIZE TEXTAREA ───────────────── */
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 140) + 'px';
}

/* ── KEYBOARD SHORTCUT ──────────────────── */
function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
}

/* ── STARTER PROMPTS ────────────────────── */
function fillAndSend(btn) {
  const text = btn.querySelector('strong').nextSibling.textContent.trim();
  document.getElementById('input').value = text;
  autoResize(document.getElementById('input'));
  send();
}

/* ── ADD MESSAGE ────────────────────────── */
function addMessage(text, role) {
  const empty = document.getElementById('empty');
  if (empty) empty.remove();

  const row = document.createElement('div');
  row.className = 'message-row ' + role;

  const avatar = document.createElement('div');
  avatar.className = 'avatar ' + (role === 'bot' ? 'bot-avatar' : 'user-avatar');

  if (role === 'bot') {
    avatar.innerHTML = `<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 2L2 7v10l10 5 10-5V7L12 2zm0 2.18L20 8.5v7L12 19.82 4 15.5v-7L12 4.18zM12 6l-5 2.5v5L12 16l5-2.5v-5L12 6zm0 2l3 1.5v3L12 14l-3-1.5v-3L12 8z"/>
    </svg>`;
  } else {
    avatar.textContent = 'YOU';
  }

  const bubble = document.createElement('div');
  bubble.className = 'bubble';

  if (role === 'user') {
    bubble.textContent = text;
  } else if (text === '__thinking__') {
    bubble.innerHTML = '<div class="thinking"><span></span><span></span><span></span></div>';
  } else {
    bubble.innerHTML = marked.parse(text);
  }

  row.appendChild(avatar);
  row.appendChild(bubble);

  const chat = document.getElementById('chat');
  chat.appendChild(row);
  chat.scrollTop = chat.scrollHeight;

  return bubble;
}

/* ── SEND ───────────────────────────────── */
async function send() {
  if (busy) return;
  if (getUsage() >= 10) { alert('Daily limit of 10 messages reached. Come back tomorrow.'); return; }

  const input = document.getElementById('input');
  const msg = input.value.trim();
  if (!msg) return;

  input.value = '';
  input.style.height = 'auto';
  busy = true;
  document.getElementById('send-btn').disabled = true;

  addMessage(msg, 'user');
  const botBubble = addMessage('__thinking__', 'bot');

  incUsage();

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg, history })
    });

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let full = '';
    let started = false;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value);
      for (const line of chunk.split('\\n')) {
        if (line.startsWith('data: ') && line !== 'data: [DONE]') {
          try {
            const { token } = JSON.parse(line.slice(6));
            full += token;
            if (!started) { started = true; }
            botBubble.innerHTML = marked.parse(full);
            document.getElementById('chat').scrollTop = 999999;
          } catch (_) {}
        }
      }
    }

    history.push([msg, full]);

  } catch (err) {
    botBubble.innerHTML = '<em style="color:var(--ember)">Connection error. Please try again.</em>';
  }

  busy = false;
  document.getElementById('send-btn').disabled = false;
  input.focus();
}
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML