// chbt_nn local UI — vanilla JS, no build, no dependencies.
//
// Talks to the FastAPI picker over /api. Streams assistant tokens via SSE
// (POST + ReadableStream — we don't use EventSource because we need to POST
// the user message body).

const $ = (sel) => document.querySelector(sel);

const state = {
  models: [],
  defaultModel: null,
  ragSubfolders: [],
  ragAvailable: false,
  conversations: [],
  activeId: null,
  streaming: null, // { abort: AbortController }
};

// ---------- API helpers
async function api(path, opts = {}) {
  const r = await fetch(path, {
    headers: { 'content-type': 'application/json' },
    ...opts,
  });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  if (r.status === 204) return null;
  return r.json();
}

async function loadHealth() {
  try {
    const h = await api('/api/health');
    const cls = h.ollama ? 'ok' : 'bad';
    const ragCls = h.rag ? 'ok' : 'warn';
    $('#health').innerHTML =
      `ollama <span class="${cls}">${h.ollama ? 'up' : 'down'}</span> · ` +
      `rag <span class="${ragCls}">${h.rag ? 'up' : 'off'}</span> · ` +
      `<span class="muted">${h.expose}</span>`;
  } catch (e) {
    $('#health').textContent = 'picker: api error';
  }
}

async function loadModels() {
  const data = await api('/api/models');
  state.models = data.models;
  state.defaultModel = data.default;
  const sel = $('#model-select');
  sel.innerHTML = '';
  for (const m of data.models) {
    const opt = document.createElement('option');
    opt.value = m.name;
    let label = m.name;
    if (m.is_finetune) label += ' ✦';
    if (m.roles && m.roles.length) label += `  · ${m.roles.join(',')}`;
    opt.textContent = label;
    sel.appendChild(opt);
  }
  if (!data.models.length) {
    const opt = document.createElement('option');
    opt.textContent = '(no models — run infra/bootstrap.sh)';
    opt.disabled = true;
    sel.appendChild(opt);
  }
}

async function loadRagSubfolders() {
  const data = await api('/api/rag/subfolders');
  state.ragSubfolders = data.subfolders;
  state.ragAvailable = data.available;
  const sel = $('#rag-filter');
  sel.innerHTML = '';
  for (const s of data.subfolders) {
    const opt = document.createElement('option');
    opt.value = s; opt.textContent = s;
    sel.appendChild(opt);
  }
  $('#rag-toggle').disabled = !data.available;
  $('#rag-filter').disabled = !data.available;
}

async function loadConversations() {
  const data = await api('/api/conversations');
  state.conversations = data.conversations;
  renderConvList();
}

function renderConvList() {
  const ul = $('#conv-list');
  ul.innerHTML = '';
  for (const c of state.conversations) {
    const li = document.createElement('li');
    li.dataset.id = c.id;
    if (c.id === state.activeId) li.classList.add('active');
    const title = document.createElement('div');
    title.textContent = c.title || '(untitled)';
    const meta = document.createElement('span');
    meta.className = 'meta';
    meta.textContent = c.model + (c.rag_enabled ? ' · rag' : '');
    li.appendChild(title);
    li.appendChild(meta);
    li.addEventListener('click', () => activate(c.id));
    ul.appendChild(li);
  }
  if (!state.conversations.length) {
    const li = document.createElement('li');
    li.className = 'muted';
    li.textContent = '(no conversations yet)';
    ul.appendChild(li);
  }
}

// ---------- conversation actions
async function newConversation() {
  const conv = await api('/api/conversations', {
    method: 'POST',
    body: JSON.stringify({ title: 'New conversation', model: $('#model-select').value || undefined }),
  });
  state.conversations.unshift(conv);
  await activate(conv.id);
  renderConvList();
}

async function activate(id) {
  state.activeId = id;
  const conv = state.conversations.find(c => c.id === id) || await api(`/api/conversations/${id}`);
  $('#conv-title').value = conv.title;
  $('#model-select').value = conv.model;
  $('#rag-toggle').checked = !!conv.rag_enabled;
  // multi-select sync
  const sel = $('#rag-filter');
  for (const opt of sel.options) {
    opt.selected = (conv.rag_filter || []).includes(opt.value);
  }
  renderConvList();
  await loadMessages(id);
}

async function loadMessages(id) {
  const data = await api(`/api/conversations/${id}/messages`);
  const box = $('#messages');
  box.innerHTML = '';
  if (!data.messages.length) {
    const div = document.createElement('div');
    div.className = 'empty';
    div.textContent = 'send a message to start.';
    box.appendChild(div);
    return;
  }
  for (const m of data.messages) renderMessage(m);
  box.scrollTop = box.scrollHeight;
}

function renderMessage(m) {
  const box = $('#messages');
  // Skip the auto-injected RAG system messages from old transcripts.
  if (m.role === 'system') return null;
  const div = document.createElement('div');
  div.className = `msg ${m.role}`;
  const role = document.createElement('div');
  role.className = 'role';
  role.textContent = m.role + (m.model ? ` · ${m.model}` : '');
  const content = document.createElement('div');
  content.className = 'content';
  content.textContent = m.content;
  div.appendChild(role);
  div.appendChild(content);
  if (m.sources && m.sources.length) {
    const s = document.createElement('div');
    s.className = 'sources';
    s.innerHTML = 'sources: ' + m.sources.map(x => `<code>${escapeHtml(x)}</code>`).join(', ');
    div.appendChild(s);
  }
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
  return content;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
}

// ---------- header edits
async function persistHeader() {
  if (!state.activeId) return;
  const sel = $('#rag-filter');
  const filter = Array.from(sel.selectedOptions).map(o => o.value);
  const body = {
    title: $('#conv-title').value,
    model: $('#model-select').value,
    rag_enabled: $('#rag-toggle').checked,
    rag_filter: filter,
  };
  const conv = await api(`/api/conversations/${state.activeId}`, {
    method: 'PATCH', body: JSON.stringify(body),
  });
  const idx = state.conversations.findIndex(c => c.id === conv.id);
  if (idx >= 0) state.conversations[idx] = conv;
  renderConvList();
}

async function deleteActive() {
  if (!state.activeId) return;
  if (!confirm('Delete this conversation?')) return;
  await api(`/api/conversations/${state.activeId}`, { method: 'DELETE' });
  state.conversations = state.conversations.filter(c => c.id !== state.activeId);
  state.activeId = null;
  $('#messages').innerHTML = '';
  $('#conv-title').value = '';
  renderConvList();
}

// ---------- send + stream
async function send() {
  const ta = $('#composer-input');
  const content = ta.value.trim();
  if (!content) return;
  if (!state.activeId) await newConversation();
  ta.value = '';
  ta.disabled = true;
  $('#send').disabled = true;
  $('#stop').hidden = false;

  // Render user msg immediately.
  renderMessage({ role: 'user', content });
  const asstContent = renderMessage({ role: 'assistant', content: '', model: $('#model-select').value });
  let asstText = '';

  const ctrl = new AbortController();
  state.streaming = { abort: ctrl };

  try {
    const r = await fetch(`/api/conversations/${state.activeId}/messages`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ content }),
      signal: ctrl.signal,
    });
    if (!r.ok || !r.body) throw new Error(`stream error: ${r.status}`);

    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      // SSE frames are separated by blank lines.
      let idx;
      while ((idx = buf.indexOf('\n\n')) >= 0) {
        const frame = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const dataLine = frame.split('\n').find(l => l.startsWith('data:'));
        if (!dataLine) continue;
        let evt;
        try { evt = JSON.parse(dataLine.slice(5).trim()); } catch { continue; }
        if (evt.event === 'token') {
          asstText += evt.delta;
          asstContent.textContent = asstText;
          $('#messages').scrollTop = $('#messages').scrollHeight;
        } else if (evt.event === 'start' && evt.sources && evt.sources.length) {
          const wrap = asstContent.parentElement;
          const s = document.createElement('div');
          s.className = 'sources';
          s.innerHTML = 'sources: ' + evt.sources.map(x => `<code>${escapeHtml(x)}</code>`).join(', ');
          wrap.appendChild(s);
        } else if (evt.event === 'error') {
          asstText += `\n\n[error: ${evt.message}]`;
          asstContent.textContent = asstText;
        }
      }
    }
  } catch (e) {
    if (e.name !== 'AbortError') {
      asstContent.textContent = (asstText || '') + `\n[fetch error: ${e.message}]`;
    }
  } finally {
    state.streaming = null;
    ta.disabled = false;
    $('#send').disabled = false;
    $('#stop').hidden = true;
    ta.focus();
    // Refresh conv list so updated_at sorting is correct.
    loadConversations();
  }
}

function stop() {
  if (state.streaming) state.streaming.abort.abort();
}

// ---------- wire-up
function init() {
  $('#new-conv').addEventListener('click', newConversation);
  $('#delete-conv').addEventListener('click', deleteActive);
  $('#composer').addEventListener('submit', (e) => { e.preventDefault(); send(); });
  $('#stop').addEventListener('click', stop);
  $('#composer-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });
  for (const id of ['#conv-title', '#model-select', '#rag-toggle', '#rag-filter']) {
    $(id).addEventListener('change', persistHeader);
  }
  $('#conv-title').addEventListener('blur', persistHeader);

  Promise.all([loadHealth(), loadModels(), loadRagSubfolders(), loadConversations()])
    .then(() => {
      if (state.conversations.length) activate(state.conversations[0].id);
    })
    .catch(err => console.error('init', err));

  // Periodic health refresh.
  setInterval(loadHealth, 10000);
}

document.addEventListener('DOMContentLoaded', init);
