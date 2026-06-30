"""
DocuQuery v4.4 — Chainlit UI

New in this version:
  - Developer mode toggle (menu option) showing:
      live server logs, token usage meter, retrieval scores, timing
  - Hallucination risk badge on every answer (🟢 low / 🟡 medium / 🔴 high)
  - Prompt injection risk indicator (dev mode only — shown when detected)
  - Cumulative session token usage meter (dev mode)
  - 🔄 Rebuild Images button in Documents panel (recovery tool)
"""

import chainlit as cl
import httpx
import json
import os
import warnings
import logging

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("anyio").setLevel(logging.ERROR)

BASE_URL   = os.getenv("BASE_URL",   "http://127.0.0.1:8000")
IMAGES_DIR = os.getenv("IMAGES_DIR", "data/images")

# ── Auth ──────────────────────────────────────────────────────────────────────

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    USERS = {
        os.getenv("DOCUQUERY_USER", "admin"): os.getenv("DOCUQUERY_PASS", "docuquery2024"),
    }
    if USERS.get(username) == password:
        return cl.User(identifier=username, metadata={"role": "user"})
    return None

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_action(name: str, label: str, payload: dict = None) -> cl.Action:
    return cl.Action(name=name, label=label, payload=payload or {"action": name})

def zone_label(alpha) -> str:
    if alpha is None:   return "🤖 Auto"
    elif alpha <= 0.33: return f"Zone 1 🔤 Keyword (α={alpha})"
    elif alpha <= 0.66: return f"Zone 2 ⚖️ Balanced (α={alpha})"
    else:               return f"Zone 3 🧠 Semantic (α={alpha})"

def zone_desc(alpha) -> str:
    if alpha is None:   return "auto"
    elif alpha <= 0.33: return "keyword-focused"
    elif alpha <= 0.66: return "balanced"
    else:               return "semantic-focused"

def _shorten(name: str, max_len: int = 28) -> str:
    name = name.replace(".pdf","").replace(".docx","").replace(".pptx","") \
               .replace(".md","").replace(".txt","").replace("_"," ")
    return name[:max_len] + "…" if len(name) > max_len else name

HALLUC_BADGE = {"low": "🟢", "medium": "🟡", "high": "🔴", None: "⚪"}
INJECTION_BADGE = {"none": "🟢", "low": "🟡", "medium": "🟠", "high": "🔴"}


def build_footer(metadata: dict, alpha, dev_mode: bool = False):
    sources    = metadata.get("sources",         [])
    confidence = metadata.get("confidence",       "unknown")
    docs_n     = metadata.get("docs_searched",    0)
    alpha_used = metadata.get("alpha_used",       alpha)
    alpha_mode = metadata.get("alpha_mode",       "")
    sub_q      = metadata.get("sub_queries_used", [])
    halluc     = metadata.get("hallucination_risk", None)
    halluc_note = metadata.get("hallucination_note", "")
    injection  = metadata.get("injection_risk", "none")
    tokens     = metadata.get("token_usage", {})

    conf_badge = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(confidence, "⚪")
    warn       = "\n> ⚠️ **Low confidence** — answer may be incomplete. Try rephrasing." if confidence == "low" else ""
    mode_label = "🤖 auto" if "auto" in str(alpha_mode) else "⚙️ manual"
    mode_desc  = zone_desc(alpha_used)
    sub_q_text = f"\n*🔀 Sub-queries: {len(sub_q)}*" if len(sub_q) > 1 else ""

    halluc_badge = HALLUC_BADGE.get(halluc, "⚪")
    halluc_line  = f" · {halluc_badge} Hallucination risk: **{halluc}**" if halluc else ""

    status_bar = (
        f"{warn}"
        f"{sub_q_text}\n\n"
        f"{conf_badge} **Confidence:** {confidence} · "
        f"**Docs searched:** {docs_n} · "
        f"{mode_label} (α={alpha_used}) · **{mode_desc}**"
        f"{halluc_line}"
    )

    # Developer mode adds a technical detail block
    if dev_mode:
        inj_badge = INJECTION_BADGE.get(injection, "⚪")
        status_bar += (
            f"\n\n---\n**🛠️ Developer Mode**\n"
            f"- {inj_badge} Injection risk: `{injection}`"
            + (f" — patterns: `{', '.join(metadata.get('injection_patterns', []))}`"
               if metadata.get('injection_patterns') else "")
            + f"\n- {halluc_badge} Hallucination note: _{halluc_note}_\n"
            f"- 🔢 Tokens — prompt: `{tokens.get('prompt_tokens',0)}` · "
            f"completion: `{tokens.get('completion_tokens',0)}` · "
            f"total: `{tokens.get('total_tokens',0)}`"
        )

    refs_text   = ""
    image_count = 0
    if sources:
        refs_text = "\n\n---\n**📄 Source details:**\n"
        for i, s in enumerate(sources, 1):
            rerank = f" · rerank: `{s['rerank_score']:.2f}`" if s.get("rerank_score") else ""
            refs_text += (
                f"**[{i}] {s['filename']}** — page {s['page']} "
                f"· score: `{s['relevance_score']}`{rerank}\n"
                f"> {s['preview']}\n\n"
            )
            imgs = s.get("images", [])
            if imgs:
                image_count += len(imgs)
                refs_text += f"  🖼️ *{len(imgs)} figure(s) on this page — shown below*\n\n"

    return status_bar, refs_text, image_count


async def _send_source_images(sources: list):
    elements = []
    seen_ids = set()
    for s in sources:
        for img in s.get("images", []):
            image_id = img.get("image_id") or img.get("id")
            if not image_id or image_id in seen_ids:
                continue
            seen_ids.add(image_id)

            doc_id    = img.get("doc_id", s.get("doc_id", ""))
            ext       = img.get("ext", "png")
            disk_path = img.get("path") or os.path.join(IMAGES_DIR, doc_id, f"{image_id}.{ext}")

            if not os.path.exists(disk_path):
                for root, _, files in os.walk(IMAGES_DIR):
                    for f in files:
                        if f.startswith(image_id):
                            disk_path = os.path.join(root, f)
                            break

            if not os.path.exists(disk_path):
                continue

            caption = f"{s.get('filename','Doc')} — page {img.get('page', s.get('page','?'))}"
            elements.append(cl.Image(path=disk_path, name=caption, display="inline"))

    if elements:
        await cl.Message(
            content  = f"🖼️ **Figures from cited pages** ({len(elements)} image(s)):",
            elements = elements
        ).send()

# ── Start ─────────────────────────────────────────────────────────────────────

@cl.on_chat_start
async def start():
    user = cl.user_session.get("user")
    name = user.identifier if user else "there"

    cl.user_session.set("doc_ids",             [])
    cl.user_session.set("history",             [])
    cl.user_session.set("alpha",               None)
    cl.user_session.set("suggested_questions", {})
    cl.user_session.set("ref_store",           {})
    cl.user_session.set("msg_counter",         0)
    cl.user_session.set("last_query",          "")
    cl.user_session.set("last_answer",         "")
    cl.user_session.set("last_sources",        [])
    cl.user_session.set("dev_mode",            False)
    cl.user_session.set("session_tokens", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

    await cl.Message(
        content=(
            f"# Welcome to DocuQuery, {name} 👋\n\n"
            "**Intelligent document Q&A — RAG-powered with hybrid search**\n\n"
            "---\n\n"
            "**Quick start:**\n"
            "1. Click **📎** below to upload PDFs or documents\n"
            "2. Review the auto-generated summary and suggested questions\n"
            "3. Ask any question — results cite exact pages\n"
            "4. Click **📄 Show Sources** to verify every answer\n"
            "5. Rate with **👍 / 👎** to track quality\n\n"
            "---\n\n"
            "**Search mode:** 🤖 Auto *(LLM picks the best mode per query)*"
        ),
        actions=[
            make_action("btn_docs",     "📄 Documents"),
            make_action("btn_mode",     "🔍 Search Zone"),
            make_action("btn_quality",  "📊 Quality"),
            make_action("btn_devmode",  "🛠️ Developer Mode"),
            make_action("btn_help",     "❓ Help"),
        ]
    ).send()

# ── Messages ──────────────────────────────────────────────────────────────────

@cl.on_message
async def main(message: cl.Message):
    doc_ids  = cl.user_session.get("doc_ids", [])
    history  = cl.user_session.get("history", [])
    alpha    = cl.user_session.get("alpha",   None)
    dev_mode = cl.user_session.get("dev_mode", False)

    if message.content.startswith("/"):
        await handle_command(message.content.strip(), doc_ids, alpha)
        return

    if message.elements:
        await handle_uploads(message.elements, doc_ids)
        return

    msg = cl.Message(content="")
    await msg.send()
    history.append({"role": "user", "content": message.content})

    full_text, metadata = "", {}

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST", f"{BASE_URL}/ask/stream",
                json={"query": message.content, "doc_ids": doc_ids,
                      "alpha": alpha, "history": history[:-1]}
            ) as response:
                if response.status_code == 429:
                    msg.content = "⏱️ **Rate limit reached.** Please wait a moment."
                    await msg.update()
                    return
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data  = json.loads(data_str)
                        token = data.get("token", "")
                        if "__METADATA__" in token:
                            try:
                                metadata = json.loads(token.split("__METADATA__", 1)[1])
                            except Exception:
                                pass
                        else:
                            full_text  += token
                            msg.content = full_text
                            await msg.update()
                    except Exception:
                        pass

        cl.user_session.set("last_query",   message.content)
        cl.user_session.set("last_answer",  full_text)
        cl.user_session.set("last_sources", metadata.get("sources", []))

        # Accumulate session token usage (for dev mode meter)
        tok = metadata.get("token_usage", {})
        sess_tok = cl.user_session.get("session_tokens", {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0})
        sess_tok["prompt_tokens"]     += tok.get("prompt_tokens", 0)
        sess_tok["completion_tokens"] += tok.get("completion_tokens", 0)
        sess_tok["total_tokens"]      += tok.get("total_tokens", 0)
        cl.user_session.set("session_tokens", sess_tok)

        counter = cl.user_session.get("msg_counter", 0)
        slot    = str(counter % 10)
        cl.user_session.set("msg_counter", counter + 1)

        status_bar, refs_text, image_count = build_footer(metadata, alpha, dev_mode)
        ref_store = cl.user_session.get("ref_store", {})
        ref_store[slot] = {
            "full_text":  full_text,
            "status_bar": status_bar,
            "refs_text":  refs_text,
            "expanded":   False,
        }
        cl.user_session.set("ref_store", ref_store)

        msg.content = full_text + "\n\n---\n" + status_bar
        msg.actions = [
            make_action(f"tr_{slot}",        "📄 Show Sources",  payload={"slot": slot}),
            make_action("btn_feedback_up",   "👍 Helpful",       payload={"slot": slot}),
            make_action("btn_feedback_down", "👎 Not helpful",   payload={"slot": slot}),
            make_action("btn_docs",          "📄 Documents"),
            make_action("btn_mode",          "🔍 Change Zone"),
        ]
        await msg.update()

        # ── Image display logic ──────────────────────────────────────────────
        sources    = metadata.get("sources", [])
        confidence = metadata.get("confidence", "unknown")
        has_images = any(s.get("images") for s in sources)

        if has_images:
            await _send_source_images(sources)
        elif image_count > 0 and confidence in ("high", "medium") and sources:
            await _send_source_images(sources)

        history.append({"role": "assistant", "content": full_text})
        cl.user_session.set("history", history[-20:])

    except Exception as e:
        msg.content = (
            f"❌ **Connection error:** {str(e)}\n\n"
            "Make sure the API server is running:\n"
            "```\nuvicorn main:app --reload --port 8000\n```"
        )
        await msg.update()

# ── Feedback ──────────────────────────────────────────────────────────────────

async def _submit_feedback(rating: str):
    query   = cl.user_session.get("last_query",   "")
    answer  = cl.user_session.get("last_answer",  "")
    sources = cl.user_session.get("last_sources", [])
    if not query:
        await cl.Message(content="No recent answer to rate.").send()
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(f"{BASE_URL}/feedback/", json={
                "query": query, "answer": answer, "rating": rating, "sources": sources
            })
        icon = "👍" if rating == "up" else "👎"
        await cl.Message(
            content=f"{icon} **Feedback recorded — thank you!**",
            actions=[make_action("btn_quality", "📊 View Quality Report")]
        ).send()
    except Exception as e:
        await cl.Message(content=f"Feedback submission failed: {e}").send()

@cl.action_callback("btn_feedback_up")
async def on_feedback_up(a): await _submit_feedback("up")

@cl.action_callback("btn_feedback_down")
async def on_feedback_down(a): await _submit_feedback("down")

# ── Quality Report ────────────────────────────────────────────────────────────

@cl.action_callback("btn_quality")
async def on_quality(action: cl.Action):
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            eval_res = (await client.get(f"{BASE_URL}/eval-log/", params={"limit": 10})).json()
            feed_res = (await client.get(f"{BASE_URL}/feedback/")).json()

        avgs   = eval_res.get("averages", {})
        recent = eval_res.get("recent",   [])
        total  = eval_res.get("total_evals", 0)
        faith  = avgs.get("faithfulness")
        relev  = avgs.get("answer_relevancy")

        def pct(v): return f"{v:.0%}" if v is not None else "—"
        def bar(v):
            if v is None: return "░░░░░░░░░░"
            return "█" * int(v * 10) + "░" * (10 - int(v * 10))

        thumbs_up   = feed_res.get("thumbs_up",   0)
        thumbs_down = feed_res.get("thumbs_down", 0)
        total_feed  = feed_res.get("total",        0)

        report = (
            "## 📊 Quality Report\n\n"
            "**RAGAS-style Evaluation** *(LLM-as-judge, after each answer)*\n\n"
            f"```\nFaithfulness     {bar(faith)} {pct(faith)}\n"
            f"Answer Relevancy {bar(relev)} {pct(relev)}\n"
            f"Total evals logged: {total}\n```\n\n"
            "**User Feedback**\n\n"
            f"👍 Helpful: **{thumbs_up}** · 👎 Not helpful: **{thumbs_down}** · Total: {total_feed}\n"
        )
        if recent:
            report += "\n**Recent Evaluations:**\n"
            for e in reversed(recent[-5:]):
                q = (e.get("query") or "")[:55]
                report += f"- `{q}...` → faith: {pct(e.get('faithfulness'))} · relev: {pct(e.get('answer_relevancy'))}\n"

        await cl.Message(
            content=report,
            actions=[make_action("btn_docs", "📄 Documents"), make_action("btn_help", "❓ Help")]
        ).send()
    except Exception as e:
        await cl.Message(content=f"Quality report unavailable: {e}").send()

# ── Developer Mode (NEW) ─────────────────────────────────────────────────────

@cl.action_callback("btn_devmode")
async def on_devmode_menu(action: cl.Action):
    dev_mode = cl.user_session.get("dev_mode", False)
    await cl.Message(
        content=(
            f"## 🛠️ Developer Mode\n\n"
            f"Currently: **{'ON ✅' if dev_mode else 'OFF ⬜'}**\n\n"
            "When ON, every answer shows: injection risk score, hallucination "
            "rationale, and per-query token usage in addition to the normal footer."
        ),
        actions=[
            make_action("devmode_on",  "✅ Turn ON"),
            make_action("devmode_off", "⬜ Turn OFF"),
            make_action("btn_dev_logs",  "📜 View Server Logs"),
            make_action("btn_dev_stats", "📈 View Stats"),
            make_action("btn_dev_tokens","🔢 Session Token Usage"),
        ]
    ).send()

@cl.action_callback("devmode_on")
async def on_devmode_on(a):
    cl.user_session.set("dev_mode", True)
    await cl.Message(content="✅ **Developer mode ON** — answers will now show injection risk, "
                              "hallucination rationale, and token usage.",
                     actions=[make_action("btn_devmode", "🛠️ Developer Mode")]).send()

@cl.action_callback("devmode_off")
async def on_devmode_off(a):
    cl.user_session.set("dev_mode", False)
    await cl.Message(content="⬜ **Developer mode OFF.**",
                     actions=[make_action("btn_devmode", "🛠️ Developer Mode")]).send()

@cl.action_callback("btn_dev_logs")
async def on_dev_logs(a):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            res = (await client.get(f"{BASE_URL}/dev/logs/", params={"limit": 30})).json()
        logs = res.get("logs", [])
        if not logs:
            content = "No logs captured yet."
        else:
            lines = [f"`{l['timestamp']}` **{l['level']}** [{l['logger']}] {l['message']}" for l in logs]
            content = "## 📜 Server Logs (last 30)\n\n" + "\n\n".join(lines)
        await cl.Message(
            content=content,
            actions=[make_action("btn_dev_logs", "🔄 Refresh"), make_action("btn_devmode", "🛠️ Developer Mode")]
        ).send()
    except Exception as e:
        await cl.Message(content=f"Could not fetch logs: {e}").send()

@cl.action_callback("btn_dev_stats")
async def on_dev_stats(a):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            res = (await client.get(f"{BASE_URL}/dev/stats/")).json()
        docs   = res.get("documents", {})
        feed   = res.get("feedback", {})
        evalq  = res.get("eval_quality", {})
        health = docs.get("health_breakdown", {})
        health_str = " · ".join(f"{k}:{v}" for k, v in health.items()) or "none"

        content = (
            "## 📈 Developer Stats\n\n"
            f"**Documents:** {docs.get('total',0)} total · health: {health_str}\n\n"
            f"**Feedback:** 👍 {feed.get('thumbs_up',0)} · 👎 {feed.get('thumbs_down',0)} · total {feed.get('total',0)}\n\n"
            f"**Eval Quality:** {evalq.get('total_evals',0)} evals · "
            f"faithfulness: {evalq.get('avg_faithfulness','—')} · "
            f"relevancy: {evalq.get('avg_answer_relevancy','—')}\n\n"
            f"**Log buffer:** {res.get('log_buffer_size',0)} entries"
        )
        await cl.Message(
            content=content,
            actions=[make_action("btn_dev_stats", "🔄 Refresh"), make_action("btn_devmode", "🛠️ Developer Mode")]
        ).send()
    except Exception as e:
        await cl.Message(content=f"Could not fetch stats: {e}").send()

@cl.action_callback("btn_dev_tokens")
async def on_dev_tokens(a):
    sess = cl.user_session.get("session_tokens", {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0})
    content = (
        "## 🔢 Session Token Usage\n\n"
        f"Prompt tokens: **{sess['prompt_tokens']:,}**\n\n"
        f"Completion tokens: **{sess['completion_tokens']:,}**\n\n"
        f"**Total: {sess['total_tokens']:,}**\n\n"
        "*Resets when you start a new chat session.*"
    )
    await cl.Message(
        content=content,
        actions=[make_action("btn_devmode", "🛠️ Developer Mode")]
    ).send()

# ── Reference toggle ──────────────────────────────────────────────────────────

async def _toggle_refs(slot: str):
    ref_store = cl.user_session.get("ref_store", {})
    if slot not in ref_store:
        return
    state    = ref_store[slot]
    expanded = state["expanded"]
    new_label   = "📄 Hide Sources" if not expanded else "📄 Show Sources"
    new_content = (
        state["full_text"] + "\n\n---\n" + state["status_bar"] +
        (state["refs_text"] if not expanded else "")
    )
    ref_store[slot]["expanded"] = not expanded
    cl.user_session.set("ref_store", ref_store)
    await cl.Message(
        content=new_content,
        actions=[
            make_action(f"tr_{slot}",        new_label, payload={"slot": slot}),
            make_action("btn_feedback_up",   "👍 Helpful"),
            make_action("btn_feedback_down", "👎 Not helpful"),
            make_action("btn_docs",          "📄 Documents"),
            make_action("btn_mode",          "🔍 Change Zone"),
        ]
    ).send()

@cl.action_callback("tr_0")
async def tr0(a): await _toggle_refs(a.payload.get("slot", "0"))
@cl.action_callback("tr_1")
async def tr1(a): await _toggle_refs(a.payload.get("slot", "1"))
@cl.action_callback("tr_2")
async def tr2(a): await _toggle_refs(a.payload.get("slot", "2"))
@cl.action_callback("tr_3")
async def tr3(a): await _toggle_refs(a.payload.get("slot", "3"))
@cl.action_callback("tr_4")
async def tr4(a): await _toggle_refs(a.payload.get("slot", "4"))
@cl.action_callback("tr_5")
async def tr5(a): await _toggle_refs(a.payload.get("slot", "5"))
@cl.action_callback("tr_6")
async def tr6(a): await _toggle_refs(a.payload.get("slot", "6"))
@cl.action_callback("tr_7")
async def tr7(a): await _toggle_refs(a.payload.get("slot", "7"))
@cl.action_callback("tr_8")
async def tr8(a): await _toggle_refs(a.payload.get("slot", "8"))
@cl.action_callback("tr_9")
async def tr9(a): await _toggle_refs(a.payload.get("slot", "9"))

# ── Uploads ───────────────────────────────────────────────────────────────────

async def handle_uploads(elements, doc_ids: list):
    processing = cl.Message(content=f"⏳ Uploading {len(elements)} file(s)...")
    await processing.send()

    files = [
        ("files", (el.name, open(el.path, "rb"), "application/octet-stream"))
        for el in elements if hasattr(el, "path") and el.path
    ]
    if not files:
        processing.content = "No valid files found."
        await processing.update()
        return

    try:
        async with httpx.AsyncClient(timeout=600) as client:
            res = await client.post(f"{BASE_URL}/upload/", files=files)
        if res.status_code == 429:
            processing.content = "⏱️ **Upload limit reached.** You can upload up to 5 documents per hour."
            await processing.update()
            return

        data     = res.json()
        uploaded = data.get("results",      [])
        failed   = data.get("errors",       [])
        skipped  = data.get("skipped_docs", [])
        total    = data.get("total_indexed", 0)

        for r in uploaded:
            if r["doc_id"] not in doc_ids:
                doc_ids.append(r["doc_id"])
        cl.user_session.set("doc_ids", doc_ids)

        msg = ""
        for r in uploaded:
            icon  = {"excellent": "🟢", "good": "🟡", "fair": "🟠", "poor": "🔴"}.get(r.get("document_health",""), "⚪")
            imgs  = r.get("images_extracted", 0)
            img_n = f" · 🖼️ {imgs} image(s)" if imgs else ""
            msg  += (
                f"✅ **{r['filename']}**\n"
                f"  `{r['doc_id']}` · {r.get('file_size_mb','?')} MB · "
                f"{r['chunks']} chunks · {r['pages_detected']} pages · "
                f"{icon} {r['document_health']}{img_n}\n\n"
            )
        for s in skipped:
            msg += f"⏭️ **{s['filename']}** — already indexed\n\n"
        for f in failed:
            msg += f"❌ **{f['filename']}** — {f['error']}\n\n"
        if uploaded or skipped:
            msg += f"📚 **{total} document(s) total in index**"

        processing.content = msg
        await processing.update()

        for r in uploaded:
            analysing = cl.Message(content=f"🔍 Analysing **{r['filename']}**...")
            await analysing.send()
            try:
                async with httpx.AsyncClient(timeout=90) as client:
                    ana = (await client.post(
                        f"{BASE_URL}/upload/analyse",
                        json={"doc_id": r["doc_id"], "filename": r["filename"],
                              "chunks_preview": r.get("chunks_preview", [])}
                    )).json()

                summary   = ana.get("summary",   [])
                questions = ana.get("questions", [])
                sum_text  = ""
                if summary:
                    sum_text = f"**📋 {r['filename']}:**\n"
                    for i, pt in enumerate(summary, 1):
                        sum_text += f"{i}. {pt}\n"

                sq_store = cl.user_session.get("suggested_questions", {})
                actions  = []
                for i, q in enumerate(questions[:5]):
                    sq_store[f"sq_{i}"] = q
                    actions.append(make_action(f"sq_{i}", f"❓ {q[:60]}", payload={"question": q}))
                cl.user_session.set("suggested_questions", sq_store)
                actions += [make_action("btn_docs", "📄 Documents"), make_action("btn_mode", "🔍 Search Zone")]

                analysing.content = sum_text + ("\n\n**💡 Try asking:**" if questions else "")
                analysing.actions = actions
                await analysing.update()
            except Exception as e:
                analysing.content = f"✅ **{r['filename']}** ready. *(Analysis failed: {e})*"
                await analysing.update()

    except Exception as e:
        processing.content = f"❌ **Upload failed:** {str(e)}"
        await processing.update()

# ── Suggested questions ───────────────────────────────────────────────────────

@cl.action_callback("sq_0")
async def sq0(a):
    q = a.payload.get("question") or cl.user_session.get("suggested_questions", {}).get("sq_0", "")
    if q: await main(cl.Message(content=q))

@cl.action_callback("sq_1")
async def sq1(a):
    q = a.payload.get("question") or cl.user_session.get("suggested_questions", {}).get("sq_1", "")
    if q: await main(cl.Message(content=q))

@cl.action_callback("sq_2")
async def sq2(a):
    q = a.payload.get("question") or cl.user_session.get("suggested_questions", {}).get("sq_2", "")
    if q: await main(cl.Message(content=q))

@cl.action_callback("sq_3")
async def sq3(a):
    q = a.payload.get("question") or cl.user_session.get("suggested_questions", {}).get("sq_3", "")
    if q: await main(cl.Message(content=q))

@cl.action_callback("sq_4")
async def sq4(a):
    q = a.payload.get("question") or cl.user_session.get("suggested_questions", {}).get("sq_4", "")
    if q: await main(cl.Message(content=q))

# ── Document management ───────────────────────────────────────────────────────

@cl.action_callback("btn_docs")
async def on_docs(action: cl.Action):
    await _render_doc_list()

async def _render_doc_list():
    doc_ids = cl.user_session.get("doc_ids", [])
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            res = await client.get(f"{BASE_URL}/documents/")
        data  = res.json()
        docs  = data.get("documents", [])
        total = data.get("total", 0)

        if not docs:
            await cl.Message(
                content="No documents indexed yet. Upload one using the 📎 button.",
                actions=[make_action("btn_help", "❓ Help")]
            ).send()
            return

        n_selected = len(doc_ids)
        header = (
            f"📚 **{total} doc(s) indexed · {n_selected} selected**\n"
            "*Tap a document button to select/deselect it.*"
        )

        actions = []
        for i, d in enumerate(docs[:10]):
            pad    = str(i).zfill(4)
            icon   = {"excellent":"🟢","good":"🟡","fair":"🟠","poor":"🔴"}.get(d.get("health",""), "⚪")
            state  = "✅" if d["doc_id"] in doc_ids else "➕"
            name   = _shorten(d["filename"], 28)
            chunks = d.get("chunks", 0)
            label  = f"{icon} {name} · {chunks}ch · {state}"
            actions.append(make_action(
                f"toggle_{pad}", label,
                payload={"doc_id": d["doc_id"], "filename": d["filename"]}
            ))

        actions += [
            make_action("btn_select_all", f"✅ Select All ({total})"),
            make_action("btn_clear",      "🗑️ Clear All"),
            make_action("btn_rebuild_img","🔄 Rebuild Images"),
            make_action("btn_mode",       "🔍 Change Zone"),
        ]

        await cl.Message(content=header, actions=actions).send()

    except Exception as e:
        await cl.Message(content=f"Could not fetch documents: {e}").send()


@cl.action_callback("btn_select_all")
async def on_select_all(action: cl.Action):
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            res = await client.get(f"{BASE_URL}/documents/")
        docs    = res.json().get("documents", [])
        doc_ids = [d["doc_id"] for d in docs]
        cl.user_session.set("doc_ids", doc_ids)
        await _render_doc_list()
    except Exception as e:
        await cl.Message(content=f"Error: {e}").send()


@cl.action_callback("btn_rebuild_img")
async def on_rebuild_images(action: cl.Action):
    """Recovery tool: re-scan PDFs on disk and re-extract images with current thresholds."""
    msg = cl.Message(content="🔄 Rebuilding image index from PDFs on disk...")
    await msg.send()
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            res = await client.post(f"{BASE_URL}/rebuild-images/")
        if res.status_code == 429:
            msg.content = "⏱️ Rebuild rate-limited (max 3/hour). Try again later."
            await msg.update()
            return
        data    = res.json()
        details = data.get("details", [])
        lines   = [f"- **{d['filename']}**: {d['images_found']} image(s)" for d in details]
        msg.content = (
            f"✅ **Rebuilt {data.get('documents_rebuilt',0)} document(s)**\n\n"
            + ("\n".join(lines) if lines else "No PDFs found to rebuild.")
        )
        await msg.update()
    except Exception as e:
        msg.content = f"❌ Rebuild failed: {e}"
        await msg.update()


async def _handle_toggle(doc_id: str, filename: str):
    doc_ids = cl.user_session.get("doc_ids", [])
    if doc_id in doc_ids:
        doc_ids.remove(doc_id)
    else:
        doc_ids.append(doc_id)
    cl.user_session.set("doc_ids", doc_ids)
    await _render_doc_list()

@cl.action_callback("toggle_0000")
async def tog0(a): await _handle_toggle(a.payload.get("doc_id",""), a.payload.get("filename",""))
@cl.action_callback("toggle_0001")
async def tog1(a): await _handle_toggle(a.payload.get("doc_id",""), a.payload.get("filename",""))
@cl.action_callback("toggle_0002")
async def tog2(a): await _handle_toggle(a.payload.get("doc_id",""), a.payload.get("filename",""))
@cl.action_callback("toggle_0003")
async def tog3(a): await _handle_toggle(a.payload.get("doc_id",""), a.payload.get("filename",""))
@cl.action_callback("toggle_0004")
async def tog4(a): await _handle_toggle(a.payload.get("doc_id",""), a.payload.get("filename",""))
@cl.action_callback("toggle_0005")
async def tog5(a): await _handle_toggle(a.payload.get("doc_id",""), a.payload.get("filename",""))
@cl.action_callback("toggle_0006")
async def tog6(a): await _handle_toggle(a.payload.get("doc_id",""), a.payload.get("filename",""))
@cl.action_callback("toggle_0007")
async def tog7(a): await _handle_toggle(a.payload.get("doc_id",""), a.payload.get("filename",""))
@cl.action_callback("toggle_0008")
async def tog8(a): await _handle_toggle(a.payload.get("doc_id",""), a.payload.get("filename",""))
@cl.action_callback("toggle_0009")
async def tog9(a): await _handle_toggle(a.payload.get("doc_id",""), a.payload.get("filename",""))

# ── Zone / Clear ──────────────────────────────────────────────────────────────

@cl.action_callback("btn_mode")
async def on_mode(a):
    await cl.Message(
        content=f"**Current:** {zone_label(cl.user_session.get('alpha'))}\n\nChoose a search mode:",
        actions=[
            make_action("set_auto",  "🤖 Auto — LLM picks"),
            make_action("set_zone1", "🔤 Zone 1 — Keyword (α=0.15)"),
            make_action("set_zone2", "⚖️ Zone 2 — Balanced (α=0.50)"),
            make_action("set_zone3", "🧠 Zone 3 — Semantic (α=0.85)"),
        ]
    ).send()

@cl.action_callback("btn_clear")
async def on_clear(a):
    cl.user_session.set("doc_ids", [])
    await cl.Message(
        content="✅ **Cleared** — all documents are now searched.",
        actions=[make_action("btn_docs", "📄 View Documents")]
    ).send()

@cl.action_callback("btn_help")
async def on_help(a):
    await cl.Message(
        content=(
            "## DocuQuery — Help\n\n"
            "**Upload:** Click 📎 to upload PDFs, DOCX, PPTX, TXT, MD (max 50 MB each).\n\n"
            "**Filter documents:**\n"
            "1. Click 📄 Documents\n"
            "2. Tap any document button to select/deselect it\n"
            "3. Empty selection = search ALL documents\n\n"
            "**Images:** Ask 'show me the images' to view figures. "
            "If images stopped appearing after an update, use "
            "🔄 Rebuild Images in the Documents panel.\n\n"
            "**Developer Mode:** Toggle from the menu to see injection risk, "
            "hallucination rationale, and token usage per answer.\n\n"
            "**Rate answers:** 👍 helpful · 👎 not helpful\n\n"
            "**Commands:** `/alpha 0.7` · `/clear_history` · `/feedback` · `/compare id_a id_b`"
        ),
        actions=[
            make_action("btn_docs",    "📄 Documents"),
            make_action("btn_mode",    "🔍 Search Zones"),
            make_action("btn_quality", "📊 Quality Report"),
            make_action("btn_devmode", "🛠️ Developer Mode"),
        ]
    ).send()

@cl.action_callback("set_auto")
async def on_set_auto(a):
    cl.user_session.set("alpha", None)
    await cl.Message(content="🤖 **Auto mode** — system picks the best zone per query.",
                     actions=[make_action("btn_mode","🔍 Change Zone")]).send()

@cl.action_callback("set_zone1")
async def on_z1(a):
    cl.user_session.set("alpha", 0.15)
    await cl.Message(content="🔤 **Zone 1 — Keyword** (α=0.15)\nBest for: names, codes, IDs.",
                     actions=[make_action("btn_mode","🔍 Change Zone")]).send()

@cl.action_callback("set_zone2")
async def on_z2(a):
    cl.user_session.set("alpha", 0.50)
    await cl.Message(content="⚖️ **Zone 2 — Balanced** (α=0.50)\nBest for: general questions.",
                     actions=[make_action("btn_mode","🔍 Change Zone")]).send()

@cl.action_callback("set_zone3")
async def on_z3(a):
    cl.user_session.set("alpha", 0.85)
    await cl.Message(content="🧠 **Zone 3 — Semantic** (α=0.85)\nBest for: concepts and explanations.",
                     actions=[make_action("btn_mode","🔍 Change Zone")]).send()

# ── Commands ──────────────────────────────────────────────────────────────────

async def handle_command(command: str, doc_ids: list, alpha):
    parts = command.strip().split()
    cmd   = parts[0].lower()

    if cmd == "/alpha" and len(parts) > 1:
        try:
            val = float(parts[1])
            if 0.0 <= val <= 1.0:
                cl.user_session.set("alpha", val)
                await cl.Message(content=f"⚙️ **Alpha = {val}** ({zone_desc(val)})",
                                 actions=[make_action("btn_mode","🔍 Change Zone")]).send()
            else:
                await cl.Message(content="⚠️ Alpha must be between 0.0 and 1.0").send()
        except ValueError:
            await cl.Message(content="Usage: `/alpha 0.7`").send()

    elif cmd == "/clear_history":
        cl.user_session.set("history", [])
        await cl.Message(content="✅ **Conversation history cleared.**").send()

    elif cmd == "/compare" and len(parts) == 3:
        await cl.Message(content=f"Comparing `{parts[1]}` vs `{parts[2]}`\n\nWhat question should I compare?").send()
        cl.user_session.set("pending_compare", [parts[1], parts[2]])

    elif cmd == "/devmode":
        dev_mode = not cl.user_session.get("dev_mode", False)
        cl.user_session.set("dev_mode", dev_mode)
        await cl.Message(content=f"🛠️ Developer mode: {'ON ✅' if dev_mode else 'OFF ⬜'}").send()

    elif cmd == "/feedback":
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                res = (await client.get(f"{BASE_URL}/feedback/")).json()
            await cl.Message(
                content=(
                    f"**Feedback summary:**\n"
                    f"👍 {res.get('thumbs_up',0)} · 👎 {res.get('thumbs_down',0)} · Total: {res.get('total',0)}\n\n"
                    "Use 👍 / 👎 buttons below each answer to rate."
                ),
                actions=[make_action("btn_quality","📊 Full Quality Report")]
            ).send()
        except Exception as e:
            await cl.Message(content=f"Could not fetch feedback: {e}").send()

    else:
        await cl.Message(
            content=(
                "**Available commands:**\n"
                "- `/alpha 0.7` — set search weight (0.0–1.0)\n"
                "- `/clear_history` — reset conversation memory\n"
                "- `/compare id_a id_b` — compare two documents\n"
                "- `/devmode` — toggle developer mode\n"
                "- `/feedback` — view feedback summary"
            ),
            actions=[
                make_action("btn_docs",    "📄 Documents"),
                make_action("btn_mode",    "🔍 Search Zone"),
                make_action("btn_quality", "📊 Quality"),
                make_action("btn_devmode", "🛠️ Developer Mode"),
                make_action("btn_help",    "❓ Help"),
            ]
        ).send()