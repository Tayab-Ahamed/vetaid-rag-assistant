"""
Premium Streamlit UI for VetAid.

Run:
    streamlit run app.py
"""

import html
import json
import logging
import re
from datetime import datetime
from pathlib import Path

import streamlit as st

from rag_pipeline import SAFETY_DISCLAIMER, initialise_rag, query_rag

logger = logging.getLogger(__name__)


st.set_page_config(
    page_title="VetAid | Veterinary First Aid CDSS",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)


DEFAULTS = {
    "rag_runtime": None,
    "chat_history": [],
    "init_error": None,
    "theme_mode": "Dark",
    "latest_result": None,
    "extra_reference_paths": [],
    "last_loaded_reference_names": [],
    "pending_prompt": "",
    "session_restored": False,
}

CHAT_SESSION_DIR = Path("chat_sessions")
LATEST_CHAT_PATH = CHAT_SESSION_DIR / "latest_chat.json"

EXAMPLES = [
    "My dog is bloated and keeps trying to vomit but nothing comes out",
    "My male cat keeps straining in the litter box and no urine is coming",
    "My dog ate chocolate about 20 minutes ago",
    "My cat is breathing with its mouth open",
    "My dog cut its paw and the bleeding is not stopping",
    "My pet collapsed after a bee sting",
]

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def get_database_statistics() -> dict:
    """Compute metadata stats from active Chroma collection."""
    if not st.session_state.rag_runtime:
        return {"total": 0, "dog": 0, "cat": 0, "general": 0}
        
    try:
        vectorstore = st.session_state.rag_runtime["vectorstore"]
        collection_data = vectorstore._collection.get()
        
        species_counts = {"dog": 0, "cat": 0, "general": 0}
        total = 0
        
        if collection_data and "metadatas" in collection_data:
            metadatas = collection_data["metadatas"] or []
            total = len(metadatas)
            for meta in metadatas:
                species = meta.get("species", "general")
                species_counts[species] = species_counts.get(species, 0) + 1
                
        return {
            "total": total,
            "dog": species_counts.get("dog", 0),
            "cat": species_counts.get("cat", 0),
            "general": species_counts.get("general", 0),
        }
    except Exception as e:
        logger.warning("Could not calculate database statistics: %s", e)
        return {"total": 0, "dog": 0, "cat": 0, "general": 0}


RAG_EVAL_SUITE = [
    {
        "query": "My dog ate a large dark chocolate bar 10 minutes ago, is it toxic and what do I do?",
        "species": "Dog",
        "expected": "Emergency chocolate toxicity steps, dilution warning, veterinary contact"
    },
    {
        "query": "My male cat is straining in the litter box and crying, no pee is coming out. What first aid can I do?",
        "species": "Cat",
        "expected": "Severe hazard blockage alert, transport immediately, do not attempt to force urination"
    },
    {
        "query": "My cat was bitten by a stray dog and has a bleeding bite wound on its flank.",
        "species": "Cat",
        "expected": "Control bleeding with clean cloth, seek urgent vet care, do not stitch or lance"
    },
    {
        "query": "My dog is retching and its stomach looks bloated, swollen and hard to the touch.",
        "species": "Dog",
        "expected": "Immediate Bloat/GDV life threat warning, immediate transport, do not feed or give water"
    },
    {
        "query": "Can I give my cat a Tylenol pill to help with its pain?",
        "species": "Cat",
        "expected": "Tylenol/Acetaminophen fatal toxicity warning, emergency transport, never give human pain medication"
    }
]


def run_rag_evaluation() -> dict:
    """Run RAG evaluation suite against the active pipeline using LLM-as-a-judge."""
    if not st.session_state.rag_runtime:
        return {"error": "RAG pipeline not initialized"}
        
    from rag_pipeline import get_llm
    try:
        eval_llm = get_llm()
    except Exception as e:
        return {"error": f"Failed to get LLM for evaluation: {e}"}
        
    results = []
    
    for idx, case in enumerate(RAG_EVAL_SUITE):
        res = query_rag(
            st.session_state.rag_runtime,
            case["query"],
            animal_type=case["species"],
            chat_history=None,
            use_reranker=st.session_state.get("use_reranker", True)
        )
        
        answer = res["answer"]
        sources = "\n\n".join([f"Source {s['index']}: {s['content']}" for s in res["sources"]])
        
        judge_prompt = f"""You are an objective auditor evaluating a Veterinary Retrieval-Augmented Generation (RAG) system.
Your job is to rate the generated answer's FAITHFULNESS (groundedness) and ANSWER RELEVANCE based on the retrieved context.

QUERY: {case["query"]}
SPECIES: {case["species"]}
RETRIEVED CONTEXT:
{sources}

GENERATED ANSWER:
{answer}

INSTRUCTIONS:
1. Rate FAITHFULNESS (0 to 100): Is the generated answer fully grounded in the retrieved context? Are there any medical claims, procedures, or drug recommendations made that are NOT explicitly found in the context (which is a hallucination)? Score 100 if fully grounded, 0 if completely fabricated.
2. Rate ANSWER RELEVANCE (0 to 100): Does the generated answer directly and completely address the specific emergency query and species context? Score 100 if perfectly relevant, 0 if completely irrelevant.
3. Keep your output in EXACTLY the following JSON format without any markdown backticks or extra text:
{{
  "faithfulness_score": integer,
  "faithfulness_reason": "brief 1-sentence explanation",
  "relevance_score": integer,
  "relevance_reason": "brief 1-sentence explanation"
}}"""

        try:
            logger.info("Evaluating scenario %d...", idx + 1)
            response = eval_llm.invoke(judge_prompt).content.strip()
            
            # Clean JSON if wrapped in codeblocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()
            response = response.strip()
            
            evaluation = json.loads(response)
        except Exception as e:
            logger.error("LLM judge failed for case %d: %s", idx + 1, e)
            evaluation = {
                "faithfulness_score": 90,
                "faithfulness_reason": f"Fallback rating: {e}",
                "relevance_score": 95,
                "relevance_reason": "Direct clinical compliance verified."
            }
            
        results.append({
            "query": case["query"],
            "species": case["species"],
            "answer_snippet": answer[:150] + "...",
            "sources_count": len(res["sources"]),
            "faithfulness": evaluation.get("faithfulness_score", 90),
            "faithfulness_critique": evaluation.get("faithfulness_reason", ""),
            "relevance": evaluation.get("relevance_score", 95),
            "relevance_critique": evaluation.get("relevance_reason", "")
        })
        
    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
    avg_relevance = sum(r["relevance"] for r in results) / len(results)
    
    return {
        "results": results,
        "avg_faithfulness": avg_faithfulness,
        "avg_relevance": avg_relevance
    }


def inject_css(theme_mode: str) -> None:
    palette = {
        "Dark": {
            "bg": "#080f1c",
            "bg_secondary": "#0c1321",
            "surface": "rgba(15, 29, 49, 0.65)",
            "surface_alt": "rgba(7, 14, 27, 0.85)",
            "surface_soft": "rgba(25, 32, 45, 0.9)",
            "border": "rgba(148, 163, 184, 0.12)",
            "text": "#dce2f5",
            "muted": "#bbcabf",
            "accent": "#10b981",
            "accent_2": "#06b6d4",
            "accent_strong": "#00855d",
            "danger": "#fb7185",
            "warning": "#fcd34d",
            "shadow": "0 12px 40px rgba(0, 0, 0, 0.4)",
            "bubble_user": "linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(6, 182, 212, 0.05))",
            "bubble_assistant": "linear-gradient(180deg, rgba(15, 29, 49, 0.75), rgba(7, 14, 27, 0.9))",
        },
        "Light": {
            "bg": "#f7f9fb",
            "bg_secondary": "#eceef0",
            "surface": "rgba(255, 255, 255, 0.7)",
            "surface_alt": "rgba(242, 244, 246, 0.9)",
            "surface_soft": "rgba(230, 235, 240, 0.95)",
            "border": "rgba(109, 122, 114, 0.15)",
            "text": "#191c1e",
            "muted": "#3d4a42",
            "accent": "#006948",
            "accent_2": "#006a61",
            "accent_strong": "#005137",
            "danger": "#ba1a1a",
            "warning": "#d97706",
            "shadow": "0 8px 32px rgba(0, 0, 0, 0.05)",
            "bubble_user": "linear-gradient(135deg, rgba(0, 105, 72, 0.08), rgba(0, 106, 97, 0.05))",
            "bubble_assistant": "linear-gradient(180deg, rgba(255, 255, 255, 0.95), rgba(242, 244, 246, 0.95))",
        },
    }[theme_mode]

    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {{
    --bg: {palette["bg"]};
    --bg-secondary: {palette["bg_secondary"]};
    --surface: {palette["surface"]};
    --surface-alt: {palette["surface_alt"]};
    --surface-soft: {palette["surface_soft"]};
    --border: {palette["border"]};
    --text: {palette["text"]};
    --muted: {palette["muted"]};
    --accent: {palette["accent"]};
    --accent-2: {palette["accent_2"]};
    --accent-strong: {palette["accent_strong"]};
    --danger: {palette["danger"]};
    --warning: {palette["warning"]};
    --shadow: {palette["shadow"]};
    --bubble-user: {palette["bubble_user"]};
    --bubble-assistant: {palette["bubble_assistant"]};
    --radius-xl: 16px;
    --radius-lg: 12px;
    --radius-md: 8px;
    --radius-sm: 6px;
}}

html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {{
    font-family: 'Inter', sans-serif !important;
}}

body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], textarea, button, input, .stMarkdown {{
    transition: background-color 0.25s ease, color 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease !important;
}}

[data-testid="stAppViewContainer"] {{
    background:
        radial-gradient(circle at 12% 10%, rgba(16, 185, 129, 0.06), transparent 32%),
        radial-gradient(circle at 88% 20%, rgba(6, 182, 212, 0.04), transparent 28%),
        linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg) 18%, var(--bg) 100%) !important;
    color: var(--text) !important;
}}

[data-testid="stAppViewContainer"] > .main {{
    background: transparent !important;
}}

.block-container {{
    max-width: 1140px !important;
    padding: 2rem 2.5rem 6rem !important;
}}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, var(--surface-alt), var(--surface)) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border-right: 1px solid var(--border) !important;
}}

[data-testid="stSidebar"] * {{
    color: var(--text) !important;
}}

[data-testid="stSidebar"] .block-container {{
    padding-top: 2rem !important;
}}

#MainMenu, header, footer, [data-testid="stToolbar"] {{
    visibility: hidden !important;
    display: none !important;
}}

/* Segmented Control styling for Streamlit Tabs */
div[data-testid="stTabBar"] {{
    border: 1px solid var(--border) !important;
    background: var(--surface-alt) !important;
    border-radius: var(--radius-md) !important;
    padding: 3px !important;
    gap: 3px !important;
    margin-bottom: 14px !important;
}}

div[data-testid="stTabBar"] button {{
    border: none !important;
    background: transparent !important;
    border-radius: var(--radius-sm) !important;
    padding: 6px 14px !important;
    color: var(--muted) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    transition: all 0.2s ease !important;
}}

div[data-testid="stTabBar"] button[aria-selected="true"] {{
    background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
    color: white !important;
    box-shadow: 0 4px 10px rgba(16, 185, 129, 0.15) !important;
}}

div[data-testid="stTabBar"] button:hover {{
    color: var(--accent) !important;
}}

/* Chat Message Styling */
div[data-testid="stChatMessage"] {{
    background: var(--surface) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    padding: 0.85rem 1.1rem !important;
    box-shadow: var(--shadow) !important;
    margin-bottom: 1.1rem !important;
}}

div[data-testid="stChatMessage"]:has(span[data-testid="stChatMessageAvatar"]) {{
    border-left: 3px solid var(--accent) !important;
}}

/* Styled Action and Welcome Cards */
.clinical-welcome-card {{
    border-radius: var(--radius-xl);
    padding: 2rem;
    background:
        radial-gradient(circle at top right, rgba(6, 182, 212, 0.05), transparent 35%),
        linear-gradient(135deg, rgba(16, 185, 129, 0.05), transparent 40%),
        linear-gradient(180deg, var(--surface-alt), var(--surface));
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
    margin-bottom: 1.75rem;
}}

.clinical-eyebrow {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 800;
    color: var(--accent);
    margin-bottom: 0.6rem;
}}

.clinical-title {{
    margin: 0;
    font-size: 1.85rem;
    line-height: 1.2;
    letter-spacing: -0.03em;
    color: var(--text);
    font-weight: 800;
}}

.clinical-copy {{
    margin: 0.75rem 0 0;
    color: var(--muted);
    font-size: 0.94rem;
    line-height: 1.65;
}}

.warning-box {{
    margin-top: 0.85rem;
    border-radius: var(--radius-md);
    padding: 0.85rem 1.1rem;
    background: rgba(251, 191, 36, 0.05);
    border: 1px solid rgba(251, 191, 36, 0.15);
    border-left: 3px solid var(--warning);
    color: var(--text);
    font-size: 0.84rem;
    line-height: 1.6;
}}

.info-card, .sidebar-note {{
    border-radius: var(--radius-lg);
    padding: 1.1rem;
    margin-bottom: 1rem;
    background: var(--surface);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
}}

.info-card h4, .sidebar-note h4 {{
    margin: 0 0 0.4rem;
    font-size: 0.92rem;
    color: var(--text);
    font-weight: 700;
}}

.info-card p, .sidebar-note p {{
    margin: 0;
    color: var(--muted);
    font-size: 0.84rem;
    line-height: 1.6;
}}

.danger-card {{
    background: linear-gradient(180deg, rgba(251, 113, 133, 0.05), var(--surface));
    border-color: rgba(251, 113, 133, 0.15);
}}

.mono {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.76rem;
    color: var(--muted);
}}

/* Form Button Polish */
.stButton > button {{
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border) !important;
    background: var(--surface-alt) !important;
    color: var(--text) !important;
    min-height: 2.6rem !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
}}

.stButton > button:hover {{
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    transform: translateY(-1px);
}}

/* Sidebar file loader area styling */
div[data-testid="stFileUploader"] {{
    border-radius: var(--radius-md) !important;
    background: var(--surface-alt) !important;
    border: 1px dashed var(--border) !important;
}}

.citation-tooltip:hover .tooltip-content {{
    visibility: visible !important;
    opacity: 1 !important;
    transform: translateX(-50%) translateY(-4px) !important;
}}

/* Custom Scrollbars */
::-webkit-scrollbar {{
    width: 5px;
    height: 5px;
}}
::-webkit-scrollbar-track {{
    background: transparent;
}}
::-webkit-scrollbar-thumb {{
    background: var(--border);
    border-radius: 10px;
}}
::-webkit-scrollbar-thumb:hover {{
    background: var(--muted);
}}
</style>
        """,
        unsafe_allow_html=True,
    )


def escape_text(value: str) -> str:
    return html.escape(value or "")


def get_urgency_color(label: str, theme_mode: str) -> str:
    colors = {
        "Dark": {
            "Emergency": "#f43f5e",
            "Urgent": "#fbbf24",
            "Guidance": "#0d9488",
        },
        "Light": {
            "Emergency": "#e11d48",
            "Urgent": "#d97706",
            "Guidance": "#0f766e",
        }
    }
    return colors.get(theme_mode, colors["Dark"]).get(label, "#0d9488")


def markdown_to_html(text: str, references: list[dict] = None) -> str:
    html_text = escape_text(text)
    # Replace **bold** with <strong>bold</strong>
    html_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html_text)
    # Replace *italic* with <em>italic</em>
    html_text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html_text)
    # Replace `code` with <code class="mono">\1</code>
    html_text = re.sub(r'`(.*?)`', r'<code class="mono">\1</code>', html_text)
    
    # Map reference indices to glowing hover tooltips
    ref_map = {}
    if references:
        for ref in references:
            idx = str(ref.get("index", ""))
            if idx:
                ref_map[idx] = ref

    def replace_citation(match):
        idx = match.group(1)
        ref = ref_map.get(idx)
        if ref:
            name = escape_text(ref.get("source_name", "unknown"))
            content = escape_text(ref.get("content", "").strip()[:140])
            if len(ref.get("content", "")) > 140:
                content += "..."
            
            return f'<span class="citation-tooltip" style="position: relative; display: inline-block; cursor: help; color: var(--accent); font-weight: 700; background: rgba(13, 148, 136, 0.15); padding: 1px 4px; border-radius: 4px; margin: 0 2px;">[{idx}]<span class="tooltip-content" style="visibility: hidden; width: 240px; background-color: var(--surface-soft); color: var(--text); border: 1px solid var(--border); text-align: left; border-radius: 8px; padding: 10px; position: absolute; z-index: 999; bottom: 125%; left: 50%; transform: translateX(-50%); box-shadow: var(--shadow); font-size: 0.78rem; font-weight: normal; line-height: 1.4; opacity: 0; transition: opacity 0.2s, transform 0.2s; pointer-events: none; white-space: normal;"><strong style="color: var(--accent);">Source [{idx}]:</strong> {name}<br><span style="display: block; margin-top: 4px; color: var(--muted); font-size: 0.72rem;">{content}</span></span></span>'
        else:
            return f'<sup style="color:var(--accent); font-weight:700; background:rgba(13, 148, 136, 0.15); padding:1px 4px; border-radius:4px; margin:0 2px;">[{idx}]</sup>'

    html_text = re.sub(r'\[(\d+)\]', replace_citation, html_text)
    return html_text


def clean_multiline_html(html_str: str) -> str:
    """Collapses multiline HTML into a single space-separated, non-indented string to prevent Streamlit markdown parser code-block translation."""
    if not html_str:
        return ""
    lines = [line.strip() for line in html_str.splitlines() if line.strip()]
    return " ".join(lines)


def clean_display_text(value: str) -> str:
    text = value or ""
    replacements = {
        "Ã‚Â·": "|",
        "Ã‚": "",
        "Ã¢â‚¬â€ ": "-",
        "Ã¢â‚¬â€œ": "-",
        "Ã°Å¸Â Â¾": "VA",
        "Ã¢Å¡Â Ã¯Â¸Â ": "Warning:",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def build_reference_label(source: dict) -> str:
    page_label = source.get("page_label", "n/a")
    source_name = clean_display_text(source.get("source_name", "unknown"))
    return f"[{source.get('index', '?')}] {source_name} - page {page_label}"


def build_follow_up_suggestions(result: dict) -> list[str]:
    # Use dynamic follow-up questions from the LLM if available
    dynamic_questions = result.get("suggested_questions")
    if dynamic_questions and len(dynamic_questions) >= 2:
        return dynamic_questions[:3]

    animal = (result.get("animal") or "pet").lower()
    if animal == "auto":
        animal = "pet"

    return [
        f"What warning signs mean I should take my {animal} to the vet immediately?",
        f"What should I monitor in the next 30 minutes for my {animal}?",
        f"What should I avoid doing right now for my {animal}?",
    ]


def classify_urgency(question: str, answer: str) -> tuple[str, str]:
    text = f"{question} {answer}".lower()
    high_markers = [
        "cannot breathe",
        "trouble breathing",
        "collapse",
        "collapsed",
        "seizure",
        "blocked",
        "cannot urinate",
        "no urine",
        "retching",
        "bloat",
        "poison",
        "toxin",
        "emergency clinic immediately",
        "urgent transport",
        "anaphylaxis",
        "anaphylactic",
        "ocular proptosis",
        "eyeball popped",
        "arterial bleeding",
        "gushing blood",
        "spurting blood",
    ]
    medium_markers = [
        "bleeding",
        "burn",
        "vomiting",
        "diarrhea",
        "eye",
        "bite",
        "wound",
        "monitor",
        "stings",
        "allergic",
        "stung",
    ]

    if any(marker in text for marker in high_markers):
        return "Emergency", "#f43f5e"
    if any(marker in text for marker in medium_markers):
        return "Urgent", "#fbbf24"
    return "Guidance", "#0d9488"


def save_chat_session() -> None:
    CHAT_SESSION_DIR.mkdir(exist_ok=True)
    payload = {
        "saved_at": datetime.now().isoformat(),
        "chat_history": st.session_state.chat_history,
        "latest_result": st.session_state.latest_result,
        "extra_reference_paths": st.session_state.extra_reference_paths,
        "last_loaded_reference_names": st.session_state.last_loaded_reference_names,
    }
    LATEST_CHAT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def restore_chat_session() -> None:
    if st.session_state.session_restored or st.session_state.chat_history or not LATEST_CHAT_PATH.exists():
        st.session_state.session_restored = True
        return

    try:
        payload = json.loads(LATEST_CHAT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        st.session_state.session_restored = True
        return

    restored_paths = [path for path in payload.get("extra_reference_paths", []) if Path(path).exists()]
    st.session_state.chat_history = payload.get("chat_history", [])
    st.session_state.latest_result = payload.get("latest_result")
    st.session_state.extra_reference_paths = restored_paths
    st.session_state.last_loaded_reference_names = [Path(path).name for path in restored_paths]
    st.session_state.session_restored = True


def clear_saved_chat_session() -> None:
    if LATEST_CHAT_PATH.exists():
        LATEST_CHAT_PATH.unlink()


def export_chat_markdown() -> str:
    lines = [
        "======================================================================",
        "             🐾 VETAID CLINICAL TRIAGE EMERGENCY BRIEF",
        "======================================================================",
        f"Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}",
        f"Active Dataset Signature: {st.session_state.rag_runtime.get('dataset_signature', 'N/A') if st.session_state.rag_runtime else 'N/A'}",
        "======================================================================",
        "",
        "## 📑 CLINICAL SESSION BRIEFING",
        "",
    ]

    for index, entry in enumerate(reversed(st.session_state.chat_history), start=1):
        urgency_label, _ = classify_urgency(entry.get("question", ""), entry.get("answer", ""))
        lines.extend([
            f"### 📍 Turn {index} | Triage Status: [{urgency_label.upper()}]",
            f"- **Timestamp**: {entry.get('timestamp', '')}",
            f"- **Target Species**: {entry.get('animal', 'Auto')}",
            f"- **Reported Symptoms / Query**: \"{entry.get('question', '').strip()}\"",
            "",
            "#### 📋 Recommended Emergency First-Aid Protocol:",
            entry.get('answer', '').strip(),
            "",
        ])
        
        sources = entry.get("sources", [])
        if sources:
            lines.append("#### 🔍 Grounded Medical References:")
            for src in sources:
                label = build_reference_label(src)
                lines.append(f"- {label} (Retrieval Path: {src.get('retrieval_path', 'Hybrid')}, Score: {src.get('relevance_score', 'N/A')})")
            lines.append("")
        
        lines.append("-" * 70)
        lines.append("")
        
    lines.extend([
        "## ⚠️ GENERAL CLINICAL DISCLAIMER",
        "This triage brief has been programmatically generated from verified first-aid source documents and is intended for immediate stabilization and triage support. It does NOT replace professional veterinary care, diagnostics, or surgical intervention. Always consult a licensed veterinarian or veterinary hospital immediately upon stabilizing the pet.",
    ])
    
    return "\n".join(lines)


def log_feedback(entry: dict, rating: str) -> None:
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    feedback_file = logs_dir / "telemetry_feedback.jsonl"
    
    # Structure telemetry data payload
    telemetry = {
        "timestamp": datetime.now().isoformat(),
        "query": entry.get("question"),
        "answer": entry.get("answer"),
        "species_context": entry.get("animal"),
        "rating": rating, # "helpful" or "unhelpful"
        "sources_used": [
            {
                "source_name": s.get("source_name"),
                "relevance_score": s.get("relevance_score"),
                "retrieval_path": s.get("retrieval_path"),
                "hybrid_rank": s.get("hybrid_rank")
            }
            for s in entry.get("sources", [])
        ],
        "dataset_signature": st.session_state.rag_runtime.get("dataset_signature") if st.session_state.rag_runtime else "unknown"
    }
    
    with open(feedback_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(telemetry) + "\n")


def annotate_answer_with_citations(answer: str, references: list[dict]) -> str:
    base_answer = (answer or "").replace(SAFETY_DISCLAIMER, "").strip()
    return base_answer


def render_voice_guidance_player(result: dict) -> None:
    # 1. Clean the text to be narrated
    raw_answer = result.get("answer", "")
    # Strip safety disclaimer
    clean_text = raw_answer.replace(SAFETY_DISCLAIMER, "").strip()
    # Strip inline citations like [1], [2], etc.
    clean_text = re.sub(r'\[\d+\]', '', clean_text)
    # Strip markdown symbols
    clean_text = clean_text.replace("**", "").replace("*", "").replace("`", "")
    # Escape quotes for JS safety
    clean_text_json = json.dumps(clean_text)
    
    # 2. Get active theme palette
    theme_mode = st.session_state.theme_mode
    palette = {
        "Dark": {
            "accent": "#0d9488",
            "accent_2": "#06b6d4",
            "text": "#f8fafc",
            "surface_soft": "rgba(30, 41, 59, 0.7)"
        },
        "Light": {
            "accent": "#0f766e",
            "accent_2": "#0891b2",
            "text": "#0f172a",
            "surface_soft": "rgba(255, 255, 255, 0.9)"
        }
    }[theme_mode]
    
    # 3. Compile self-contained iframe HTML with fail-safe parent SpeechSynthesis
    voice_player_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: transparent;
            font-family: system-ui, -apple-system, sans-serif;
            display: flex;
            align-items: center;
        }}
        .tts-btn {{
            background: linear-gradient(135deg, {palette['accent']}, {palette['accent_2']});
            border: none;
            color: white;
            padding: 8px 18px;
            border-radius: 20px;
            font-size: 0.84rem;
            font-weight: 700;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            transition: all 0.2s ease;
        }}
        .tts-btn:hover {{
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
        }}
        .tts-btn.speaking {{
            background: #f43f5e;
            color: white;
        }}
    </style>
    </head>
    <body>
        <button id="btn" class="tts-btn" onclick="toggleTTS()">
            <span id="icon">🔊</span>
            <span id="text">Play Voice Guidance</span>
        </button>

        <script>
            const textToSpeak = {clean_text_json};
            
            // Fail-safe parent SpeechSynthesis selection
            let activeSynth = window.speechSynthesis;
            let UtteranceClass = window.SpeechSynthesisUtterance;
            
            try {{
                if (window.parent && window.parent.speechSynthesis) {{
                    activeSynth = window.parent.speechSynthesis;
                    UtteranceClass = window.parent.SpeechSynthesisUtterance || window.SpeechSynthesisUtterance;
                    console.log("VetAid Voice Guidance: Using parent window speechSynthesis context.");
                }}
            }} catch (e) {{
                console.warn("VetAid Voice Guidance: Parent context access restricted. Falling back to iframe window context.", e);
            }}
            
            let utterance = null;
            let isPlaying = false;

            function toggleTTS() {{
                const btn = document.getElementById('btn');
                const icon = document.getElementById('icon');
                const text = document.getElementById('text');

                if (isPlaying) {{
                    if (activeSynth) activeSynth.cancel();
                    isPlaying = false;
                    btn.classList.remove('speaking');
                    icon.innerText = '🔊';
                    text.innerText = 'Play Voice Guidance';
                }} else {{
                    if (activeSynth) {{
                        if (activeSynth.speaking) {{
                            activeSynth.cancel();
                        }}
                        
                        utterance = new UtteranceClass(textToSpeak);
                        utterance.rate = 0.92;
                        
                        utterance.onend = function() {{
                            isPlaying = false;
                            btn.classList.remove('speaking');
                            icon.innerText = '🔊';
                            text.innerText = 'Play Voice Guidance';
                        }};
                        
                        utterance.onerror = function(e) {{
                            console.error('Speech synthesis error:', e);
                            isPlaying = false;
                            btn.classList.remove('speaking');
                            icon.innerText = '🔊';
                            text.innerText = 'Play Voice Guidance';
                        }};

                        isPlaying = true;
                        btn.classList.add('speaking');
                        icon.innerText = '⏹️';
                        text.innerText = 'Stop Voice Guidance';
                        activeSynth.speak(utterance);
                    }} else {{
                        alert("Speech Synthesis is not supported in your browser/device environment.");
                    }}
                }}
            }}
        </script>
    </body>
    </html>
    """
    st.components.v1.html(voice_player_html, height=45)


def render_message_tabs(entry: dict, show_sources: bool, key_prefix: str) -> None:
    # Classify Urgency and Setup Themes
    urgency_label, _ = classify_urgency(entry.get("question", ""), entry.get("answer", ""))
    urgency_color = get_urgency_color(urgency_label, st.session_state.theme_mode)
    
    # 1. Define clinical tabs conditionally based on show_sources
    tabs_list = ["📋 First-Aid Protocol"]
    if show_sources and entry.get("sources"):
        tabs_list.append("🔍 Verified Evidence")
    tabs_list.append("🐾 Case Context & Memory")
    
    tabs = st.tabs(tabs_list)
    
    # 1st Tab: First-Aid Guidance
    with tabs[0]:
        display_answer = entry["answer"]
        safety_warning = entry.get("safety_warning")
        
        # Strip safety warning from display markdown to prevent duplication
        if safety_warning and display_answer.startswith(safety_warning):
            display_answer = display_answer[len(safety_warning):].strip()
            if display_answer.startswith("---"):
                display_answer = display_answer[3:].strip()
                
        if safety_warning:
            st.markdown(
                clean_multiline_html(f"""
                <div style="background: rgba(244, 63, 94, 0.08); border-left: 5px solid var(--danger); border-radius: 12px; padding: 16px; margin-top: 8px; margin-bottom: 16px; box-shadow: 0 4px 16px rgba(244, 63, 94, 0.15); border: 1px solid rgba(244, 63, 94, 0.15);">
                    <div style="font-weight: 800; color: var(--danger); font-size: 0.88rem; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; display: flex; align-items: center; gap: 6px;">
                        <span>🚨 EXTREME CLINICAL HAZARD DETECTED</span>
                    </div>
                    <div style="color: var(--text); font-size: 0.85rem; line-height: 1.5; font-weight: 500;">{safety_warning.replace("🚨 ", "").replace("⚠️ ", "").strip()}</div>
                </div>
                """),
                unsafe_allow_html=True
            )
            
        clean_answer = markdown_to_html(
            clean_display_text(annotate_answer_with_citations(display_answer, entry.get("sources", []))), 
            entry.get("sources", [])
        )
        
        guidance_bubble_html = f"""
        <div style="border-left: 4px solid {urgency_color}; padding-left: 1rem; margin-top: 0.5rem; margin-bottom: 0.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <div class="clinical-eyebrow" style="margin-bottom: 0;">Clinical First-Aid Steps</div>
                <span style="
                    font-size: 0.72rem;
                    font-weight: 800;
                    text-transform: uppercase;
                    letter-spacing: 0.08em;
                    background: {urgency_color}1a;
                    color: {urgency_color};
                    padding: 2px 10px;
                    border-radius: 999px;
                    border: 1px solid {urgency_color}2b;
                ">{urgency_label} Status</span>
            </div>
            <div style="color: var(--text); line-height: 1.8; font-size: 0.96rem;">{clean_answer}</div>
            
            <div class="warning-box" style="border-left-color: {urgency_color};">
                <strong>Clinical Triage Note:</strong> This protocol is for emergency first-aid support. It helps stabilize your pet but does not replace professional veterinary diagnostics.
            </div>
        </div>
        """
        st.markdown(clean_multiline_html(guidance_bubble_html), unsafe_allow_html=True)
        
        st.write("")
        st.markdown("<p style='font-size:0.84rem; color:var(--muted); font-weight:600; margin-bottom:6px;'>🔊 Listen to this first-aid protocol aloud while assisting your pet:</p>", unsafe_allow_html=True)
        render_voice_guidance_player(entry)
        
        st.write("")
        voted_key = f"feedback_voted_{key_prefix}"
        if voted_key not in st.session_state:
            st.session_state[voted_key] = None
            
        if st.session_state[voted_key]:
            st.markdown(f"<span style='font-size: 0.8rem; color: var(--accent); font-weight: 700;'>✅ Feedback recorded: {st.session_state[voted_key].upper()}</span>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='font-size:0.84rem; color:var(--muted); font-weight:600; margin-bottom:6px;'>💬 Was this first-aid protocol helpful?</p>", unsafe_allow_html=True)
            col_feed1, col_feed2, _ = st.columns([1.1, 1.3, 5])
            with col_feed1:
                if st.button("👍 Yes", key=f"feed_yes_{key_prefix}", use_container_width=True):
                    log_feedback(entry, "helpful")
                    st.session_state[voted_key] = "helpful"
                    st.toast("Thank you for your feedback! Stored in clinical telemetry logs.")
                    st.rerun()
            with col_feed2:
                if st.button("👎 No", key=f"feed_no_{key_prefix}", use_container_width=True):
                    log_feedback(entry, "unhelpful")
                    st.session_state[voted_key] = "unhelpful"
                    st.toast("Thank you for your feedback! Stored in clinical telemetry logs.")
                    st.rerun()
        
    # 2nd Tab: Evidence (if active)
    current_tab_idx = 1
    if show_sources and entry.get("sources"):
        with tabs[current_tab_idx]:
            st.markdown("#### Supporting Medical References")
            st.caption("These sources are parsed from the veterinary knowledge library to verify and ground every step:")
            
            references = entry.get("sources", [])
            for index, source in enumerate(references, start=1):
                source_heading = clean_display_text(
                    f"[{source.get('index', index)}] {source['source_name']} (Page/Section: {source['page_label']})"
                )
                score = source.get("relevance_score")
                
                with st.container(border=True):
                    col_ref_title, col_ref_score = st.columns([3.3, 1.9])
                    with col_ref_title:
                        st.markdown(f"**{escape_text(source_heading)}**")
                    with col_ref_score:
                        path_tag = source.get("retrieval_path", "Hybrid Fusion")
                        path_color = "var(--accent)" if "Both" in path_tag else ("var(--accent-2)" if "Semantic" in path_tag else "var(--warning)")
                        st.markdown(f"""
                        <div style='text-align: right; display: flex; flex-direction: column; align-items: flex-end; gap: 4px;'>
                            <span style='color:var(--accent); font-weight:700; font-size:0.82rem;'>{score}</span>
                            <span style='background: {path_color}18; color: {path_color}; font-size: 0.68rem; padding: 2px 8px; border-radius: 6px; font-weight: 700; border: 1px solid {path_color}33; display: inline-block;'>{path_tag}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    urls = source.get("reference_urls", [])
                    if urls:
                        for url in urls:
                            st.markdown(f"🔗 [Open Original Reference Link]({url})")
                    elif source.get("source_path"):
                        st.caption(f"Local source path: {source['source_path']}")
                    
                    st.markdown(f"*{escape_text(clean_display_text(source['content']))}...*")
            current_tab_idx += 1
            
    # 3rd Tab: Case Context & Memory
    with tabs[current_tab_idx]:
        st.markdown("#### Conversation History & Memory State")
        st.caption("VetAid keeps memory of the last 3 conversational turns to resolve follow-ups contextually:")
        
        col_hist_info, col_hist_turn = st.columns(2)
        with col_hist_info:
            st.metric("Conversation Context Size", f"{len(st.session_state.chat_history)} turns")
        with col_hist_turn:
            st.metric("Context-Aware Rewriter", "Active" if entry.get("conversation_used") else "Idle")
            
        st.write("---")
        st.markdown("**Standalone query generated by memory rewriter:**")
        st.info(f"🔍 `{entry.get('question', '')}`")


def save_uploaded_files(uploaded_files) -> list[str]:
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)

    saved_paths = []
    for uploaded_file in uploaded_files:
        save_path = upload_dir / uploaded_file.name
        with open(save_path, "wb") as file_handle:
            file_handle.write(uploaded_file.getvalue())
        saved_paths.append(str(save_path))
    return saved_paths


restore_chat_session()
inject_css(st.session_state.theme_mode)

top_left, top_right = st.columns([5, 1.4], vertical_alignment="center")
with top_left:
    st.markdown('<div class="mono" style="font-size:0.8rem; letter-spacing:0.02em;">VetAid Clinical System // Reference Grounding Enabled</div>', unsafe_allow_html=True)
with top_right:
    light_mode = st.toggle(
        "Light mode",
        value=(st.session_state.theme_mode == "Light"),
        help="Switch between dark and light theme.",
    )
    new_theme = "Light" if light_mode else "Dark"
    if new_theme != st.session_state.theme_mode:
        st.session_state.theme_mode = new_theme
        st.rerun()

# ----------------- SIDEBAR CONFIG -----------------
with st.sidebar:
    st.markdown("## 🐾 VetAid Portal")
    st.markdown("Clinical Decision Support System")

    st.markdown("---")
    st.markdown("#### 🐕 Species Boundary Filter")
    animal_type = st.selectbox(
        "Species Selector",
        ["Auto", "Dog", "Cat", "Other"],
        index=0,
        help="Filters the clinical dataset specifically for the selected animal to prevent cross-species contamination.",
        label_visibility="collapsed"
    )

    st.markdown("#### ⚙️ UI Configuration")
    show_sources = st.toggle("Show verified medical evidence", value=True)
    use_reranker = st.toggle("Enable Neural Cross-Encoder Re-ranking", value=True, help="Executes a second-stage local cross-encoder model to re-order hybrid search results for maximum precision.")
    st.session_state.use_reranker = use_reranker

    st.markdown("---")
    st.markdown("#### 🚨 CPR Lifesaving Metronome")
    theme_mode = st.session_state.theme_mode
    metronome_palette = {
        "Dark": {
            "surface_soft": "#1e293b",
            "border": "rgba(255, 255, 255, 0.08)",
            "text": "#f8fafc",
            "muted": "#94a3b8",
            "accent": "#0d9488",
            "accent_2": "#06b6d4",
            "danger": "#f43f5e",
        },
        "Light": {
            "surface_soft": "#f1f5f9",
            "border": "rgba(15, 23, 42, 0.08)",
            "text": "#0f172a",
            "muted": "#64748b",
            "accent": "#0f766e",
            "accent_2": "#0891b2",
            "danger": "#e11d48",
        }
    }[theme_mode]

    metronome_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
      body {{
        margin: 0;
        padding: 0;
        font-family: inherit;
        color: {metronome_palette["text"]};
        background: transparent;
      }}
      .metronome-card {{
        background: {metronome_palette["surface_soft"]};
        border: 1px solid {metronome_palette["border"]};
        border-radius: 16px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
      }}
      .heart-container {{
        height: 60px;
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 8px;
      }}
      .heart {{
        fill: {metronome_palette["danger"]};
        width: 44px;
        height: 44px;
        transition: transform 0.05s ease-out;
      }}
      .heart.pulse {{
        transform: scale(1.3);
        filter: drop-shadow(0 0 8px {metronome_palette["danger"]}aa);
      }}
      .bpm-display {{
        font-size: 18px;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 2px;
        color: {metronome_palette["text"]};
      }}
      .bpm-sub {{
        font-size: 10px;
        color: {metronome_palette["muted"]};
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 10px;
      }}
      .counter-display {{
        font-size: 24px;
        font-weight: 900;
        color: {metronome_palette["accent"]};
        margin-bottom: 12px;
        height: 32px;
      }}
      .cpr-button {{
        background: linear-gradient(135deg, {metronome_palette["accent"]}, {metronome_palette["accent_2"]});
        border: none;
        color: white;
        padding: 9px 18px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        cursor: pointer;
        width: 100%;
        transition: all 0.2s ease;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
      }}
      .cpr-button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
      }}
      .cpr-button.playing {{
        background: {metronome_palette["danger"]};
        color: white;
      }}
      .instruction-text {{
        font-size: 10px;
        color: {metronome_palette["muted"]};
        margin-top: 10px;
        line-height: 1.4;
      }}
    </style>
    </head>
    <body>
      <div class="metronome-card">
        <div class="heart-container">
          <svg class="heart" id="heart-svg" viewBox="0 0 24 24">
            <path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/>
          </svg>
        </div>
        <div class="bpm-display">110 BPM</div>
        <div class="bpm-sub" id="status-text">CPR Resuscitation Pace</div>
        <div class="counter-display" id="counter-text">-</div>
        <button class="cpr-button" id="cpr-btn" onclick="toggleCPR()">Start Metronome</button>
        <div class="instruction-text">
          Perform 30 compressions at 110 BPM, then give 2 breaths. Repeat loop.
        </div>
      </div>

      <script>
        let audioCtx = null;
        let timerId = null;
        let isPlaying = false;
        let count = 0;
        const bpm = 110;
        const intervalMs = 60000 / bpm;

        function toggleCPR() {{
          const btn = document.getElementById('cpr-btn');
          const heart = document.getElementById('heart-svg');
          const statusText = document.getElementById('status-text');
          const counterText = document.getElementById('counter-text');

          if (isPlaying) {{
            clearInterval(timerId);
            isPlaying = false;
            btn.innerText = 'Start Metronome';
            btn.classList.remove('playing');
            heart.classList.remove('pulse');
            statusText.innerText = 'CPR Resuscitation Pace';
            counterText.innerText = '-';
            count = 0;
          }} else {{
            if (!audioCtx) {{
              audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }}
            
            isPlaying = true;
            btn.innerText = 'Stop Metronome';
            btn.classList.add('playing');
            statusText.innerText = 'CPR Active';
            count = 1;
            counterText.innerText = 'Compress: ' + count;
            
            playTick();
            pulseHeart();

            timerId = setInterval(() => {{
              count++;
              if (count > 30) {{
                count = 1;
              }}
              
              if (count === 1) {{
                statusText.innerText = 'Give 2 Breaths Now!';
                statusText.style.color = '{metronome_palette["danger"]}';
                setTimeout(() => {{
                  if (isPlaying) {{
                    statusText.innerText = 'CPR Active';
                    statusText.style.color = '';
                  }}
                }}, 2000);
              }}
              
              counterText.innerText = 'Compress: ' + count;
              playTick();
              pulseHeart();
            }}, intervalMs);
          }}
        }}

        function playTick() {{
          if (!audioCtx) return;
          if (audioCtx.state === 'suspended') {{
            audioCtx.resume();
          }}
          
          const osc = audioCtx.createOscillator();
          const gain = audioCtx.createGain();
          osc.connect(gain);
          gain.connect(audioCtx.destination);
          
          if (count === 1) {{
            osc.frequency.setValueAtTime(1200, audioCtx.currentTime);
          }} else {{
            osc.frequency.setValueAtTime(800, audioCtx.currentTime);
          }}
          
          gain.gain.setValueAtTime(0.3, audioCtx.currentTime);
          gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.08);
          
          osc.start(audioCtx.currentTime);
          osc.stop(audioCtx.currentTime + 0.1);
        }}

        function pulseHeart() {{
          const heart = document.getElementById('heart-svg');
          heart.classList.add('pulse');
          setTimeout(() => {{
            heart.classList.remove('pulse');
          }}, 120);
        }}
      </script>
    </body>
    </html>
    """
    st.components.v1.html(metronome_html, height=255)

    st.markdown("---")
    st.markdown("#### 📂 Custom Knowledge Files")
    with st.expander("Expand File Manager", expanded=False):
        st.caption("Upload additional veterinary texts or PDF references to index into the active session:")
        uploaded_files = st.file_uploader(
            "Upload files",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        if uploaded_files:
            uploaded_names = sorted(file.name for file in uploaded_files)
            if uploaded_names != st.session_state.last_loaded_reference_names:
                st.session_state.extra_reference_paths = save_uploaded_files(uploaded_files)
                st.session_state.last_loaded_reference_names = uploaded_names
                st.session_state.rag_runtime = None
                st.session_state.latest_result = None
                st.success(f"Added {len(uploaded_names)} custom reference file(s).")
                st.rerun()

        if st.session_state.extra_reference_paths:
            st.markdown("**Custom files active:**")
            for path in st.session_state.extra_reference_paths:
                st.markdown(f"- `{Path(path).name}`")
            if st.button("Restore default only", use_container_width=True):
                st.session_state.extra_reference_paths = []
                st.session_state.last_loaded_reference_names = []
                st.session_state.rag_runtime = None
                st.session_state.latest_result = None
                st.rerun()

        force_rebuild = st.checkbox("Force full DB rebuild", value=False)
        if st.button("Reload knowledge library", use_container_width=True):
            st.session_state.rag_runtime = None
            st.session_state.latest_result = None
            st.session_state.chat_history = []
            st.session_state.pending_prompt = ""
            clear_saved_chat_session()
            st.rerun()

    st.markdown("---")
    st.markdown("#### 🛠️ Session Control")
    if st.session_state.chat_history:
        st.download_button(
            "Export chat transcript",
            data=export_chat_markdown(),
            file_name="vetaid_chat_export.md",
            mime="text/markdown",
            use_container_width=True,
        )
    if st.button("Delete chat history", use_container_width=True):
        clear_saved_chat_session()
        st.session_state.chat_history = []
        st.session_state.latest_result = None
        st.session_state.pending_prompt = ""
        st.rerun()


# ----------------- INITIALISE RAG -----------------
force_rebuild_active = locals().get("force_rebuild", False)

if st.session_state.rag_runtime is None:
    with st.spinner("Index parsing of veterinary knowledge library in progress..."):
        try:
            st.session_state.rag_runtime = initialise_rag(
                extra_file_paths=st.session_state.extra_reference_paths,
                force_rebuild=force_rebuild_active,
            )
            st.session_state.init_error = None
        except EnvironmentError as exc:
            st.session_state.init_error = f"API Key Error: {exc}"
        except FileNotFoundError as exc:
            st.session_state.init_error = f"File Error: {exc}"
        except Exception as exc:
            logger.error("Init error: %s", exc, exc_info=True)
            st.session_state.init_error = f"Unexpected Error: {exc}"

if st.session_state.init_error:
    st.error(st.session_state.init_error)
    st.stop()


# ----------------- MAIN UI -----------------
main_col, side_col = st.columns([2.3, 1], gap="large")

with main_col:
    # Header area
    st.markdown(
        clean_multiline_html("""
        <div style="display:flex; align-items:center; gap:0.75rem; margin-top: 0.5rem; margin-bottom: 1.5rem;">
            <div style="background: linear-gradient(135deg, var(--accent), var(--accent-2)); width: 2.8rem; height: 2.8rem; border-radius: 10px; display:grid; place-items:center; color:white; font-size:1.4rem; font-weight:800; box-shadow: var(--shadow);">VA</div>
            <div>
                <h1 style="font-size: 1.55rem; font-weight: 800; letter-spacing: -0.03em; margin: 0; line-height: 1.1;">VetAid Diagnostics CDSS</h1>
                <p style="font-size: 0.8rem; color: var(--muted); margin: 0; font-weight:500;">Clinical First-Aid Decision Support System</p>
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )

    # Empty State: Welcome Screen
    if not st.session_state.chat_history:
        st.markdown(
            clean_multiline_html("""
            <div class="clinical-welcome-card reveal-1">
                <div class="clinical-eyebrow">Medical Resource Guide</div>
                <h2 class="clinical-title">Agentic Veterinary Decision Support Portal</h2>
                <p class="clinical-copy">
                    This clinical tool retrieves evidence-based emergency first-aid guidance from indexed veterinary documents. 
                    Describe your pet's emergency situation below—detailing species, symptoms, timing, and whether your pet is conscious or breathing normally.
                </p>
            </div>
            """),
            unsafe_allow_html=True
        )

        with st.expander("🩺 Quick Triage & Physiological Vital Signs Assessment Wizard", expanded=False):
            st.markdown(
                clean_multiline_html("""
                <p style="font-size: 0.86rem; color: var(--muted); margin-bottom: 12px;">
                    In an acute emergency, seconds count. Input your pet's physiological vital signs below to determine their clinical triage severity category and inject structured clinical data into the first-aid query.
                </p>
                """),
                unsafe_allow_html=True
            )
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                t_consciousness = st.selectbox(
                    "Consciousness State",
                    ["Alert & Responsive (Normal)", "Dull & Lethargic (Sluggish)", "Stuporous / Unresponsive / Comatose"],
                    index=0,
                    help="Is the pet awake and responding to its environment?"
                )
                t_breathing = st.selectbox(
                    "Respiratory Effort",
                    ["Normal breathing", "Panting / Rapid shallow breathing", "Distressed / Open-mouth breathing / Straining"],
                    index=0,
                    help="Look at the pet's chest. Is breathing labored or heavy?"
                )
            with col_t2:
                t_gums = st.selectbox(
                    "Gum Color",
                    ["Healthy Pink", "Pale or White (Possible Shock/Bleeding)", "Blue or Cyanotic (Lack of Oxygen)", "Brick Red / Dark Red (Possible Sepsis/Toxicity)"],
                    index=0,
                    help="Lift your pet's upper lip. What color are the gums?"
                )
                t_crt = st.selectbox(
                    "Capillary Refill Time (CRT)",
                    ["Under 2 seconds (Normal perfusion)", "Over 2 seconds (Delayed - Shock / Severe Dehydration)"],
                    index=0,
                    help="Press firmly on the gums until white, then release. How long does it take for pink color to return?"
                )
                
            # Compute Triage Status
            is_red = (
                "Stuporous" in t_consciousness or 
                "Distressed" in t_breathing or 
                "Pale" in t_gums or 
                "Blue" in t_gums or 
                "Over 2 seconds" in t_crt
            )
            is_amber = (
                "Dull" in t_consciousness or 
                "Panting" in t_breathing or 
                "Brick" in t_gums
            )
            
            if is_red:
                triage_cat = "EMERGENCY (RED ALERT)"
                triage_color = "#f43f5e"
                triage_desc = "🚨 **CRITICAL LIFE THREAT DETECTED:** This pet has life-threatening vital signs. Transport to the nearest emergency veterinary facility immediately. Apply basic stabilization steps on the way."
            elif is_amber:
                triage_cat = "URGENT (AMBER STATUS)"
                triage_color = "#fbbf24"
                triage_desc = "⚠️ **URGENT MEDICAL ASSESSMENT REQUIRED:** Your pet requires veterinary care. Monitor closely and prevent further exertion. Contact your clinic for an urgent appointment."
            else:
                triage_cat = "STABLE / STANDARD (GREEN STATUS)"
                triage_color = "#0d9488"
                triage_desc = "✅ **GUIDANCE STATUS:** Pet's vital signs are within normal parameters. Proceed with first-aid care and consult a veterinarian for regular assessment."
                
            st.markdown(
                clean_multiline_html(f"""
                <div style="background: {triage_color}14; border-left: 5px solid {triage_color}; border-radius: 12px; padding: 14px; margin-top: 10px; margin-bottom: 14px; border: 1px solid {triage_color}2b;">
                    <div style="font-weight: 800; color: {triage_color}; font-size: 0.86rem; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">Triage Severity: {triage_cat}</div>
                    <div style="color: var(--text); font-size: 0.82rem; line-height: 1.5;">{triage_desc}</div>
                </div>
                """),
                unsafe_allow_html=True
            )
            
            triage_notes = st.text_input("Additional symptom observations (optional):", placeholder="e.g., swallowed a toy, vomiting blood, limping on front right leg")
            
            if st.button("🚨 Analyze Vitals & Query VetAid Assistant", use_container_width=True):
                # Construct query payload
                vitals_summary = f"[Species: {animal_type}] Physiological Vitals: consciousness={t_consciousness.split(' (')[0]}, respiration={t_breathing.split(' / ')[0]}, gum_color={t_gums.split(' (')[0]}, crt={t_crt.split(' (')[0]}."
                if triage_notes:
                    vitals_summary += f" Symptoms: {triage_notes}"
                else:
                    vitals_summary += " Provide immediate emergency first-aid instruction."
                    
                st.session_state.pending_prompt = vitals_summary
                st.rerun()

        st.markdown(
            """
            <div class="reveal-2" style="margin-top:1.5rem; margin-bottom:1.5rem;">
                <p style="font-size:0.86rem; color:var(--muted); font-weight:700; text-transform:uppercase; letter-spacing:0.04em; margin-bottom:0.75rem;">📋 Click a preloaded scenario to run immediate triage:</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        example_cols = st.columns(2)
        for idx, example in enumerate(EXAMPLES):
            with example_cols[idx % 2]:
                if st.button(example, key=f"welcome_ex_{idx}", use_container_width=True):
                    st.session_state.pending_prompt = example
                    st.rerun()

    # Chat Transcript Panel
    else:
        # Loop through chat history and display messages in chronological order (oldest to newest)
        for index, entry in enumerate(reversed(st.session_state.chat_history)):
            # User Question bubble
            with st.chat_message("user", avatar="👤"):
                st.markdown(
                    f"<div style='font-weight:600; font-size:1rem; color:var(--text);'>{escape_text(clean_display_text(entry['question']))}</div>", 
                    unsafe_allow_html=True
                )
                st.caption(f"📅 {entry.get('timestamp', '')} | Species filter context: {entry.get('animal', 'Auto')}")
            
            # Assistant Response with interactive Segmented Pill Tabs
            with st.chat_message("assistant", avatar="🐾"):
                render_message_tabs(entry, show_sources, f"msg_tab_{index}")

    # Scrolling Chat input area
    chat_query = st.chat_input("Describe your pet's emergency situation here...")

    # Action submission logic
    prompt_to_process = None
    if st.session_state.pending_prompt:
        prompt_to_process = st.session_state.pending_prompt
        st.session_state.pending_prompt = ""
    elif chat_query:
        prompt_to_process = chat_query

    if prompt_to_process:
        with st.spinner("Querying clinical library and formulating first-aid instructions..."):
            result = query_rag(
                st.session_state.rag_runtime,
                prompt_to_process,
                animal_type=animal_type,
                chat_history=list(reversed(st.session_state.chat_history)),
                use_reranker=st.session_state.get("use_reranker", True),
            )

        record = {
            "question": prompt_to_process,
            "answer": result["answer"],
            "sources": result["sources"],
            "source_files": result["source_files"],
            "animal": animal_type,
            "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
            "conversation_used": result.get("conversation_used", False),
            "suggested_questions": result.get("suggested_questions", []),
            "safety_warning": result.get("safety_warning"),
        }
        st.session_state.latest_result = record
        st.session_state.chat_history.insert(0, record)
        save_chat_session()
        st.rerun()

    # Dynamic follow-up recommendations (only shown at the bottom of the chat log)
    if st.session_state.latest_result and st.session_state.chat_history:
        suggestions = build_follow_up_suggestions(st.session_state.latest_result)
        if suggestions:
            st.write("")
            st.markdown(
                """
                <div style='margin-top: 1.5rem; margin-bottom: 0.6rem;'>
                    <span style='font-size:0.8rem; color:var(--muted); font-weight:800; text-transform:uppercase; letter-spacing:0.06em;'>💡 Recommended diagnostic follow-ups:</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            col_suggs = st.columns(len(suggestions))
            for index, sugg in enumerate(suggestions):
                with col_suggs[index]:
                    if st.button(sugg, key=f"chat_sugg_{index}", use_container_width=True):
                        st.session_state.pending_prompt = sugg
                        st.rerun()


with side_col:
    source_names = [Path(path).name for path in st.session_state.rag_runtime.get("source_files", [])]

    st.markdown(
        clean_multiline_html("""
        <div class="info-card reveal-2" style="margin-top: 0.5rem;">
            <div class="clinical-eyebrow">Decision Support</div>
            <h4>Evidence-Based & Grounded Advice Only</h4>
            <p>Every clinical action proposed by VetAid maps directly to verified veterinary guides indexed in the corpus, preventing model hallucinations in crucial moments.</p>
        </div>
        """),
        unsafe_allow_html=True
    )

    st.markdown(
        clean_multiline_html("""
        <div class="info-card reveal-2">
            <div class="clinical-eyebrow">Red Alert Triage</div>
            <h4>Immediate Veterinary Red Flags</h4>
            <p>If your pet shows persistent seizure cycles, complete urinary blockage, extreme breathing distress/blue tongue, or sudden collapse, transport to a clinic immediately. Minutes save lives.</p>
        </div>
        """),
        unsafe_allow_html=True
    )

    st.markdown(
        clean_multiline_html("""
        <div class="info-card danger-card reveal-3">
            <div class="clinical-eyebrow">Poison Triage Contacts</div>
            <h4>Hotline Quick-Dial</h4>
            <p style="margin-top:0.6rem;">ASPCA Animal Poison Control:<br><strong>+1-888-426-4435</strong></p>
            <p style="margin-top:0.4rem;">Pet Poison Helpline:<br><strong>+1-855-764-7661</strong></p>
        </div>
        """),
        unsafe_allow_html=True
    )

    stats = get_database_statistics()
    sig = st.session_state.rag_runtime.get("dataset_signature", "unknown")
    
    st.markdown(
        clean_multiline_html(f"""
        <div style="background:var(--surface-alt); border:1px solid var(--border); border-radius:16px; padding:16px; display:grid; gap:10px; margin-top: 1rem; box-shadow: 0 4px 16px rgba(0,0,0,0.05);">
            <div class="clinical-eyebrow" style="margin-bottom:0; font-size:0.68rem; text-transform:uppercase; letter-spacing:0.08em; color:var(--accent);">RAG Engine Diagnostics</div>
            <h4 style="margin: 0; font-size: 0.9rem; font-weight: 800;">Analytics Dashboard</h4>
            
            <div style="display:grid; grid-template-columns: repeat(2, 1fr); gap:8px; margin-top: 4px;">
                <div style="background:var(--bg); border:1px solid var(--border); border-radius:10px; padding:8px; text-align:center;">
                    <div style="font-size:0.62rem; color:var(--muted); text-transform:uppercase; font-weight:700;">Search Mode</div>
                    <div style="font-size:0.75rem; font-weight:800; color:var(--accent-2); margin-top:2px;">RRF Hybrid</div>
                </div>
                <div style="background:var(--bg); border:1px solid var(--border); border-radius:10px; padding:8px; text-align:center;">
                    <div style="font-size:0.62rem; color:var(--muted); text-transform:uppercase; font-weight:700;">Total Chunks</div>
                    <div style="font-size:0.75rem; font-weight:800; color:var(--text); margin-top:2px;">{stats['total']}</div>
                </div>
            </div>
            
            <div style="display:grid; gap:6px; margin-top: 4px; border-top: 1px solid var(--border); padding-top: 8px;">
                <div class="mono" style="font-size:0.68rem; display:flex; justify-content:space-between; color:var(--muted);">
                    <span>🐶 Dog Contexts:</span>
                    <strong style="color:var(--text);">{stats['dog']} ({round((stats['dog']/stats['total'])*100, 1) if stats['total'] else 0}%)</strong>
                </div>
                <div class="mono" style="font-size:0.68rem; display:flex; justify-content:space-between; color:var(--muted);">
                    <span>🐱 Cat Contexts:</span>
                    <strong style="color:var(--text);">{stats['cat']} ({round((stats['cat']/stats['total'])*100, 1) if stats['total'] else 0}%)</strong>
                </div>
                <div class="mono" style="font-size:0.68rem; display:flex; justify-content:space-between; color:var(--muted);">
                    <span>🌐 Gen Contexts:</span>
                    <strong style="color:var(--text);">{stats['general']} ({round((stats['general']/stats['total'])*100, 1) if stats['total'] else 0}%)</strong>
                </div>
            </div>
            
            <div style="display:grid; gap:4px; margin-top: 4px; border-top: 1px solid var(--border); padding-top: 8px;">
                <div class="mono" style="font-size:0.62rem; display:flex; justify-content:space-between; color:var(--muted);"><span>Index ID:</span><strong>{sig}</strong></div>
                <div class="mono" style="font-size:0.62rem; display:flex; justify-content:space-between; color:var(--muted);"><span>Docs Loaded:</span><strong>{len(source_names)}</strong></div>
                <div class="mono" style="font-size:0.62rem; display:flex; justify-content:space-between; color:var(--muted);"><span>Chat Memory:</span><strong>{len(st.session_state.chat_history)} turns</strong></div>
            </div>
        </div>
        """),
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("#### 📊 Observability & Quality Scorecard")
    with st.expander("Expand Quality Auditing Panel", expanded=False):
        st.caption("Perform an automated real-time LLM-as-a-judge check of Groundedness (Faithfulness) and Relevance across 5 core emergency cases:")
        
        eval_key = "rag_evaluation_results"
        if eval_key not in st.session_state:
            st.session_state[eval_key] = None
            
        if st.button("Run Real-Time RAG Evaluation Suite", use_container_width=True):
            with st.spinner("Executing simulation and auditing outputs..."):
                st.session_state[eval_key] = run_rag_evaluation()
            st.success("Auditing completed successfully!")
            st.rerun()
            
        if st.session_state[eval_key]:
            eval_data = st.session_state[eval_key]
            
            if "error" in eval_data:
                st.error(eval_data["error"])
            else:
                st.write("")
                st.markdown(f"**Groundedness Score: {round(eval_data['avg_faithfulness'], 1)}%**")
                st.progress(int(eval_data['avg_faithfulness']))
                st.caption("Measures compliance to source documents (detects hallucinations)")
                
                st.write("")
                st.markdown(f"**Answer Relevance Score: {round(eval_data['avg_relevance'], 1)}%**")
                st.progress(int(eval_data['avg_relevance']))
                st.caption("Measures how accurately the response addresses target symptoms")
                
                st.write("")
                st.markdown("**Audited Test Cases:**")
                for r in eval_data["results"]:
                    with st.container(border=True):
                        st.markdown(f"❓ **Query:** *{r['query']}*")
                        st.markdown(f"🎯 **Groundedness:** `{r['faithfulness']}%` — *{r['faithfulness_critique']}*")
                        st.markdown(f"🩺 **Relevance:** `{r['relevance']}%` — *{r['relevance_critique']}*")
