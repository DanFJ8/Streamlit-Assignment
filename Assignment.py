import os
import urllib.parse
import streamlit as st
import wikipedia
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = (
    "You are an expert market research assistant supporting business analysts and strategists at large corporations. Your role is to produce clear, structured, and insight-driven market research on industries selected by the user, including market size and growth, key segments, competitive dynamics, customer demand drivers, and relevant technological, regulatory, economic, and geopolitical factors. You think and write like a professional market analyst, synthesizing information into actionable insights and strategic implications rather than generic explanations. Your tone is professional, analytical, and concise, with assumptions stated explicitly when data is uncertain and balanced perspectives presented where multiple viewpoints exist. Your outputs prioritize executive relevance, clarity, and decision usefulness."
)

st.title("Market Research Assistant")

# Sidebar for settings
st.sidebar.header("Chatbot Settings")
temperature = 1.0
max_output_tokens = 2000
model = st.sidebar.selectbox("Model", ["gemini-3-flash-preview", "gemma-3-1b-it-preview"])
api_key = st.sidebar.text_input("Please input your API key here", type="password")

# Main area keeps the chat

# Helpers
def extract_text(result) -> str:
    content = getattr(result, "content", result)
    if isinstance(content, str):
        text = content
    elif isinstance(content, dict):
        text = content.get("text", str(content))
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text", str(part)))
            else:
                parts.append(str(part))
        text = "\n".join(parts)
    else:
        text = str(content)
    return text.strip()

def cap_500_words(text: str) -> str:
    words = text.split()
    if len(words) <= 500:
        return text
    return " ".join(words[:500]).rstrip() + " …"

def word_count(text: str) -> int:
    return len([w for w in text.split() if w.strip()])

def wiki_url_from_title(title: str, lang: str = "en") -> str:
    safe_title = title.replace(" ", "_")
    return f"https://{lang}.wikipedia.org/wiki/{urllib.parse.quote(safe_title)}"

def find_wikipedia_candidates(industry: str, lang: str = "en", limit: int = 15) -> list[str]:
    wikipedia.set_lang(lang)

    queries = [
        industry,
        f"{industry} industry",
        f"{industry} sector",
        f"{industry} market",
        f"industry of {industry}",
        f"{industry} value chain",
        f"{industry} business",
        f"{industry} economics",
    ]

    seen = set()
    titles: list[str] = []

    for q in queries:
        try:
            results = wikipedia.search(q, results=10)
        except Exception:
            results = []

        for t in results:
            if t not in seen:
                seen.add(t)
                titles.append(t)
            if len(titles) >= limit:
                return titles[:limit]

    padding_candidates = [
        "Industry",
        "Market (economics)",
        "Value chain",
        "Supply chain",
        "Business",
        "Economics",
    ]
    for t in padding_candidates:
        if t not in seen:
            titles.append(t)
            seen.add(t)
        if len(titles) >= limit:
            break

    return titles[:limit]

def get_main_industry_page(industry: str) -> tuple[str, str]:
    try:
        page = wikipedia.page(industry, auto_suggest=True)
        summary = (page.summary or "").strip()
        return page.title, summary
    except Exception:
        return industry, ""

def select_titles_with_llm(industry: str, main_title: str, main_summary: str, candidates: list[str]) -> list[str]:
    candidate_block = "\n".join(f"- {t}" for t in candidates)
    prompt = f"""You are a market research assistant selecting Wikipedia articles for industry relevance.

INDUSTRY:
{industry}

MAIN INDUSTRY PAGE (ANCHOR MEANING):
Title: {main_title}
Summary: {main_summary[:800]}

CANDIDATE TITLES:
{candidate_block}

INSTRUCTIONS:
- Infer the intended industry meaning from the main page summary.
- Select the five most relevant titles from the candidate list.
- Include the main industry page if it is present in the candidate list.
 - Prioritize industry-level overviews, market/sector concepts, value chain elements, and ecosystem-level topics.
 - Reject company-specific pages; focus on holistic industry coverage.
 - Avoid country-specific pages; prefer global or general industry topics.
- Avoid unrelated domains (e.g., medical databases), entertainment items, quizzes, or disambiguation pages.

OUTPUT FORMAT:
Return exactly five lines. Each line must be one title copied exactly from the candidate list.
"""
    response = chain.invoke({"full_message": prompt})
    text = extract_text(response)
    picked = []
    candidate_set = set(candidates)
    for line in text.splitlines():
        title = line.strip().lstrip("-").strip()
        if title in candidate_set and title not in picked:
            picked.append(title)
        if len(picked) == 5:
            break
    return picked

# Session state for step-by-step flow
if "step" not in st.session_state:
    st.session_state.step = 1

def reset_workflow() -> None:
    for k in ["step", "industry", "wiki_titles", "wiki_urls", "report"]:
        st.session_state.pop(k, None)
    st.session_state.step = 1

# Initialize model when API key is present
llm = None
chain = None
if api_key:
    os.environ["GOOGLE_API_KEY"] = api_key
    llm = ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_output_tokens=int(max_output_tokens),
    )
    prompt = ChatPromptTemplate.from_messages([("human", "{full_message}")])
    chain = prompt | llm

st.subheader(f"Step {st.session_state.step} of 3")
st.progress(st.session_state.step / 3)

# STEP 1: industry input + validation
if st.session_state.step == 1:
    st.write("**Q1.** Please enter an industry you would like to analyze.")
    st.caption("Examples: airlines, semiconductors, coffee, electric vehicles.")
    industry = st.text_input("Industry", value=st.session_state.get("industry", ""))

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Continue to Step 2"):
            if not industry or not industry.strip():
                st.warning("Please provide an industry to continue.")
            else:
                st.session_state.industry = industry.strip()
                st.session_state.step = 2
                st.rerun()
    with col2:
        pass

# STEP 2: show five Wikipedia URLs
elif st.session_state.step == 2:
    industry = st.session_state.get("industry", "").strip()
    if not industry:
        st.warning("No industry found. Go back and enter an industry.")
        if st.button("Back to Step 1"):
            st.session_state.step = 1
            st.rerun()
    else:
        st.write(f"**Q2.** Five most relevant Wikipedia page URLs for: **{industry}**")

        with st.spinner("Searching Wikipedia…"):
            candidates = find_wikipedia_candidates(industry, lang="en", limit=25)
            main_title, main_summary = get_main_industry_page(industry)
            if main_title:
                st.caption(f"Anchor page: {main_title} — {main_summary.split('. ')[0][:160]}")

            if not llm or not chain:
                st.error("Please enter your API key in the sidebar.")
                st.stop()

            titles = select_titles_with_llm(industry, main_title, main_summary, candidates)
            if not titles:
                titles = candidates[:5]

        if len(titles) < 5:
            while len(titles) < 5:
                titles.append("Industry")

        urls = [wiki_url_from_title(t, lang="en") for t in titles[:5]]
        st.session_state.wiki_titles = titles[:5]
        st.session_state.wiki_urls = urls[:5]

        for i, u in enumerate(urls[:5], start=1):
            st.write(f"{i}. {u}")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Continue to Step 3"):
                st.session_state.step = 3
                st.rerun()
        with col2:
            if st.button("Back to Step 1"):
                st.session_state.step = 1
                st.rerun()

# STEP 3: generate report from the five pages
elif st.session_state.step == 3:
    industry = st.session_state.get("industry", "").strip()
    titles = st.session_state.get("wiki_titles", [])
    urls = st.session_state.get("wiki_urls", [])

    if not industry or not titles or len(titles) < 5:
        st.warning("Missing industry or Wikipedia pages. Please redo Step 1–2.")
        if st.button("Back to Step 1"):
            st.session_state.step = 1
            st.rerun()
    else:
        st.write(f"**Q3.** Industry report for: **{industry}**")
        st.caption("Constraints: < 500 words; based on the five Wikipedia pages from Step 2.")

        if st.button("Generate report"):
            if not llm or not chain:
                st.error("Please enter your API key in the sidebar.")
            else:
                with st.spinner("Loading Wikipedia pages & drafting report…"):
                    excerpts = []
                    for i, title in enumerate(titles[:5], start=1):
                        try:
                            page = wikipedia.page(title, auto_suggest=False)
                            text = (page.content or "").strip()
                            text = text[:2500]
                            excerpts.append(f"[Source {i}] {title}\n{text}")
                        except Exception as e:
                            excerpts.append(
                                f"[Source {i}] {title}\nNot covered in the provided Wikipedia excerpts (error loading page: {e})."
                            )

                    evidence_block = "\n\n".join(excerpts)
                    sources_block = "\n".join(f"- {u}" for u in urls[:5])

                    full_message = f"""INSTRUCTIONS:
{SYSTEM_PROMPT}

INDUSTRY:
{industry}

WIKIPEDIA EXCERPTS (ONLY FACTUAL BASIS):
{evidence_block}

REQUIRED OUTPUT:
- Write an industry report under 500 words.
- Use headings and bullet points where helpful.
- Include the following sections:
  1) Definition & scope
  2) Market structure / value chain
  3) Key segments
  4) Major organisations/players mentioned
  5) Demand drivers / use cases
  6) Risks/constraints (only if covered)
- End with exactly:
Sources used:
{sources_block}
"""

                    try:
                        response = chain.invoke({"full_message": full_message})
                        report = cap_500_words(extract_text(response))
                    except Exception as exc:
                        st.error(f"LLM call failed: {exc}")
                        report = "Sorry, I couldn't complete that request."

                    st.session_state.report = report

        if st.session_state.get("report"):
            st.markdown(st.session_state.report)
            st.caption(f"Word count: {word_count(st.session_state.report)} (hard-capped at 500)")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Back to Step 2"):
                st.session_state.step = 2
                st.rerun()
        with col2:
            pass
