
import streamlit as st
import sqlite3
import pandas as pd
import json
import re
import os
from datetime import date
from typing import TypedDict, List, Dict, Any

from openai import OpenAI
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kartify Support",
    page_icon="🛒",
    layout="centered",
)


# Load the JSON file and extract values
file_name = 'config.json'
with open(file_name, 'r') as file:
    config = json.load(file)
    OPENAI_API_KEY = config.get("OPENAI_API_KEY") # Loading the API Key
    OPENAI_API_BASE = config.get("OPENAI_API_BASE") # Loading the API Base Url


# Storing API credentials in environment variables
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ["OPENAI_BASE_URL"] = OPENAI_API_BASE

# ── LLMs ─────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_llms():
    llm          = ChatOpenAI(model_name="gpt-4o")
    evaluate_llm = ChatOpenAI(model_name="gpt-4o")
    return llm, evaluate_llm

llm, evaluate_llm = load_llms()

# ── State ─────────────────────────────────────────────────────────────────────
class OrderState(TypedDict):
    cust_id:          str
    order_id:         str
    order_context:    str
    query:            str
    raw_agent_response: str
    final_response:   str
    history:          List[Dict[str, str]]
    intent:           str
    evaluation:       Dict[str, float]
    guard_result:     str
    conv_guard_result: str
    retry_count: int

# ── Conversation memory ───────────────────────────────────────────────────────
class ConversationMemory:
    def __init__(self):
        self.history: List[Dict[str, str]] = []

    def add(self, msg: dict):
        self.history.append(msg)

    def get(self) -> List[Dict[str, str]]:
        return self.history

    def clear(self):
        self.history = []

# ── SQL tool ──────────────────────────────────────────────────────────────────
@tool
def fetch_order_details(order_id: str) -> str:
    """
    Fetch all order details for a given order_id from the Kartify database.
    Use this tool whenever the customer's query requires order-specific information.
    Returns a formatted string of order details, or an error message if not found.
    """
    if not re.match(r"^O\d+$", order_id.strip()):
        return f"Invalid order ID format: '{order_id}'. Expected format: O followed by digits (e.g. O40327)."
    try:
        with sqlite3.connect("kartify.db") as conn:
            df = pd.read_sql_query(
                "SELECT * FROM orders WHERE order_id = ?",
                conn,
                params=(order_id.strip(),),
            )
        if df.empty:
            return f"No order found with ID {order_id}."
        return df.to_string(index=False)
    except Exception as e:
        return f"Database error while fetching order {order_id}: {str(e)}"

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a Kartify Customer Service Agent. You help customers with questions about their orders.

You have access to the following tool:
  fetch_order_details(order_id) — retrieves all order information from the database.

Follow the ReAct pattern strictly:
  Thought: <your reasoning about what to do next>
  Action: fetch_order_details with the order_id from the customer's query
  Observation: <tool result>
  Thought: <reason about the observation and form your answer>
  Final Answer: <short, polite, conversational reply — no greetings, no sign-off>

Policy rules (apply before writing Final Answer):
  - If Actual Delivery has a date, that means the order has been delivered on that particular date  
  - If actual_delivery is null, the order has not been delivered yet — do not mention return/replacement eligibility.
  - Share tracking details if the user has some doubts regarding the delivery of the order
  - Only mention return or replacement terms when the customer explicitly asks, and calculate whether that is possible and respond accordingly.
  - Never invent data. Only use what the tool returned.
  - Keep the Final Answer concise and empathetic.
  - Never reveal internal data fields or technical reasons in your reply (e.g. do not mention that actual_delivery is null or any other raw database values).
  - If a customer asks why their order hasn't arrived yet, only state that it is still on the way and share the expected delivery date — never explain the technical reason behind the delay status.
  - Never promise or suggest an early delivery. Always communicate the expected delivery date as-is without implying it could arrive sooner.
  - If the order has not arrived by the expected delivery date, empathetically acknowledge the delay and advise the customer to wait a little longer or contact support — do not speculate on reasons.

Answer Guidelines:
  - Only answer what is asked in the Query
  - Double-check that all calculations, answers, and outputs match the tool's results before generating the final response.
  - Check the Previous conversation (if any) before generating the reply
  """

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_json_from_llm(text: str):
    for pattern in [r"```json\s*(.*?)\s*```", r"\{.*\}", r"\[.*\]"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1) if "```" in pattern else match.group(0))
            except Exception:
                continue
    return json.loads(text)

# ── Order agent ───────────────────────────────────────────────────────────────
def order_agent(query: str, order_id: str, history: list) -> tuple:
    llm_with_tools = llm.bind_tools([fetch_order_details])

    history_text = ""
    if history:
        history_text = "\nPrevious conversation:\n" + "\n".join(
            f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history
        ) + "\n"

    user_content = (
        f"Previous Conversation:{history_text}\n"
        f"Customer query: {query}\n"
        f"Order ID: {order_id}\n"
        f"Today's date: 25 July"
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    order_context = ""
    max_iterations = 5

    for _ in range(max_iterations):
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

        if not getattr(ai_msg, "tool_calls", None):
            break

        for tc in ai_msg.tool_calls:
            if tc["name"] == "fetch_order_details":
                result = fetch_order_details.invoke(tc["args"])
                order_context = result
                messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    final_response = ai_msg.content.strip()
    for prefix in ("Final Answer:", "final answer:"):
        if final_response.lower().startswith(prefix.lower()):
            final_response = final_response[len(prefix):].strip()
            break

    return order_context, final_response

# ── Node functions ────────────────────────────────────────────────────────────
def user_input_node(state: OrderState):
    return state

def memory_node(state: OrderState):
    st.session_state.conversation_memory.add(
        {"user": state["query"], "assistant": state["final_response"]}
    )
    return state

def order_agent_node(state: OrderState):
    order_context, final_response = order_agent(
        query=state["query"],
        order_id=state["order_id"],
        history=state["history"],
    )


    return {
        "order_context": order_context,
        "final_response": final_response,
        "retry_count": state["retry_count"] + 1,
    }

def intent_node(state: OrderState):
    prompt = f"""You are an intent classifier for customer service queries. Classify the user's query into one of these categories.
Return ONLY the numeric ID (0, 1, 2, or 3). No explanation.

0 - Escalation: user is very angry/frustrated, wants a human now.
1 - Exit: user is ending the conversation ("Thanks", "Bye", "Resolved").
2 - Process: clear, actionable order query — proceed normally.
3 - Random/Unrelated/Vulnerable: out-of-scope or potentially unsafe query.

Query: {state['query']}"""
    result = llm.invoke([HumanMessage(content=prompt)]).content.strip()

    match = re.search(r"[0-3]", result)
    intent = match.group(0) if match else "3"

    return {"intent": intent}
    
def router_node(state: OrderState):
    return "order_agent" if state["intent"] == "2" else "exit_node"

def exit_node(state: OrderState):
    if state.get("final_response"):
        return {}

    mapping = {
        "0": "Sorry for the inconvenience. A human support agent will assist you shortly.",
        "1": "Thank you! I hope I was able to assist with your query.",
        "3": "Apologies, I'm currently only able to help with information about your placed orders.",
    }

    return {"final_response": mapping.get(state["intent"], "")}

def evaluation_node(state: OrderState):
    prompt = f"""Evaluate the assistant's response to a customer query using the provided order context.

Context: {state['order_context']}
Query: {state['query']}
Response: {state['final_response']}

Instructions:
1. **Groundedness (0.0 to 1.0)**: Score based on how well the response is factually supported by the context.
                                - Score closer to 1 if all facts are accurate and derived from the context.
                                - Score closer to 0 if there is hallucination, guesswork, or any fabricated information.

2. **Precision (0.0 to 1.0)**: Score based on how directly and accurately the assistant addresses the query.
                                - Score closer to 1 if the response is concise, focused, and answers the exact user query.
                                - Score closer to 0 if it includes irrelevant details or misses the main point.

Output format (JSON only):
   groundedness: float between 0 and 1 ,
   precision: float between 0 and 1

Return ONLY JSON:
{{
 "groundedness": float,
 "precision": float
}}
"""
    try:
        raw = evaluate_llm.invoke([HumanMessage(content=prompt)]).content.strip()
        evaluation = extract_json_from_llm(raw)
    except Exception:
        evaluation = {"groundedness": 0.0, "precision": 0.0}
    return {"evaluation": evaluation}

def retry_router(state: OrderState):
    score = state.get("evaluation", {})
    retry_count = state.get("retry_count", 0)

    # Stop retrying after 3 attempts
    if retry_count >= 3:
        return "safety_check"

    if (
        score.get("groundedness", 0) < 0.75
        or score.get("precision", 0) < 0.75
    ):
        return "order_agent"

    return "safety_check"

def guard_node(state: OrderState):
    prompt = f"""You are a content safety assistant. Your task is to classify if the assistant's response is appropriate.
If the message contains:
- Requests for bank details, OTPs, account numbers
- Harassment or offensive tone
- Privacy concerns or unsafe advice
- Misunderstanding and miscommunication word
Return: BLOCK
Otherwise, return: SAFE
Response: {state["final_response"]}"""
    result = evaluate_llm.invoke([HumanMessage(content=prompt)]).content.strip()
    guard_result = result if result in ("BLOCK", "SAFE") else "SAFE"
    if guard_result == "BLOCK":
        return {
            "guard_result": guard_result,
            "final_response": "Your request is being forwarded to a customer support specialist.",
        }
    return {"guard_result": guard_result}

def guard_router(state: OrderState):
    return "exit" if state.get("guard_result") == "BLOCK" else "memory_save"

def conversational_guard_node(state: OrderState):
    prompt = f"""You are a conversation monitor AI. Review the conversation and detect if the assistant:
- Repeatedly gives the same advice to multiple questions
- Offers solutions the user did not ask for
- Ignores user frustration or contradictions

If any occur, return BLOCK. Otherwise return SAFE.

Conversation:
{state.get('history', [])}"""
    result = evaluate_llm.invoke([HumanMessage(content=prompt)]).content.strip()
    conv_result = result if result in ("BLOCK", "SAFE") else "SAFE"
    if conv_result == "BLOCK":
        return {
            "conv_guard_result": conv_result,
            "final_response": "Your request is being forwarded to a customer support specialist.",
        }
    return {"conv_guard_result": conv_result}

def conv_guard_router(state: OrderState):
    return "exit" if state.get("conv_guard_result") == "BLOCK" else "done"

# ── Build LangGraph ───────────────────────────────────────────────────────────
@st.cache_resource
def build_graph():
    g = StateGraph(OrderState)
    g.add_node("user_input",        user_input_node)
    g.add_node("intent_classifier", intent_node)
    g.add_node("order_agent",       order_agent_node)
    g.add_node("evaluate",          evaluation_node)
    g.add_node("safety_check",      guard_node)
    g.add_node("conv_safety_check", conversational_guard_node)
    g.add_node("memory_save",       memory_node)
    g.add_node("exit_node",         exit_node)

    g.set_entry_point("user_input")
    g.add_edge("user_input", "intent_classifier")
    g.add_conditional_edges(
        "intent_classifier", router_node,
        {"order_agent": "order_agent", "exit_node": "exit_node"},
    )
    g.add_edge("order_agent", "evaluate")
    g.add_conditional_edges(
        "evaluate", retry_router,
        {"order_agent": "order_agent", "safety_check": "safety_check"},
    )
    g.add_conditional_edges(
        "safety_check", guard_router,
        {"memory_save": "memory_save", "exit": "exit_node"},
    )
    g.add_edge("memory_save", "conv_safety_check")
    g.add_conditional_edges(
        "conv_safety_check", conv_guard_router,
        {"done": END, "exit": "exit_node"},
    )
    g.add_edge("exit_node", END)
    return g.compile()

order_graph = build_graph()

# ── Session state defaults ────────────────────────────────────────────────────
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = ConversationMemory()
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "chat_active" not in st.session_state:
    st.session_state.chat_active = False
if "cust_id" not in st.session_state:
    st.session_state.cust_id = ""
if "order_id" not in st.session_state:
    st.session_state.order_id = ""
if "orders_df" not in st.session_state:
    st.session_state.orders_df = None

# ── Helper: fetch customer orders ─────────────────────────────────────────────
def fetch_customer_orders(cust_id: str) -> pd.DataFrame | None:
    try:
        with sqlite3.connect("kartify.db") as conn:
            df = pd.read_sql_query(
                "SELECT order_id, product_description, order_status FROM orders WHERE customer_id = ?",
                conn,
                params=(cust_id.strip(),),
            )
        return df if not df.empty else None
    except Exception:
        return None

# ── Helper: run one turn through the graph ────────────────────────────────────
def run_turn(query: str, cust_id: str, order_id: str) -> str:
    state: OrderState = {
        "cust_id":           cust_id,
        "order_id":          order_id,
        "order_context":     "",
        "query":             query,
        "raw_agent_response": "",
        "final_response":    "",
        "history":           st.session_state.conversation_memory.get(),
        "intent":            "",
        "evaluation":        {},
        "guard_result":      "",
        "conv_guard_result": "",
        "retry_count": 0,
    }
    result = order_graph.invoke(state, config={"recursion_limit": 100})
    # Sync memory from the graph's memory_node writes
    # (memory_node uses st.session_state.conversation_memory directly)
    return result.get("final_response", "I'm sorry, I couldn't process that request.")

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

st.markdown(
    """
    <style>
    .block-container { max-width: 720px; }
    .chat-bubble-user {
        background: #e8f4fd;
        border-radius: 12px 12px 2px 12px;
        padding: 10px 14px;
        margin: 4px 0;
        max-width: 85%;
        margin-left: auto;
        color: #1a1a2e;
    }
    .chat-bubble-bot {
        background: #f4f4f4;
        border-radius: 12px 12px 12px 2px;
        padding: 10px 14px;
        margin: 4px 0;
        max-width: 85%;
        color: #1a1a2e;
    }
    .order-badge {
        display: inline-block;
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #856404;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 6])
with col_logo:
    st.markdown("## 🛒")
with col_title:
    st.markdown("## Kartify Customer Support")
    st.caption("AI-powered order query assistant")

st.divider()

# ── Phase 1: Customer ID lookup ───────────────────────────────────────────────
if not st.session_state.chat_active:
    st.markdown("### Step 1 — Enter your Customer ID")

    with st.form("customer_form"):
        cust_input = st.text_input(
            "Customer ID",
            placeholder="e.g. C1010",
            value=st.session_state.cust_id,
        )
        submitted = st.form_submit_button("🔍  Fetch Orders", use_container_width=True)

    if submitted and cust_input.strip():
        with st.spinner("Looking up your orders…"):
            df = fetch_customer_orders(cust_input.strip())
        if df is not None:
            st.session_state.cust_id  = cust_input.strip()
            st.session_state.orders_df = df
        else:
            st.error(f"No orders found for Customer ID **{cust_input.strip()}**. Please check and try again.")

    # ── Phase 2: Order selection ──────────────────────────────────────────────
    if st.session_state.orders_df is not None:
        st.markdown("### Step 2 — Select an Order")

        df = st.session_state.orders_df

        # Build display labels for the dropdown
        options = {
            f"{row['order_id']} - {row['product_description'][:45]}  [{row['order_status']}]": row["order_id"]
            for _, row in df.iterrows()
        }

        selected_label = st.selectbox(
            "Your orders",
            list(options.keys()),
            index=0,
        )
        selected_order_id = options[selected_label]

        # Preview card
        selected_row = df[df["order_id"] == selected_order_id].iloc[0]
        st.markdown(
            f"""
            <div style="background:#f8f9fa;border:1px solid #dee2e6;border-radius:8px;padding:12px 16px;margin:8px 0">
                <span class="order-badge">{selected_row['order_id']}</span>&nbsp;&nbsp;
                <strong>{selected_row['product_description']}</strong><br>
                <span style="font-size:0.85rem;color:#6c757d">Status: {selected_row['order_status']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("💬  Start Chat", use_container_width=True, type="primary"):
            st.session_state.order_id = selected_order_id
            st.session_state.chat_active = True
            st.session_state.conversation_memory.clear()
            st.session_state.chat_messages = []
            # Greeting
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": (
                    f"Hi! I'm your Kartify support assistant. "
                    f"I can see you're asking about order **{selected_order_id}**. "
                    f"How can I help you today?"
                ),
            })
            st.rerun()

# ── Phase 3: Chat interface ───────────────────────────────────────────────────
else:
    # Sidebar info
    with st.sidebar:
        st.markdown("### Active Session")
        st.markdown(f"**Customer:** `{st.session_state.cust_id}`")
        st.markdown(f"**Order:** `{st.session_state.order_id}`")
        st.divider()
        if st.button("🔄  New Session", use_container_width=True):
            st.session_state.chat_active = False
            st.session_state.chat_messages = []
            st.session_state.conversation_memory.clear()
            st.session_state.orders_df = None
            st.session_state.cust_id = ""
            st.session_state.order_id = ""
            st.rerun()
        st.divider()
        st.caption(
            "Powered by LangGraph · GPT-4o-mini\n\n"
            "Guardrails: Input intent · Output safety · Conversation monitor"
        )

    st.markdown(f"**Order** `{st.session_state.order_id}` — ask me anything about this order.")
    st.markdown("")

    # Render chat history
    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🛒"):
                st.markdown(msg["content"])

    # Chat input
    user_query = st.chat_input("Type your question here…")

    if user_query:
        # Display user message
        st.session_state.chat_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Run agent
        with st.chat_message("assistant", avatar="🛒"):
            with st.spinner("Thinking…"):
                response = run_turn(
                    query=user_query,
                    cust_id=st.session_state.cust_id,
                    order_id=st.session_state.order_id,
                )
            st.markdown(response)

        st.session_state.chat_messages.append({"role": "assistant", "content": response})

        # If the agent exits (intent 0/1/3), offer to restart
        exit_phrases = [
            "human support agent",
            "customer support specialist",
            "I hope I was able to assist",
            "only able to help with information",
        ]
        if any(p.lower() in response.lower() for p in exit_phrases):
            st.info("This conversation has ended. Use **New Session** in the sidebar to start over.")
