# ui.py
import streamlit as st
import importlib
import time
import chatbot

st.title("🚀 Space Facts Using RAG")

st.markdown("""
<style>
div[data-testid="stExpander"] summary {
    padding: 20px 16px;
    display: flex;
    align-items: center;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

st.selectbox("", ["📚 Available Space Facts"])

query = st.text_input("Enter your question about space: ", placeholder="e.g., What is the Hubble Space Telescope?")

st.sidebar.title("Model Configuration")
modeltype = st.sidebar.radio("Select LLM Model:", ["Ollama Llama2", "Ollama Llama3.2"], index=1)
embedmodel = st.sidebar.radio("Select Embedding Model:", ["Chroma Default", "Nomic Emebed Text(Ollama)"], index=1)

# if query:
#     with st.spinner("Processing your query..."):
#         importlib.reload(chatbot)  # ensures the latest code is used
#         answer, elapsed = chatbot.run_chatbot(query, modeltype, embedmodel)
#         time.sleep(0.5)  # optional UX delay

#     st.write(answer)
#     st.info(f"Time taken: {elapsed:.2f} seconds")


if query:
    res, ref = st.columns(2)

    with st.spinner("Processing your query..."):
        time.sleep(0.5)
        response, relchunks = chatbot.run_chatbot(query)

    # ---------------- RESPONSE COLUMN ----------------
    with res:
        st.subheader("🤖 Response")

        placeholder = st.empty()
        current_text = ""
        buffer = ""

        for chunk in response:
            if hasattr(chunk, "message"):
                buffer += chunk.message.content

                if len(buffer) > 20 or buffer.endswith((" ", ".", "\n")):
                    current_text += buffer
                    placeholder.write(current_text)
                    buffer = ""
                    time.sleep(0.015)

        if buffer:
            current_text += buffer
            placeholder.write(current_text)

    # ---------------- REFERENCES COLUMN ----------------
    with ref:
        st.subheader("📖 References Used")

        for chunk in relchunks:
            st.write(f"• {chunk}")

    st.write("---")

    # ---------------- AUGMENTED PROMPT ----------------
    st.subheader("Augmented Prompt")

    context_text = "\n\n".join(relchunks)

    st.code(f"""
Context:
{context_text}

Question:
{query}

Answer:
""")

    st.write("---")

    # ---------------- MODEL CONFIG ----------------
    st.subheader("Model Configuration")

    st.write(f"LLM Model: {modeltype}")
    st.write(f"Embedding Model: {embedmodel}")

