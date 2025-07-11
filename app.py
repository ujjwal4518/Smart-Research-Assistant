import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

# LangChain core tools
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory

# 🔐 Load API keys
load_dotenv()
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Load LLM + Embeddings
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="mixtral-8x7b")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# UI setup
st.set_page_config(page_title="Smart Research Assistant", layout="wide")
st.title("📚 Smart Research Assistant")
st.write("Upload a PDF or TXT file and interact with its content using Ask Anything or Challenge Me mode!")

# Session memory
session_id = st.text_input("Session ID (optional for multi-chat memory):", value="default_session")
if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session: str) -> BaseChatMessageHistory:
    if session not in st.session_state.store:
        st.session_state.store[session] = ChatMessageHistory()
    return st.session_state.store[session]

# File upload
uploaded_files = st.file_uploader("📄 Upload your documents (PDF or TXT)", type=["pdf", "txt"], accept_multiple_files=True)
documents = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        loader = PyPDFLoader(temp_path) if uploaded_file.name.endswith(".pdf") else TextLoader(temp_path)
        documents.extend(loader.load())

    # Split and embed
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)  # ✅ FAISS used here
    retriever = vectorstore.as_retriever()

    # Auto-summary
    st.subheader("🔍 Auto-Summary")
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the following document in under 150 words:\n\n{context}")
    ])
    summary_chain = create_stuff_documents_chain(llm, summary_prompt)
    summary = summary_chain.invoke({"context": chunks})
    st.success(summary)

    # Ask Anything
    with st.expander("💬 Ask Anything", expanded=True):
        user_question = st.text_input("What would you like to know?")
        if user_question:
            standalone_prompt = ChatPromptTemplate.from_messages([
                ("system", "Turn the user's question into a standalone one, using chat history if needed."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            history_aware_retriever = create_history_aware_retriever(llm, retriever, standalone_prompt)

            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", 
                 "You're a helpful assistant. Use the context to answer in 3 sentences max. "
                 "Always provide justification like: 'This is supported by Section 2.1'.\n\n{context}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ])
            qa_chain = create_stuff_documents_chain(llm, qa_prompt)
            rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

            chat_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )

            result = chat_chain.invoke(
                {"input": user_question},
                config={"configurable": {"session_id": session_id}}
            )

            st.markdown(f"**🧠 Answer:** {result['answer']}")

    # Challenge Me
    with st.expander("🧠 Challenge Me"):
        if st.button("🎯 Generate Challenge Questions"):
            challenge_prompt = ChatPromptTemplate.from_messages([
                ("system", "Generate 3 logic-based or comprehension questions from this document:\n\n{context}")
            ])
            challenge_chain = create_stuff_documents_chain(llm, challenge_prompt)
            output = challenge_chain.invoke({"context": chunks})

            text = output.get("output", "") if isinstance(output, dict) else str(output)
            questions = [line.strip("-•1234567890. ").strip() for line in text.split("\n") if "?" in line][:3]

            if questions:
                st.session_state["challenge_questions"] = questions
            else:
                st.error("❌ Could not extract valid questions. Try again or check the document.")

        if "challenge_questions" in st.session_state:
            st.subheader("📌 Your Challenge Questions")
            user_responses = []
            for i, question in enumerate(st.session_state["challenge_questions"]):
                st.markdown(f"**Q{i+1}:** {question}")
                user_input = st.text_input(f"Your Answer to Q{i+1}:", key=f"challenge_answer_{i}")
                user_responses.append((question, user_input))

            if st.button("✅ Evaluate My Answers"):
                for i, (question, answer) in enumerate(user_responses):
                    eval_prompt = ChatPromptTemplate.from_messages([
                        ("system", 
                         "You are a tutor evaluating a student's response. Evaluate the answer based on the document. "
                         "Say whether it's correct, and explain briefly with reference to the content.\n\n{context}"),
                        ("human", f"Question: {question}\nAnswer: {answer}")
                    ])
                    eval_chain = create_stuff_documents_chain(llm, eval_prompt)
                    feedback = eval_chain.invoke({"context": chunks})
                    st.markdown(f"**Feedback for Q{i+1}:** {feedback}")
                    st.markdown("---")

else:
    st.info("⬆️ Upload a PDF or TXT file to begin.")
