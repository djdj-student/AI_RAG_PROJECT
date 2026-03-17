import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import streamlit as st  
import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Whitley's RAG 助手", page_icon="🤖")
st.title("🤖 论文解析：DINOv2 与颜色选择性")

# DeepSeek 只负责最后的“理解与对话”
llm = ChatOpenAI(
    model='deepseek-chat', 
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE"),
    temperature=0.3
)

@st.cache_resource
def load_data():
    if not os.path.exists("data.pdf"):
        st.error("找不到 data.pdf，请确认它在项目根目录")
        return None
    
    with st.spinner("正在本地生成索引（首次运行需下载 100MB 模型，请耐心等待）..."):
        # 1. 加载并“切割”论文
        loader = PyPDFLoader("data.pdf")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
        split_docs = text_splitter.split_documents(docs)
        
        # 2. 本地 Embedding：CPU 亲自下场干活
        # 第一次运行会自动从镜像站下载 BAAI/bge-small-en-v1.5
        embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 3. 存入 FAISS
        return FAISS.from_documents(split_docs, embeddings)

vector_db = load_data()

if vector_db:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("问问关于论文的问题..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            qa = RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="stuff", 
                retriever=vector_db.as_retriever(search_kwargs={"k": 3})
            )
            with st.spinner("DeepSeek 正在思考答案..."):
                response = qa.invoke(prompt)["result"]
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})