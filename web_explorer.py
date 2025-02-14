import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers.web_research import WebResearchRetriever

import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyArkdTOXTwk4009wc_-ub5c6vfQ97O_5TI" # Get it at https://console.cloud.google.com/apis/api/customsearch.googleapis.com/credentials
os.environ["GOOGLE_CSE_ID"] = "a57ccd7b2fa0b4777" #"AIzaSyA1puf3hG62YxDhHX9EFgGkPzmCs1L5wc4"# #"https://cse.google.com/cse.js?cx=01b39422d16714754" # Get it at https://programmablesearchengine.google.com/
# # os.environ["OPENAI_API_BASE"] = "https://api.openai.com/v1"
# os.environ["OPENAI_API_KEY"] = "" # Get it at https://beta.openai.com/account/api-keys

st.set_page_config(page_title="Interweb Explorer", page_icon="🌐")

def settings():

    # Vectorstore
    # import faiss
    # from langchain_community.vectorstores import FAISS 
    import chromadb
    from langchain_community.vectorstores import Chroma 
    # from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from langchain_community.docstore import InMemoryDocstore  
    # embeddings_model = OpenAIEmbeddings()  
    embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")#, task_type="retrieval_query") 
    # embedding_size = 1536  
    # index = faiss.IndexFlatL2(embedding_size)  
    # vectorstore_public = FAISS(embedding_function=embeddings_model,index=index, docstore=InMemoryDocstore({}),index_to_docstore_id={})
    # vectorstore_public = FAISS.from_documents(InMemoryDocstore({}),embeddings_model.embed_query)

    persist_client = chromadb.EphemeralClient()
    collection = persist_client.get_or_create_collection("my_collection")
    vectorstore_public = Chroma(
                                client=persist_client,
                                collection_name="my_collection",
                                embedding_function=embeddings_model)
    # vectorstore_public = Chroma(embedding_function=embeddings_model.embed_documents, persist_directory="./chroma_db_oai")

    # vectorstore_public = Chroma("langchain_store", embeddings_model.embed_query)

    # LLM
    # from langchain.chat_models import ChatOpenAI
    # llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0, streaming=True)

    # from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-pro")


    # Search
    from langchain_community.utilities import GoogleSearchAPIWrapper
    search = GoogleSearchAPIWrapper()   

    # Initialize 
    web_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore_public,
        llm=llm, 
        search=search, 
        num_search_results=3
    )

    return web_retriever, llm

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.info(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")

    def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        # self.container.write(documents)
        for idx, doc in enumerate(documents):
            source = doc.metadata["source"]
            self.container.write(f"**Results from {source}**")
            self.container.text(doc.page_content)


st.sidebar.image("img/ai.png")
st.header("`Interweb Explorer`")
st.info("`I am an AI that can answer questions by exploring, reading, and summarizing web pages."
    "I can be configured to use different modes: public API or private (no data sharing).`")

# Make retriever and llm
if 'retriever' not in st.session_state:
    st.session_state['retriever'], st.session_state['llm'] = settings()
web_retriever = st.session_state.retriever
llm = st.session_state.llm

# User input 
question = st.text_input("`Ask a question:`")

if question:

    # Generate answer (w/ citations)
    import logging
    logging.basicConfig()
    logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)    
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_retriever)

    # Write answer and sources
    retrieval_streamer_cb = PrintRetrievalHandler(st.container())
    answer = st.empty()
    stream_handler = StreamHandler(answer, initial_text="`Answer:`\n\n")
    result = qa_chain({"question": question},callbacks=[retrieval_streamer_cb, stream_handler])
    answer.info('`Answer:`\n\n' + result['answer'])
    st.info('`Sources:`\n\n' + result['sources'])
