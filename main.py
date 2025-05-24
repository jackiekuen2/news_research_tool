import os
import streamlit as st
import time
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv


# Loading environment variables
load_dotenv()
azure_endpoint = os.getenv("ENDPOINT_URL")
azure_key = os.getenv("API_KEY")

# llm deployment
# model_name = "gpt-4o-mini"
deployment_name = "gpt-4o-mini"
api_version = "2024-05-01-preview"

# embedding deployment
embedding_deployment_name = "text-embedding-3-small"
embedding_api_version = "2024-12-01-preview",


# Initialise the llm and embeddings
llm = AzureChatOpenAI(
    azure_endpoint = azure_endpoint,
    api_key = azure_key,
    azure_deployment = deployment_name,
    model_name = deployment_name,
    api_version = api_version,
    max_tokens = 300,
    temperature = 0.3
)
if not llm:
    st.error("Error: LLM not initialized. Please check your Azure OpenAI credentials.")
    st.stop()

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint = azure_endpoint,
    api_key = azure_key,
    azure_deployment = embedding_deployment_name
    )
if not embeddings:
    st.error("Error: Embeddings not initialized. Please check your Azure OpenAI credentials.")
    st.stop()



# UI
st.title("News Bot: News Research Tool")
st.sidebar.title("News article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()


if process_url_clicked:
    # load data
    loader = WebBaseLoader(web_paths=urls)
    main_placeholder.text("Loading data...✅✅✅")
    data = loader.load()

    # split data
    r_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", ","],
    chunk_size = 500,
    chunk_overlap = 0
    )
    main_placeholder.text("Splitting data...✅✅✅")
    docs = r_splitter.split_documents(data)

    # creat embeddings and save it to FAISS index
    vectorstore = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Budding embedding vectorstore...✅✅✅")
    time.sleep(2)

    vectorstore.save_local("faiss_vectorstore")
    main_placeholder.text("FAISS vectorstore created!✅✅✅")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists("faiss_vectorstore"):
        try:
            # Load FAISS indext
            faiss_vectorstore = FAISS.load_local(
                "faiss_vectorstore", 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            main_placeholder.text("FAISS vectorstore loaded!✅✅✅")
        
        except Exception as e:
            main_placeholder.text(f"Error loading FAISS vectorstore: {e}")
            st.stop()

        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=faiss_vectorstore.as_retriever())
        result = chain.invoke({"question": query}, return_only_outputs=True)
        st.header("Question:")
        st.write(query)
        st.header("Answer:")
        st.write(result["answer"])

        # Display the sources
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)