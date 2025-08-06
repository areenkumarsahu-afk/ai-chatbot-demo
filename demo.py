import streamlit as st 
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import tempfile
import os 
import fitz

MAX_QUERIES=3

def load_docs(uploaded_file):
    if uploaded_file.size>100000:
        st.warning("File too large for demo. Please upload a smaller file.")
        return None
    file_ext=os.path.splitext(uploaded_file.name)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path=tmp_file.name
    if file_ext=='.pdf':
        try:
            raw_text=""
            for page in doc:
                raw_text+=page.get_text()
        except Exception as e:
            st.error("Error reading PDF:"+str(e))
    elif file_ext=='.txt':
        with open(file_path,'r',encoding='utf-8') as f:
            raw_text=f.read()
    else:
        st.warning("Unsupported file type. Please upload a .txt or .pdf file.")
        return None
    text_splitter=CharacterTextSplitter(separator='/n',chunk_size=1000,chunk_overlap=200)
    docs=text_splitter.create_documents([raw_text])
    return docs

def get_answers(docs,query):
    embeddings=OpenAIEmbeddings()
    vectorstore=FAISS.from_documents(docs,embeddings)
    retriever=vectorstore.as_retriever()
    relevant_docs=retriever.get_relevent_documents(query)
    llm=ChatOpenAI(temprature=0)
    chain=load_qa_chain(llm,chain_type='stuff')
    answer=chain.run(input_documents=relevant_docs,question=query)
    max_length=200
    if len(answer)>max_length:
        answer=answer[:max_length]+"..."
    return answer

st.title("Demo: AI Chatbot with Your Documents")
st.markdown("*Demo version - limited file size & only 3 questions per session.*")
uploaded_file=st.file_uploader("Upload a file")
if 'query_count' not in st.session_state:
    st.session_state.query_count=0
if uploaded_file:
    docs=load_docs(uploaded_file)
    if docs:
        query=st.text_input("Ask something:")
        if query:
            if st.session_state.query_count<MAX_QUERIES:
                answer=get_answers(docs,query)
                st.write("Answer:",answer)
                st.session_state.query_count+=1
                remaining=MAX_QUERIES-st.session_state.query_count
                st.info(f"You have {remaining} queries remaining in demo.")
            else:

                st.error("Demo limit reached.[Request full access](https://www.notion.so/AI-Chatbots-That-Work-While-You-Sleep-245eb1fcbdfb80678680f57248d685c8?source=copy_link)")
