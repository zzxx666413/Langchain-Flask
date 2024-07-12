import os
os.environ["OPENAI_API_KEY"] = ""
apikey = os.environ["OPENAI_API_KEY"]
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI,VectorDBQA
from langchain.document_loaders import DirectoryLoader

from flask import Flask, render_template, request, jsonify

from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory 
from langchain.chains import  ConversationalRetrievalChain

app = Flask(__name__)

from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.3,
    max_tokens=256,
    )                                                  
# 載入資料夾中所有TXT檔案
loader = DirectoryLoader('D:\Langchain_RAG_Docker\RAG_Data', glob='**/*.txt')
# 將資料轉成document物佚，每個檔案會為作為一個document
documents = loader.load()
# 初始化載入器
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
# 切割加载的 document
split_docs = text_splitter.split_documents(documents)
# 初始化 openai 的 embeddings 物件
embeddings = OpenAIEmbeddings()
# 將 document 透過 openai 的 embeddings 物件計算 embedding向量資料暫時存入 Chroma 向量資料庫用於後續的搜尋
docsearch = Chroma.from_documents(split_docs, embeddings)
# 建立回答物件
# llm = ChatOpenAI(model_name="gpt-3.5-turbo",temperature=0.1)
qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=docsearch, return_source_documents=True)


retriever = docsearch.as_retriever(search_type = "similarity",search_kwargs={"k":5})
#測試2
memory = ConversationBufferMemory (llm=llm , output_key='answer', memory_key='chat_history',return_messages=True)
converstation = ConversationalRetrievalChain.from_llm(llm=llm,retriever=retriever, memory=memory)

@app.route('/')
def index():
    return render_template('index.html')
#@app.route('/ask', methods=['POST'])
@app.route('/ask', methods=['GET'])
def ask_question():
    #question = request.form['question']
    question = request.args.get('question')
    # result = qa({"query": question})
    result = converstation({"question":question})
    # return result['result']
    return result['answer']


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
