from flask import Flask, render_template, jsonify, request
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.helper import download_embeddings
from src.prompt import *
import os


app = Flask(__name__)


load_dotenv()

print("PINECONE:", os.getenv("PINECONE_API_KEY")[:10])
print("GOOGLE:", os.getenv("GOOGLE_API_KEY")[:10])

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

embeddings = download_embeddings()

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    embedding=embeddings,
    index_name=index_name
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)



question_answer_chain = create_stuff_documents_chain(
    llm,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)


@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]

        print("Question:", msg)

        response = rag_chain.invoke({"input": msg})

        print("Answer:", response["answer"])

        return str(response["answer"])

    except Exception as e:
        print("ERROR:", e)
        return str(e)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)