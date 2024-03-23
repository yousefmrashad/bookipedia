from root_config import *
from utils.init import *
# ================================================== #
from utils.db_config import DB
from RAG.web_weaviate import WebWeaviate
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from RAG.web_researcher import WebResearchRetriever
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from RAG.web_researcher import WebResearchRetriever
from preprocessing.embeddings_class import AnglEEmbedding
from RAG.weaviate_class import Weaviate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class RagPipeline:
    
    def __init__(self, ) -> None:
        open_ai_key = "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"
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0, streaming=True, openai_api_key=open_ai_key)
        self.client = DB().connect()
        self.embedding_model = AnglEEmbedding()
        self.web_client = weaviate.connect_to_local()
        self.web_db = WebWeaviate(self.web_client, embedder=self.embedding_model)



    def generate_retrieval_query(self, user_prompt:str, chat_summary:str):

        # Define the retrieval query template
        RETRIEVING_QUERY_PROMPT = ChatPromptTemplate.from_template(
            """You are an assistant tasked with optimizing the retrieval.
            Given a user prompt and a summary of the chat, your task is to generate a retrieval query that is both concise and effective.
            The query should be designed to retrieve the most relevant information efficiently.
            Inputs: User Prompt: '''{user_prompt}'''
            Chat Summary: '''{chat_summary}'''""",
        )

        # Create the retrieval query chain
        retrieving_query_chain = RETRIEVING_QUERY_PROMPT | self.llm

        # Invoke the retrieval query chain with the user prompt and chat summary
        retrieving_query = retrieving_query_chain.invoke({"user_prompt": user_prompt, "chat_summary": chat_summary})

        # Return the content of the retrieval query
        return retrieving_query.content
    

    def generate_vecdb_context(self, retrieval_query: str, book_ids: list[str]):
        # Assuming DB().connect() returns a connection object that can be used directly with Weaviate
        client = DB().connect()
        
        # Initialize Weaviate instance
        db = Weaviate(client, embedder=self.embedding_model)
        
        # Perform similarity search
        docs = db.similarity_search(query=retrieval_query, source_ids=book_ids)
        
        # Use list comprehensions to create content and metadata lists
        content = [doc.page_content for doc, _ in docs]
        metadata = [f"book_id: {doc.metadata['source_id']}, page_no: {doc.metadata['page_no']}" for doc, _ in docs]
        
        # Ensure the Weaviate instance is properly closed or managed
        # This is a placeholder. You might need to implement a proper context management or close method
        # db.close()
        
        return content, metadata
    def generate_web_context(self, retrieval_query: str):
        # Initialize the retriever
        retriever = WebResearchRetriever.from_llm( vectorstore=self.web_db , llm=self.llm,  search=self.search)
        docs = retriever.get_relevant_documents(retrieval_query)
        content, metadata = list() ,list()

        for doc in docs:
            content.append(doc.page_content)
            metadata.append(doc.metadata['source'])
        return content, metadata
    
    def generate_context(self, user_prompt: str, chat_summary: str, book_ids: list[str] = None, enable_web_retrieval=True) :
        #  Generate the retrieval query
        retrieval_query = self.generate_retrieval_query(user_prompt, chat_summary)
        
        #  Generate context from the Weaviate Vector Database
        vecdb_context, vecdb_metadata = [] , []
        if book_ids:
            vecdb_context, vecdb_metadata = self.generate_vecdb_context(retrieval_query, book_ids)
        
        # Optionally generate context from the web (if book_ids are provided)
        web_context , web_metadata = [] , []
        
        if book_ids:
            web_context, web_metadata = self.generate_web_context(retrieval_query)
        
        # Combine the contexts
        context = vecdb_context + web_context
        metadata = vecdb_metadata + web_metadata
        
        #****************************************** 
        # TODO: USE RERANKER TO FILTER CHUNCKS
        #******************************************
        
        return context, metadata
    

    def generate_chat_summary(self, response: str, retrieving_query: str, chat: str):
        # Define the chat summary prompt template
        CHAT_SUMMARY_PROMPT = ChatPromptTemplate.from_template(
            """You are an assistant tasked with optimizing the process of summarizing user chat .
            Given the previous chat history, user prompt, and LLM response, summarize the chat effectively.
            Focus on new information and important highlights.
            Previous Chat History: '''{chat}'''
            User Prompt: '''{retrieving_query}'''
            LLM Response: '''{response}'''""",
        )

        # Create the chat summary chain
        chat_summary_chain = CHAT_SUMMARY_PROMPT | self.llm

        # Invoke the chat summary chain with the chat, retrieving_query, and response
        chat_summary = chat_summary_chain.invoke({"chat": chat, "retrieving_query": retrieving_query, "response": response})

        # Return the content of the chat summary
        return chat_summary.content
    
    def generate_answer(self, user_prompt: str, chat_summary: str, book_ids: list[str] = None, enable_web_retrieval=True ):
        context, metadata = self.generate_context(user_prompt=user_prompt, chat_summary= chat_summary,
                                                  book_ids= book_ids, enable_web_retrieval= enable_web_retrieval)
        # TODO:
        #  1- generate answer (stream the output)
        #  2- post processing: source provision and linking
        pass