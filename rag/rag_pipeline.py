from root_config import *
from utils.init import *

from rag.web_weaviate import WebWeaviate
from rag.web_researcher import WebResearchRetriever
from rag.weaviate_retriever import Weaviate
# ================================================== #

class RAGPipeline:
    
    def __init__(self,embedding_model:Embeddings, client:WeaviateClient) -> None:
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0, streaming=True, openai_api_key=OPEN_AI_KEY)
        self.embedding_model = embedding_model
        self.client = client
        self.db = Weaviate(self.client, self.embedding_model)
        self.web_db = WebWeaviate(self.client, embedder=self.embedding_model)
        self.search =  DuckDuckGoSearchAPIWrapper(backend='html')
        self.web_retriever = WebResearchRetriever.from_llm(vectorstore=self.web_db , llm=self.llm,  search=self.search)

    def get_page_text(self, doc_id: str, page_no: int) -> str:
        col = self.client.collections.get("bookipedia")
        filters = id_filter(doc_id) & page_filter(page_no)
        res = col.query.fetch_objects(filters= filters, limit = FETCHING_LIMIT, sort = SORT)
        texts = []
        for o in res.objects:
            texts.append(o.properties["text"])
        text = merge_chunks(texts)
        return text
    
    def summary_splitter(self, text:str, token_limit:int) -> list[str]:
        token_count = count_tokens(text)
        no_splits = math.ceil(token_count/token_limit)
        chunk_size = token_count//no_splits
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=128,
                                                        length_function=count_tokens, separators=SEPARATORS,
                                                        is_separator_regex=True)
        text_chunks = text_splitter.split_text(text)
        return text_chunks
    
    async def summarize_text(self, text:str, token_limit:int):
        # chunk pages_text with overlap. 
        text_chunks = self.summary_splitter(text, token_limit)

        # use llm to summarize each chunk
        sub_summaries = await gather(*(self.summary_chain.ainvoke({"text": text_chunk}) for text_chunk in text_chunks))
        joined_summary = '\n\n'.join(sub_summaries)
        return joined_summary
    
    async def summarize_pages(self, doc_id:str , page_nos : list[str], token_limit:int = 15872):
        #  get chunks and concatenate pages into a single string
        pages_text = '\n\n'.join([self.get_page_text(doc_id , page_no) for page_no in page_nos])
        
        SUMMARY_PROMPT = ChatPromptTemplate.from_template("""
            You are an assistant tasked with summarizing the provided text.
            Your goal is to create a concise summary that captures the essence of the original content, focusing on the most important points and key information.
            Please ensure that the summary is clear, informative, and easy to understand.
            THE INPUT TEXT: '''{text}'''  """)
        
        self.summary_chain = SUMMARY_PROMPT | self.llm | StrOutputParser()

        sub_summaries_joined = await self.summarize_text(pages_text, token_limit)

        if(count_tokens(sub_summaries_joined) > token_limit):
            sub_summaries_joined = await self.summarize_text(pages_text, token_limit)
        # Streaming Final summary. 
        return self.summary_chain.astream({"text": sub_summaries_joined})


    def generate_retrieval_query(self, user_prompt:str, chat_summary:str):

        # Define the retrieval query template
        RETRIEVING_QUERY_PROMPT = ChatPromptTemplate.from_template(
            """You are an assistant tasked with optimizing user prompt for better chuncks retrieval from web and vectorstore.
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
    

    
    
    def generate_context(self, user_prompt: str, chat_summary: str, doc_ids: list[str] = None, enable_web_retrieval=True) :
        
        def generate_vecdb_context(retrieval_query: str):
        
            docs = self.db.similarity_search(query=retrieval_query, source_ids=doc_ids, auto_merge = True)
            

            content = [doc.page_content for doc in docs]
            metadata = [{"doc_id": doc.metadata['source_id'], "page_no": doc.metadata['page_no'], "text": doc.page_content} for doc in docs]
            
            return content, metadata
        
        def generate_web_context(retrieval_query: str, rerank: bool = True):
            if rerank:
                docs = self.web_retriever.invoke(retrieval_query, k = 3)
                docs = self.db.rerank_docs(retrieval_query, docs, top_k= 5)
            else:    
                docs = self.web_retriever.invoke(retrieval_query, k = 2)

            content = [doc.page_content for doc in docs]
            metadata = [doc.metadata['source'] for doc in docs]
            return content, list(set(metadata))

        
        #  Generate the retrieval query
        retrieval_query = self.generate_retrieval_query(user_prompt, chat_summary)
        
        #  Generate context from the Weaviate Vector Database
        vecdb_context, vecdb_metadata = [] , []
        if doc_ids:
            vecdb_context, vecdb_metadata = generate_vecdb_context(retrieval_query)
        
        # Optionally generate context from the web (if doc_ids are provided)
        web_context , web_metadata = [] , []
        
        if enable_web_retrieval:
            web_context, web_metadata = generate_web_context(retrieval_query)
        
        # Combine the contexts
        context = vecdb_context + web_context
        metadata = {"doc_sources": vecdb_metadata, "web_sources": web_metadata}

        return context, metadata
    

    def generate_chat_summary(self, response: str, user_question: str, summary: str):
        # Define the chat summary prompt template
        CHAT_SUMMARY_PROMPT = ChatPromptTemplate.from_template(
            """You are an assistant tasked with optimizing the process of summarizing user chat .
            Given the previous chat history, user prompt, and LLM response, summarize the chat effectively.
            Focus on new information and important highlights.
            Previous Chat Summary: '''{chat}'''
            User Question: '''{user_question}'''
            LLM Response: '''{response}'''""",
        )

        # Create the chat summary chain
        chat_summary_chain = CHAT_SUMMARY_PROMPT | self.llm

        # Invoke the chat summary chain with the chat, retrieving_query, and response
        chat_summary = chat_summary_chain.invoke({"chat": summary, "user_question": user_question, "response": response})

        # Return the content of the chat summary
        return chat_summary.content

    def generate_answer(self, user_prompt: str, chat: str, context: list[str]):
        
        QA_PROMPT = ChatPromptTemplate.from_template(
            """You are an assistant tasked with answering user question.
            Given the previous chat history, user question, and relevant context, provide a clear, concise, and informative answer.
            Your response should be structured in a way that is easy to understand and directly addresses the user's question.
            Previous Chat History: '''{chat}'''
            User Question: '''{user_prompt}'''
            Relevant Context: '''{context}''' """
        )


        qa_chain = QA_PROMPT | self.llm | StrOutputParser()

        return qa_chain.astream({"chat": chat, "user_prompt": user_prompt, "context": ('\n'.join(context))})