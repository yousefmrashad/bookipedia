from collections import Counter
import weaviate.classes as wvc
from weaviate.collections.classes.internal import QueryReturn, ReturnProperties
import torch
from weaviate import WeaviateClient
from langchain_core.documents import Document
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def rerank_docs(
        query: str ,
        docs: list[Document],
        top_k :int) -> list[Document]:
    """Re-ranks a list of documents based on their relevance to a given query using a pre-trained sequence classification model.

    Args:
        - query (str): The query string used to determine the relevance of the documents.
        - docs (list[Document]): A list of Document objects to be re-ranked. Each Document object should have a `page_content` attribute containing the text of the document.
        - top_k (int): The number of top-ranked documents to return.


    Returns:
        - list[Document]: A list of the top-ranked Document objects, sorted by relevance to the query.
    """
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-large')
    model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-reranker-large')
    model.eval()
    # Prepare the query-document pairs for the model
    pairs = [[query , doc.page_content] for doc in docs]
    # Tokenize the pairs and generate scores
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        # Sort the documents based on the scores and return the top-k documents
        sorted_indices = torch.argsort(scores, descending=True)
        sorted_docs = [docs[i] for i in sorted_indices]
    return sorted_docs[: top_k]

def merger(client:WeaviateClient, response: QueryReturn[ReturnProperties, None]):
    cls = client.collections.get("am_chunks")

    child_counter = Counter(obj.properties['child'] for obj in response.objects)
    parent_counter = Counter(obj.properties['parent'] for obj in response.objects)

    child_merge = [key for key, value in child_counter.items() if value >= 3]
    parent_merge = [key for key, value in parent_counter.items() if value >= 9]
    leaf = [key for key, value in child_counter.items() if value < 3]

    child_merge = [c for c in child_merge if all(c not in range(p*4, (p+1)*4) for p in parent_merge)]

    leaves = [obj for obj in response.objects if obj.properties['child'] in leaf]

            
    parents = None
    if parent_merge:
        parents = cls.query.fetch_objects(filters=wvc.query.Filter.by_property("parent").contains_any(parent_merge))

    children = None
    if child_merge:
        children = cls.query.fetch_objects(filters=wvc.query.Filter.by_property("child").contains_any(child_merge))

    return parents, children, leaves

def parent_to_dict(response: QueryReturn[ReturnProperties, None]) -> dict:
    parent_dict = {}
    for o in response.objects:
        key = o.properties['parent']
        if key in parent_dict:
            parent_dict[key]['text'] += (' ' + o.properties['text'])
        else:
            mets = {key: value for key, value in o.properties.items() if key == 'source_id' or key == 'page_no'}
            parent_dict[key] = {'text': o.properties['text'], 'metadata': mets} 
    return parent_dict

def merger_to_docs(client:WeaviateClient, response: QueryReturn[ReturnProperties, None]) -> list[Document]:
    parents, children, leaves = merger(client, response)
    docs = []
    
    if parents is not None:
        parent_dict = parent_to_dict(parents)
        for v in parent_dict.values():
            docs.append(Document(page_content=v['text'], metadata=v['metadata']))
    if children is not None:
        children_dict = parent_to_dict(children)
        for v in children_dict.values():
            docs.append(Document(page_content=v['text'], metadata=v['metadata']))

    for o in leaves:
        text = o.properties['text']
        mets = {key: value for key, value in o.properties.items() if key == 'source_id' or key == 'page_no'}
        docs.append(Document(page_content=text, metadata=mets))

    return docs