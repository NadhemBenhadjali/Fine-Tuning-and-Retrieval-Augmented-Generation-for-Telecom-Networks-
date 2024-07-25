from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
import chromadb
import pandas as pd
import re
from tqdm import tqdm
import time
import os
import torch

def rag_inference_on_train(data: pd.DataFrame, engine: RetrieverQueryEngine, batch_size=10):
    create_empty_directory('results')
    context_all_train = []

    start_time = time.time()

    for idx, row in tqdm(data[['question', 'answer']].reset_index().iterrows()):
        question = row['question']
        answer = row['answer']
        response = engine.query(question)
        try:
            response_1 = response.source_nodes[0].text
        except:
            response_1 = ''
        response_1 = re.sub('\s+', ' ', response_1)
        context_all_train.append([question, response_1, answer])

        if idx % batch_size == 0:
            elapsed_time = time.time() - start_time
            estimated_total_time = (elapsed_time / (idx + 1)) * len(data)
            remaining_time = estimated_total_time - elapsed_time

            print(question)
            print(f'\n{response_1}')
            print(f'\nAnswer:\n{answer}')
            print(f'Processed {idx+1}/{len(data)} questions.')
            print(f'Elapsed time: {elapsed_time:.2f} seconds.')
            print(f'Estimated total time: {estimated_total_time:.2f} seconds.')
            print(f'Estimated remaining time: {remaining_time:.2f} seconds.\n')

    context_all_train_df = pd.DataFrame(context_all_train, columns=['Question', 'Context_1', 'Answer'])
    context_all_train_df.to_csv('results/context_all_train2.csv', index=False)
    context_all_train_df.to_pickle('results/context_all_train2.pkl')

def create_empty_directory(directory):
    if os.path.exists(directory):
        for file in os.scandir(directory):
            os.remove(file.path)
    else:
        os.makedirs(directory)

# Main Code
DOCS_PATH = '/export/livia/home/vision/Mkdayem/MkdayemyoloN/data/rel18'
VECTOR_PATH = '/export/livia/home/vision/Mkdayem/MkdayemyoloN/rag2'
SAMPLE_DOCS = False
RAG_INFERENCE = True

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Ensure PyTorch uses GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the embedding model with GPU support
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5", device=device)
Settings.llm = None
Settings.chunk_size = 128
Settings.chunk_overlap = 20

if SAMPLE_DOCS:
    SAMPLED_DOCS_PATH = '/export/livia/home/vision/Mkdayem/MkdayemyoloN/rag2'
    SAMPLE_FRAC = 0.5
    create_dir_with_sampled_docs(DOCS_PATH, SAMPLED_DOCS_PATH, SAMPLE_FRAC)
    documents = SimpleDirectoryReader(SAMPLED_DOCS_PATH).load_data()
else:
    documents = SimpleDirectoryReader(DOCS_PATH).load_data()

db = chromadb.PersistentClient(path=VECTOR_PATH)
chroma_collection = db.get_or_create_collection("rel18")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
if RAG_INFERENCE:
    top_k = 3  # Increased to 3 to get more diverse context
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)])

    train = pd.read_json('/export/livia/home/vision/Mkdayem/MkdayemyoloN/data/TeleQnA_training.txt').T
    train['Question_ID'] = train.index.str.split(' ').str[-1]
    train = remove_release_number(train, 'question')

    rag_inference_on_train(train, query_engine, batch_size=10)

