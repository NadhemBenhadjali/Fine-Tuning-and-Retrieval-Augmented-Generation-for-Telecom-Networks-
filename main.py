import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb


def load_model(model_path: str):
    """
    Load the model and tokenizer from the specified path.

    Args:
        model_path (str): Path to the model.

    Returns:
        model, tokenizer: Loaded model and tokenizer.
    """
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype="auto",
                                                 trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2',
                                              trust_remote_code=True)
    return model, tokenizer


def load_data(train_path: str, labels_path: str, test_path: str, test_new_path: str):
    """
    Load the training and test datasets.

    Args:
        train_path (str): Path to the training data.
        labels_path (str): Path to the labels data.
        test_path (str): Path to the test data.
        test_new_path (str): Path to additional test data.

    Returns:
        train, labels, test: Loaded datasets.
    """
    train = pd.read_json(train_path).T
    labels = pd.read_csv(labels_path)
    test = pd.read_json(test_path).T
    test_new = pd.read_json(test_new_path).T
    test = pd.concat([test, test_new])
    return train, labels, test


def prepare_data(train: pd.DataFrame, test: pd.DataFrame):
    """
    Prepare the data for processing.

    Args:
        train (pd.DataFrame): Training data.
        test (pd.DataFrame): Test data.

    Returns:
        train, test: Prepared datasets.
    """
    train['Question_ID'] = train.index.str.split(' ').str[-1]
    test['Question_ID'] = test.index.str.split(' ').str[-1]
    train = remove_release_number(train, 'question')
    test = remove_release_number(test, 'question')
    return train, test


def configure_rag(vector_path: str):
    """
    Configure RAG settings and load vectorized documents.

    Args:
        vector_path (str): Path to the vector storage.

    Returns:
        query_engine, top_k: Configured query engine and number of chunks to retrieve.
    """
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    Settings.llm = None
    Settings.chunk_size = 128
    Settings.chunk_overlap = 20

    db = chromadb.PersistentClient(path=vector_path)
    chroma_collection = db.get_or_create_collection("rel18")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

    top_k = 3
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    query_engine = RetrieverQueryEngine(retriever=retriever,
                                        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.5)])
    return query_engine, top_k


def perform_inference(data: pd.DataFrame, model, tokenizer, perform_rag: bool, query_engine=None, top_k=None):
    """
    Perform inference on the given data.

    Args:
        data (pd.DataFrame): Data for inference.
        model: Model to use for inference.
        tokenizer: Tokenizer to use.
        perform_rag (bool): Whether to perform RAG inference.
        query_engine: Query engine for RAG inference.
        top_k: Number of chunks to retrieve for RAG inference.

    Returns:
        results: Inference results.
    """
    if perform_rag:
        results, _ = llm_inference(data, model, tokenizer, perform_rag, query_engine, top_k)
    else:
        results, _ = llm_inference(data, model, tokenizer)
    return results


def save_results(results: pd.DataFrame, file_path: str):
    """
    Save the inference results to a CSV file.

    Args:
        results (pd.DataFrame): Inference results.
        file_path (str): Path to save the results.
    """
    results.to_csv(file_path, index=False)


def main():
    MODEL_USED = 'Phi-2'
    USE_REPO_MODEL = False
    USE_LOCAL_FINE_TUNED = True
    USE_MODEL_FROM_HUGGINGFACE = False
    DO_TRAIN_INFERENCE = True
    PERFORM_RAG = True

    if USE_REPO_MODEL:
        model_path = 'models/peft_phi_2_repo'
    elif USE_LOCAL_FINE_TUNED:
        model_path = '/export/livia/home/vision/Mkdayem/MkdayemyoloN/models'
    else:
        model_path = 'microsoft/phi-2'

    model, tokenizer = load_model(model_path)

    train, labels, test = load_data('/export/livia/home/vision/Mkdayem/MkdayemyoloN/data/TeleQnA_training.txt',
                                    '/export/livia/home/vision/Mkdayem/MkdayemyoloN/data/Q_A_ID_training.csv',
                                    '/export/livia/home/vision/Mkdayem/MkdayemyoloN/data/TeleQnA_testing1.txt',
                                    '/export/livia/home/vision/Mkdayem/MkdayemyoloN/data/questions_new.txt')

    train, test = prepare_data(train, test)

    create_empty_directory('results')
    today_date = pd.to_datetime('today').strftime('%Y_%m_%d')

    if PERFORM_RAG:
        vector_path = '/export/livia/home/vision/Mkdayem/MkdayemyoloN/rag3'
        query_engine, top_k = configure_rag(vector_path)
        if DO_TRAIN_INFERENCE:
            results_train = perform_inference(train, model, tokenizer, PERFORM_RAG, query_engine, top_k)
    else:
        query_engine = None
        top_k = None
        if DO_TRAIN_INFERENCE:
            results_train = perform_inference(train, model, tokenizer, PERFORM_RAG)

    if DO_TRAIN_INFERENCE:
        results_labels, train_acc = get_results_with_labels(results_train, labels)
        save_results(results_labels, f'results/{today_date}_{MODEL_USED}_train_results.csv')

    results_test = perform_inference(test, model, tokenizer, PERFORM_RAG, query_engine, top_k)
    results_test = results_test.astype('int')
    results_test['Task'] = MODEL_USED
    save_results(results_test, f'{today_date}_{MODEL_USED}_test_results.csv')


if __name__ == "__main__":
    main()
