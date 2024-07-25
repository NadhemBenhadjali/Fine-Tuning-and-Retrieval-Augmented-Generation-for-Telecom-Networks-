import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling
import datasets
from datasets import Dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

def tokenize_function(examples: datasets.arrow_dataset.Dataset):
    """
    Tokenize input.

    Args:
        examples (datasets.arrow_dataset.Dataset): Samples to tokenize
    Returns:
        tokenized_dataset (datasets.arrow_dataset.Dataset): Tokenized dataset
    """
    return tokenizer(examples['text'], max_length=512, padding='max_length', truncation=True)

def load_model_and_tokenizer(model_path: str):
    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_quant_type='nf4',
                                    bnb_4bit_compute_dtype='float16',
                                    bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def prepare_data(train_path: str, labels_path: str):
    train = pd.read_json(train_path).T
    labels = pd.read_csv(labels_path)
    train['Question_ID'] = train.index.str.split(' ').str[-1].astype('int')
    labels['Answer_letter'] = labels.Answer_ID.apply(lambda x: encode_answer(x, False))
    train = pd.merge(train, labels[['Question_ID', 'Answer_letter']], how='left', on='Question_ID')
    train['answer'] = train.Answer_letter + ')' + train.answer.str[9:]
    train = remove_release_number(train, 'question')
    return train, labels

def add_context_to_data(train: pd.DataFrame, context_path: str, use_rag: bool):
    if use_rag:
        context_all_train = pd.read_pickle(context_path)
        train['Context_1'] = context_all_train['Context_1']
        train['text'] = train.apply(lambda x: generate_prompt(x, 'Context:\n' + x['Context_1'] + '\n') + x['answer'], axis=1)
    else:
        train['text'] = train.apply(lambda x: generate_prompt(x) + x['answer'], axis=1)
    return train

def prepare_training_data(train: pd.DataFrame):
    instruction_dataset = train['text'].sample(frac=0.7, random_state=22)
    test_idx = train[~train.index.isin(instruction_dataset.index)].index
    instruction_dataset = instruction_dataset.reset_index(drop=True)
    instruction_dataset = Dataset.from_pandas(pd.DataFrame(instruction_dataset))
    tokenized_dataset = instruction_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.3, seed=22)
    return tokenized_dataset, test_idx

def configure_and_prepare_model(model, use_gradient_checkpointing: bool):
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    peft_config = LoraConfig(task_type="CAUSAL_LM",
                             r=16,
                             lora_alpha=32,
                             target_modules=['q_proj', 'k_proj', 'v_proj', 'dense'],
                             lora_dropout=0.05)
    peft_model = get_peft_model(model, peft_config)
    return peft_model

def train_model(peft_model, tokenized_dataset, output_dir: str, tokenizer):
    training_args = TrainingArguments(output_dir=output_dir,
                                      learning_rate=1e-5,
                                      per_device_train_batch_size=4,
                                      num_train_epochs=10,
                                      weight_decay=0.01,
                                      eval_strategy='epoch',
                                      logging_steps=10,
                                      fp16=True,
                                      save_strategy='no')
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(model=peft_model,
                      args=training_args,
                      train_dataset=tokenized_dataset['train'],
                      eval_dataset=tokenized_dataset['test'],
                      tokenizer=tokenizer,
                      data_collator=data_collator)
    trainer.train()
    return trainer.model

def test_inference(train: pd.DataFrame, test_idx, model_final, tokenizer, labels: pd.DataFrame):
    test_set = train.reset_index(drop=True).loc[test_idx]
    test_labels = labels.loc[test_idx]
    results_test_set, _ = llm_inference(train, model_final, tokenizer)
    results_test_set, test_set_acc = get_results_with_labels(results_test_set, test_labels)
    return results_test_set, test_set_acc

# Main script execution
MODEL_PATH = 'microsoft/phi-2'
TUNED_MODEL_PATH = '/export/livia/home/vision/Mkdayem/MkdayemyoloN/models'
USE_RAG = True

model, tokenizer = load_model_and_tokenizer(MODEL_PATH)

train, labels = prepare_data('/export/livia/home/vision/Mkdayem/MkdayemyoloN/data/TeleQnA_training.txt',
                             '/export/livia/home/vision/Mkdayem/MkdayemyoloN/data/Q_A_ID_training.csv')

train = add_context_to_data(train, '/export/livia/home/vision/Mkdayem/MkdayemyoloN/results/context_all_train3.pkl', USE_RAG)

tokenized_dataset, test_idx = prepare_training_data(train)

peft_model = configure_and_prepare_model(model, use_gradient_checkpointing=True)

model_final = train_model(peft_model, tokenized_dataset, TUNED_MODEL_PATH, tokenizer)

model_final.save_pretrained(TUNED_MODEL_PATH)

results_test_set, test_set_acc = test_inference(train, test_idx, model_final, tokenizer, labels)
