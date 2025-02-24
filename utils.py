import pandas as pd
import ast
from gptinference.utils import read_jsonl_or_json
from collections import OrderedDict
import torch
import pickle
import random
import os
import contextlib
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import dgl


SURVEY_TO_TOPIC = {
    'American_Trends_Panel_W26': "Guns",
    'American_Trends_Panel_W27': "Automation and driverless vehicles",
    'American_Trends_Panel_W29': "Views on gender",
    'American_Trends_Panel_W32': "Community types & sexual harassment",
    'American_Trends_Panel_W34': "Biomedical & food issues",
    'American_Trends_Panel_W36': "Gender & Leadership",
    'American_Trends_Panel_W41': "America in 2050",
    'American_Trends_Panel_W42': "Trust in science",
    'American_Trends_Panel_W43': "Race",
    'American_Trends_Panel_W45': "Misinformation",
    'American_Trends_Panel_W49': "Privacy & Surveilance",
    'American_Trends_Panel_W50': "Family & Relationships",
    'American_Trends_Panel_W54': "Economic inequality",
    'American_Trends_Panel_W82': "Global attitudes",
    'American_Trends_Panel_W92': "Political views"
}

SURVEY_NUM = ['26', '27', '29', '32', '34', '36', '41', '42', '43', '45', '49', '50', '54', '82', '92']
OUTPUT_MAP = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

DEMO_NAME = {
        "CREGION": "Region",
        "AGE": "Age",
        "SEX": "Gender",
        "EDUCATION": "Education",
        "CITIZEN": "Citizenship",
        "MARITAL": "Marital status",
        "RELIG": "Religion",
        "RELIGATTEND": "Frequency of religious attendance",
        "POLPARTY": "Political party",
        "INCOME": "Income",
        "POLIDEOLOGY": "Political ideology",
        "RACE": "Race",
    }


@contextlib.contextmanager
def temp_seed(seed):
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


def get_roberta_embedding(sentences, tokenizer, model):
    inputs = tokenizer(sentences, max_length=64, padding=True, return_tensors="pt")
    outputs = model(**inputs)
    
    pooled_output = outputs.pooler_output
    # last_hidden_states = outputs.last_hidden_state
    return pooled_output



def load_demographic_data(root_dir):
    path = os.path.join(root_dir, "data/human_resp/American_Trends_Panel_W26/metadata.csv")
    demo_df = pd.read_csv(path)
    demographic_keys = demo_df['key']
    demographic_options = demo_df['options']

    demographics = []
    for key, option in zip(demographic_keys, demographic_options):
        # print(key, type(option), option)
        option = ast.literal_eval(option)
        # print(key, type(option), option)
        for i, o in enumerate(option):
            if key == "CITIZEN":
                if o == "Yes":
                    demo_string = "This person is American citizen."
                else:
                    demo_string = "This person is not American citizen."
            else:
                demo_string = f"This person's {DEMO_NAME[key].lower()} is {o}."
            demographics.append(demo_string)

    # print("demographics:", demographics)
    return demographics


def load_opinion_data(root_dir, survey_num, id=0, add_imp=0):
    path = os.path.join(root_dir, f"data/retrieve_opinion_implications/opinion_implications3_W{survey_num}.json")
    opinion_implications = read_jsonl_or_json(path)
    opinion_implication_dict = OrderedDict()
    
    id_num = id
    opinions = []
    questions = []
    decl_list, imp_list = [], []
    for op_imp in opinion_implications:
        qid = op_imp['qid']
        question = op_imp['question']
        choice = op_imp['choice']
        generations = op_imp['filtered_generation']

        questions.append(question)
        if add_imp > 0:
            imp_num = min(add_imp, len(generations))
            with temp_seed(42):
                generations = random.sample(generations, imp_num)
                for gen in generations:
                    opinions.append(gen)

        opinion_implication_dict[f"{qid}|{question}|{choice}"] = {
            "question": op_imp['question'],
            "declarative_opinion": op_imp['declarative_opinion'],
            "generation": generations,
            # "id": id_num
        }
        opinions.append(op_imp['declarative_opinion'])
        decl_list.append(op_imp['declarative_opinion'])
        imp_list.append(generations)
        id_num += 1
    questions = list(set(questions))
    return opinions, opinion_implication_dict, questions, decl_list, imp_list


def save_model(model, optimizer, model_save_path):
    print(f"Saving model to this path: {model_save_path}...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        model_save_path)


def set_seed(seed):
    print("set seed for everything:", seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    dgl.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    


def load_model(model_save_path, model, optimizer):
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer

def save_pickle(filename, data):
    # Store data (serialize)
    print("Saving data as pickle...", end="")
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("end!")

def load_pickle(filename):
    # Load data (deserialize)
    print("Loading pickled data...", end="")
    with open(filename, 'rb') as handle:
        unserialized_data = pickle.load(handle)
    print("end!")
    return unserialized_data

def calibrate_data(answer):
    if answer.startswith("As more women move into management roles, it is only a matter of time before there are as many women as men in top execu"):
        answer = "As more women move into management roles, it is only a matter of time before there are as many women as men in top executives."
    
    return answer

def cluster_opinions(corpus_sentences, tokenizer, model):
    corpus_sentences = list(corpus_sentences)

    encoded_input = tokenizer(corpus_sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(
            input_ids=encoded_input['input_ids'].to(model.device),
            attention_mask=encoded_input['attention_mask'].to(model.device),
            token_type_ids=encoded_input['token_type_ids'].to(model.device)
        )
    corpus_embeddings = model_output[0][:, 0] # Perform pooling. In this case, cls pooling.
    # corpus_embeddings = torch.mean(model_output.last_hidden_state, dim=1) # mean pooling
    print("corpus_embeddings:", corpus_embeddings.shape)

    embedding_dict = dict()
    for sent, emb in zip(corpus_sentences, corpus_embeddings):
        embedding_dict[sent] = emb

    # Normalize the embeddings to unit length
    corpus_embeddings = corpus_embeddings / torch.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

    # Perform kmean clustering
    clustering_model = AgglomerativeClustering(n_clusters=None,
                                               affinity='euclidean',
                                               linkage='ward', #'ward',
                                               distance_threshold=0.45
                                               )  # , affinity='cosine', linkage='average', distance_threshold=0.4)
    clustering_model.fit(corpus_embeddings.cpu().numpy())
    cluster_assignment = clustering_model.labels_
    # print("clustering_model:", clustering_model)
    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        clustered_sentences[cluster_id].append(corpus_sentences[sentence_id])

    print("total number of clusters: {}".format(len(clustered_sentences)))

    sentence_to_cluster = dict()
    cluster_to_sentence = dict()
    cluster_to_embedding = dict()
    for i, cluster in clustered_sentences.items():
        for s in cluster:
            sentence_to_cluster[s] = i
        cluster_to_sentence[i] = cluster

        if len(cluster) == 1:
            cluster_to_embedding[i] = embedding_dict[cluster[0]].unsqueeze(0)
        else:
            embs = []
            for c in cluster:
                embs.append(embedding_dict[c])
            embs = torch.stack(embs, dim=0)
            embs = torch.mean(embs, dim=0, keepdim=True)
            cluster_to_embedding[i] = embs

    return sentence_to_cluster, cluster_to_sentence, cluster_to_embedding