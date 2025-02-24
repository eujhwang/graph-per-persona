# %%
import glob

import torch
import logging
import argparse
import datetime, time
from pathlib import Path
from gptinference.utils import read_jsonl_or_json
from transformers import set_seed, AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
import json, os
import random
import numpy as np
from transformers import T5Tokenizer, T5ForSequenceClassification
import tqdm
import dgl
# from dgl.sampling import sample_neighbors
import wandb

from opinion_dataset import OpinionDataset
from torch.utils.data import DataLoader
from opinion_graph_walker_model import AttnIO # ModalityAttentionLayer, KGPathWalker,
from utils import *
from train import train


# from graph_encoder import GraphEncoder
label_names = ["entailment", "neutral", "contradiction"]

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

# llm-graph
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../"
print("ROOT_DIR: {}".format(ROOT_DIR))

def set_logger(args):
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H')
    timestamp_yymmdd = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d')
    
    log_dir = f"./log/opinion_graph_walker/{timestamp_yymmdd}/" #{dir_name}/"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    if args.survey != -1:
        logging_path = log_dir + f"output_{timestamp}_topic_W{SURVEY_NUM[args.survey]}_ques{args.num_ques}_op{args.num_op}_{args.infer_type}_path{args.path_len}.log"
    else:
        logging_path = log_dir + f"output_{timestamp}_topic_all_ques{args.num_ques}_op{args.num_op}_{args.infer_type}_path{args.path_len}.log"

    logging.basicConfig(
        level=logging.WARN,  # logging.WARN
        format="%(asctime)s %(message)s",  # "%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(logging_path),
            logging.StreamHandler()
        ]
    )



def preprocess_input(user_responses, survey_name):
    survey_to_context = []
    user_id_list = []
    for i, user_response in enumerate(user_responses):
        user_id = user_response['user_id']
        survey = user_response['survey']

        if survey != survey_name:
            continue

        # if survey not in survey_to_context:
        #     survey_to_context[survey] = []
        if user_id not in user_id_list:
            user_id_list.append(user_id)

        explicit_persona = user_response['explicit_persona']
        implicit_persona = user_response['implicit_persona']
        implicit_questions = user_response['implicit_questions']
        
        # _implicit_questions = []
        num_ques = min(30, len(implicit_questions))
        for j, imp_que in enumerate(implicit_questions[:num_ques]):
            qid = imp_que['qid']
            question = imp_que['question']
            choices = user_responses[i]['implicit_questions'][j]['choices']
            answer = imp_que['answer']
            answer = calibrate_data(answer)

            if answer not in choices:
                continue

            # _implicit_questions.append({
            #     "qid": qid,
            #     "question": question,
            #     "choices": choices,
            #     "answer": answer
            # })
            
            survey_to_context.append({
                "user_id": user_id,
                "qid": qid,
                "explicit_persona": explicit_persona,
                "implicit_persona": implicit_persona,
                # "implicit_questions": _implicit_questions,
                "question": question,
                "choices": choices,
                "answer": answer
            })
    print("survey_to_context:",len(survey_to_context), "user_id_list:", len(user_id_list))

    train_user_id = user_id_list[45:100]
    val_user_id = user_id_list[35:45]
    test_user_id = user_id_list[0:35]
    print("len_train: {}, len_val: {}, len_test: {}".format(len(train_user_id), len(val_user_id), len(test_user_id)))

    assert len(set(train_user_id).intersection(set(val_user_id))) == 0
    assert len(set(test_user_id).intersection(set(val_user_id))) == 0
    assert len(set(test_user_id).intersection(set(train_user_id))) == 0

    train_survey_to_context, val_survey_to_context, test_survey_to_context = [], [], []
    for item in survey_to_context:
        if item['user_id'] in train_user_id:
            train_survey_to_context.append(item)
        elif item['user_id'] in val_user_id:
            val_survey_to_context.append(item)
        elif  item['user_id'] in test_user_id:
            test_survey_to_context.append(item)

    return train_survey_to_context, val_survey_to_context, test_survey_to_context


def collate(batch):
    graphs = [item[0] for item in batch]
    seed_entities = [item[1] for item in batch]
    demographic_embeddings = [item[2] for item in batch]
    opinion_embeddings = [item[3] for item in batch]

    indexed_candidate_nodes = [item[4] for item in batch]
    indexed_target_node = [item[5] for item in batch]
    triples = [item[6] for item in batch]
    nodeId2node = [item[7] for item in batch]
    questions = [item[8] for item in batch]
    edge_id_to_ht = [item[9] for item in batch]
    ht_to_edge_id = [item[10] for item in batch]
    
    return graphs, seed_entities, demographic_embeddings, opinion_embeddings, indexed_candidate_nodes, indexed_target_node, triples, nodeId2node, questions, edge_id_to_ht, ht_to_edge_id


# %%
def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.use_wandb:
        wandb.init(project="opinion-graph", entity="ejhwang-ubc")
        if wandb.run.sweep_id is None:
            run_id = "none"
            sweep_id = "none"
        else:
            run_id = wandb.run.id
            sweep_id = wandb.run.sweep_id
        
        wandb.config.update(args)
        print("wandb.config: %s" % (wandb.config))
    else:
        sweep_id = None
        run_id = None
    # batch_size = 8
    # # lr = 5e-4
    # lr = 0.00005
    # path_len = 3
    batch_size = args.batch_size
    lr = args.learning_rate
    path_len = args.path_len
    epochs = args.epoch
    path_topk = args.path_topk

    file = "../data/sampled_user_responses_20_final.json"
    user_responses = read_jsonl_or_json(file)
    
    if args.survey > 14:
        raise Exception(f"survey number: {args.survey} is not supported!")

    demographic_list = load_demographic_data(ROOT_DIR)
    if args.survey != -1:
        opinion_list, opinion_implication_dict, question_list, declarative_list, implication_list = load_opinion_data(ROOT_DIR, SURVEY_NUM[args.survey], id=0, add_imp=args.add_imp)
    else:
        opinion_implication_dict = OrderedDict()
        opinion_list = []
        for survey_num in SURVEY_NUM:
            _opinion_list, _opinion_implication_dict, question_list, declarative_list, implication_list = load_opinion_data(ROOT_DIR, survey_num, id=len(opinion_list), add_imp=args.add_imp)
            opinion_list.extend(_opinion_list)
            opinion_implication_dict.update(_opinion_implication_dict)
    
    print("opinion_list: {}, opinion_implication_dict: {}, question_list: {}, declarative_list: {}, implication_list: {}".format(len(opinion_list), len(opinion_implication_dict), len(question_list), len(declarative_list), len(implication_list)))
    if args.add_imp == 0:
        assert len(opinion_list) == len(opinion_implication_dict)

    # e.g. roberta, deberta, albert, ...BAAI/bge-base-en-v1.5
    sent_tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    sent_model = AutoModel.from_pretrained(args.pretrained_model)
    sent_model = sent_model.to(device)

    if args.use_cluster:
        sentence_to_cluster, cluster_to_sentence, cluster_to_embedding = cluster_opinions(opinion_list, sent_tokenizer, sent_model)
        cluster_embeddings = []
        for i in range(len(cluster_to_embedding)):
            cluster_embeddings.append(cluster_to_embedding[i])
        cluster_embeddings = torch.cat(cluster_embeddings, dim=0)
    else:
        sentence_to_cluster, cluster_to_sentence, cluster_to_embedding = None, None, None

    if args.survey != -1:
        survey_name = f'American_Trends_Panel_W{SURVEY_NUM[args.survey]}'
        train_survey_to_context, val_survey_to_context, test_survey_to_context = preprocess_input(user_responses=user_responses, survey_name=survey_name)
    else:
        train_survey_to_context, val_survey_to_context, test_survey_to_context = [], [], []
        for survey_num in SURVEY_NUM:
            survey_name = f'American_Trends_Panel_W{survey_num}'
            _train_survey_to_context, _val_survey_to_context, _test_survey_to_context = preprocess_input(user_responses=user_responses, survey_name=survey_name)
            train_survey_to_context.extend(_train_survey_to_context)
            val_survey_to_context.extend(_val_survey_to_context)
            test_survey_to_context.extend(_test_survey_to_context)
        survey_name = "all" # later used as dir name
    print("train_survey_to_context: {}, val_survey_to_context: {}, test_survey_to_context: {}".format(len(train_survey_to_context), len(val_survey_to_context), len(test_survey_to_context)))

    if args.use_cluster:
        candidate_responses = []
        for item in train_survey_to_context:
            qid = item['qid']
            question = item['question']
            choices = item['choices']
            for choice in choices:
                choice = calibrate_data(choice)
                decl_opinion = opinion_implication_dict[f"{qid}|{question}|{choice}"]["declarative_opinion"]
                candidate_responses.append(decl_opinion)
        for item in val_survey_to_context:
            qid = item['qid']
            question = item['question']
            choices = item['choices']
            for choice in choices:
                choice = calibrate_data(choice)
                decl_opinion = opinion_implication_dict[f"{qid}|{question}|{choice}"]["declarative_opinion"]
                candidate_responses.append(decl_opinion)
        for item in test_survey_to_context:
            qid = item['qid']
            question = item['question']
            choices = item['choices']
            for choice in choices:
                choice = calibrate_data(choice)
                decl_opinion = opinion_implication_dict[f"{qid}|{question}|{choice}"]["declarative_opinion"]
                candidate_responses.append(decl_opinion)
        candidate_responses = list(set(candidate_responses))

        encoded_input = sent_tokenizer(candidate_responses, padding=True, truncation=True, return_tensors='pt')
        # Compute token embeddings
        with torch.no_grad():
            model_output = sent_model(
                input_ids=encoded_input['input_ids'].to(sent_model.device),
                attention_mask=encoded_input['attention_mask'].to(sent_model.device),
                token_type_ids=encoded_input['token_type_ids'].to(sent_model.device)
            )
        response_embeddings = model_output[0][:, 0] # Perform pooling. In this case, cls pooling.
        print("response_embeddings:", response_embeddings.shape)
        cluster_embeddings = torch.cat([cluster_embeddings, response_embeddings], dim=0)
        print("cluster_embeddings:", cluster_embeddings.shape)
        response_to_id = dict()
        # response_to_emb = dict()
        for i, resp in enumerate(candidate_responses):
            response_to_id[resp] = i
        print("len(candidate_responses):", len(candidate_responses), len(set(candidate_responses)))
    else:
        response_to_id = None
        cluster_embeddings = None

    if args.conn_type.startswith("t5"):
        assert args.survey != -1
        t5_model = args.conn_type
        model_save_path = os.path.join(ROOT_DIR, f"{t5_model}_finetuned_model/filtered/American_Trends_Panel_W{SURVEY_NUM[args.survey]}/")

        entail_tokenizer = T5Tokenizer.from_pretrained(t5_model)
        entail_model = T5ForSequenceClassification.from_pretrained(model_save_path).to(device)
        entail_model.eval()
    elif args.conn_type.startswith("google"):
        assert args.survey != -1
        entail_tokenizer = None
        entail_model = None
        # model_save_path = args.conn_type
        # entail_tokenizer = AutoTokenizer.from_pretrained(model_save_path)
        # if model_save_path.endswith("flan-t5-xxl"):
        #     entail_model = AutoModelForSeq2SeqLM.from_pretrained(model_save_path, device_map=device, torch_dtype=torch.bfloat16).to(device)
        # else:
        #     entail_model = AutoModelForSeq2SeqLM.from_pretrained(model_save_path).to(device)
        # entail_model.eval()
    else:
        entail_tokenizer = None
        entail_model = None

    train_ds = OpinionDataset(
        root_dir=ROOT_DIR,
        user_responses=train_survey_to_context,
        demographics=demographic_list,
        opinions=opinion_list,
        questions=question_list,
        declaratives=declarative_list,
        implications=implication_list,
        opinion_implication_dict=opinion_implication_dict,
        sentence_to_cluster=sentence_to_cluster,
        cluster_to_sentence=cluster_to_sentence,
        response_to_id=response_to_id,
        use_cluster=args.use_cluster,
        conn_type=args.conn_type,
        entail_tokenizer=entail_tokenizer,
        entail_model=entail_model,
        ds_type="train", # for saving t5 entailment triples
        survey_name=survey_name, # for saving t5 entailment triples
        add_imp=args.add_imp,
        entail_threshold=args.threshold,
        seed=args.seed,
    )
    val_ds = OpinionDataset(
        root_dir=ROOT_DIR,
        user_responses=val_survey_to_context,
        demographics=demographic_list,
        opinions=opinion_list,
        questions=question_list,
        declaratives=declarative_list,
        implications=implication_list,
        opinion_implication_dict=opinion_implication_dict,
        sentence_to_cluster=sentence_to_cluster,
        cluster_to_sentence=cluster_to_sentence,
        response_to_id=response_to_id,
        use_cluster=args.use_cluster,
        conn_type=args.conn_type,
        entail_tokenizer=entail_tokenizer,
        entail_model=entail_model,
        ds_type="val", # for saving t5 entailment triples
        survey_name=survey_name, # for saving t5 entailment triples
        add_imp=args.add_imp,
        entail_threshold=args.threshold,
        seed=args.seed,
    )
    test_ds = OpinionDataset(
        root_dir=ROOT_DIR,
        user_responses=test_survey_to_context,
        demographics=demographic_list,
        opinions=opinion_list,
        questions=question_list,
        declaratives=declarative_list,
        implications=implication_list,
        opinion_implication_dict=opinion_implication_dict,
        sentence_to_cluster=sentence_to_cluster,
        cluster_to_sentence=cluster_to_sentence,
        response_to_id=response_to_id,
        use_cluster=args.use_cluster,
        conn_type=args.conn_type,
        entail_tokenizer=entail_tokenizer,
        entail_model=entail_model,
        ds_type="test", # for saving t5 entailment triples
        survey_name=survey_name, # for saving t5 entailment triples
        add_imp=args.add_imp,
        entail_threshold=args.threshold,
        seed=args.seed,
    )

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate)

    model = AttnIO(
        init_dim=args.init_dim,
        opinions=opinion_list, 
        demographics=demographic_list,
        questions=question_list,
        declaratives=declarative_list,
        implications=implication_list,
        sentence_to_cluster=sentence_to_cluster,
        cluster_to_sentence=cluster_to_sentence,
        response_to_id=response_to_id,
        cluster_embeddings=cluster_embeddings,
        use_cluster=args.use_cluster,
        cluster_type=args.cluster_type,
        pretrained_model=args.pretrained_model,
        path_len=path_len,
        sent_tokenizer=sent_tokenizer,
        sent_model=sent_model,
        no_demo=args.no_demo,
        num_head=args.num_head,
    )
    model = model.to(device)
    optimizer = torch.optim.Adam( filter(lambda p: p.requires_grad, model.parameters()), lr, weight_decay=args.weight_decay)
    train_steps = int(len(train_dataloader) * epochs)
    print("train_steps:", train_steps, "warmup_steps:", int(train_steps * 0.1))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(train_steps * 0.1),
        num_training_steps=train_steps
    )

    if args.use_wandb:
        wandb.watch(model)

    if sweep_id:
        model_save_dir = os.path.join(ROOT_DIR, f"model-output/{survey_name}/{sweep_id}/")
    else:
        model_save_dir = os.path.join(ROOT_DIR, f"model-output/{survey_name}/")
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)

    if args.saved_model != "":
        """
        example command:
        python main.py --add_imp=5 --batch_size=8 --conn_type=t5-base --epoch=30 --learning_rate=0.0001 --loss_type=full --num_head=3 --num_op=16 --num_ques=30 --path_len=5 --pretrained_model=BAAI/bge-base-en-v1.5 --seed=42 --survey=0 --threshold=0.1 --weight_decay=0 --no_demo --saved_model=/scratch/ssd004/scratch/ejhwang/llm-graph/model-output/American_Trends_Panel_W26/none/test_model_none.pth
        """
        model_save_path = args.saved_model
        if not os.path.exists(model_save_path):
            raise Exception(f"model doesn't exist! -- {model_save_path}")

        print(f"Loading saved model: {model_save_path}...")

        # Loading
        model, optimizer = load_model(model_save_path, model, optimizer)
        
        val_accuracy, _, _, _, _, _ = evaluate(val_dataloader, model, path_len, path_topk)
        test_accuracy, test_path_indices, test_path_list, test_path_score_list, test_path_score_sum_list, target_indices = evaluate(test_dataloader, model, path_len, path_topk)
        metrics = {
            "[Val] Accuracy": val_accuracy,
            "[Test] Accuracy": test_accuracy,
        }
        print(metrics)
        
        # print("len(test):", len(test_path_indices), len(test_path_list), len(test_path_score_list), len(test_path_score_sum_list), len(target_indices))
        # print("len(test_survey_to_context):", len(test_survey_to_context))
        # print(test_survey_to_context[0].keys())
        # assert False
        
        final_incorrect_items, final_correct_items = [], []
        final_survey_items = []
        for final_idx, paths, path_scores, path_sum_scores, target_idx, survey_item in zip(test_path_indices, test_path_list, test_path_score_list, test_path_score_sum_list, target_indices, test_survey_to_context):
            if final_idx == -1:
                print("no path exists for target_idx: {} ({})".format(target_idx, opinion_list[target_idx]))
                continue
            
            path_info = [(p, ps, pss) for p, ps, pss in zip(paths, path_scores, path_sum_scores)]
            path_info = sorted(path_info, key=lambda x: x[2], reverse=True)
            
            _opinions, _opinions_scores = [], []
            for i, (path, path_score, path_sum_score) in enumerate(path_info[:5]):
                _opinions.append([(opinion_list[p], ps) for p, ps in zip(path, path_score)])
                _opinions_scores.append((_opinions[i], path_sum_score))

            if final_idx == target_idx:
                final_correct_items.append({
                    "selected_idx": (final_idx, opinion_list[final_idx]),
                    "target_idx": (target_idx, opinion_list[target_idx]),
                    "opinions": _opinions_scores,
                })
            else:
                final_incorrect_items.append({
                    "selected_idx": (final_idx, opinion_list[final_idx]),
                    "target_idx": (target_idx, opinion_list[target_idx]),
                    "opinions": _opinions_scores,
                })
            survey_item.update({
                "selected_idx": (final_idx, opinion_list[final_idx]),
                "target_idx": (target_idx, opinion_list[target_idx]),
                "opinions": _opinions_scores,
            })
            final_survey_items.append(survey_item)

        analysis_out = model_save_path.replace(".pth", "_test_out_all.json")
        analysis_incorrect_out = model_save_path.replace(".pth", "_test_out_incorrect.json")
        analysis_correct_out = model_save_path.replace(".pth", "_test_out_correct.json")
        with open(analysis_out, 'w') as txtfile:
            json.dump(final_survey_items, txtfile, indent=4)
        
        with open(analysis_incorrect_out, 'w') as txtfile:
            json.dump(final_incorrect_items, txtfile, indent=4)

        with open(analysis_correct_out, 'w') as txtfile:
            json.dump(final_correct_items, txtfile, indent=4)

    else:
        if run_id:
            model_name = f"model_{run_id}.pth"
        else:
            model_name = f"model.pth"
        model_save_path = os.path.join(model_save_dir, model_name)
    print("model_save_path: {}".format(model_save_path))

    if args.do_train:
        model.zero_grad()

        best_val_accuracy, best_test_accuracy = 0, 0
        best_model = None
        best_epoch = -1
        for epoch in range(epochs):
            model.train()
            losses = []
            for batch in tqdm.tqdm(train_dataloader, total=len(train_dataloader)):
                loss = train(batch, model, optimizer, path_len, args.loss_type, scheduler)
                losses.append(loss)
            
            avg_loss = sum(losses) / len(losses)
            if args.use_wandb:
                wandb.log({
                    "Epoch": epoch,
                    "[Train] Loss": avg_loss
                }, commit=False)

            train_accuracy, _, _, _, _, _ = evaluate(train_dataloader, model, path_len, path_topk)            
            val_accuracy, _, _, _, _, _ = evaluate(val_dataloader, model, path_len, path_topk)
            test_accuracy, _, _, _, _, _ = evaluate(test_dataloader, model, path_len, path_topk)

            # train_accuracy2, _, _, _, _, _ = evaluate_edge(train_dataloader, model, path_len, path_topk)
            # val_accuracy2, _, _, _, _, _ = evaluate_edge(val_dataloader, model, path_len, path_topk)
            # test_accuracy2, _, _, _, _, _ = evaluate_edge(test_dataloader, model, path_len, path_topk)

            metrics = {
                "Epoch": epoch,
                "[Train] Loss": avg_loss,
                "[Train] Accuracy": train_accuracy,
                "[Val] Accuracy": val_accuracy,
                "[Test] Accuracy": test_accuracy,
                # "[Train-v2] Accuracy": train_accuracy2,
                # "[Val-v2] Accuracy": val_accuracy2,
                # "[Test-v2] Accuracy": test_accuracy2,
            }
            print(metrics)
            if args.use_wandb:
                wandb.log({
                    "[Train] Accuracy": train_accuracy,
                    "[Val] Accuracy": val_accuracy,
                    "[Test] Accuracy": test_accuracy,
                }, commit=False)

            if best_val_accuracy < val_accuracy:
                best_val_accuracy = val_accuracy
                best_test_accuracy = test_accuracy
                best_model = model
                best_epoch = epoch
                metrics = {
                    "[Train] Best Accuracy": train_accuracy,
                    "[Val] Best Accuracy": best_val_accuracy,
                    "[Test] Best Accuracy": best_test_accuracy,
                    "Best Epoch": best_epoch,
                }
                print(metrics)
                if args.use_wandb:
                    wandb.log(metrics, commit=False)
            
                save_model(best_model, optimizer, model_save_path)
            if (epoch - best_epoch) >= args.early_stopping:
                print("Validation Accuracy haven't changed for a while! Stopping the loop -- current epoch: {}, best epoch: {}".format(epoch, best_epoch))
                break
            if args.use_wandb:
                wandb.log({}, commit=True)
        
        if best_model is not None:
            model, optimizer = load_model(model_save_path, model, optimizer)
            test_accuracy, _, _, _, _, _ = evaluate(test_dataloader, model, path_len, path_topk)
            metrics = {"[Test] Final Accuracy": test_accuracy}
            if args.use_wandb:
                wandb.log(metrics)
            print(metrics)

    if args.do_eval:
        model, optimizer = load_model(model_save_path, model, optimizer)
        train_accuracy, _, _, _, _, _ = evaluate(train_dataloader, model, path_len, path_topk)
        val_accuracy, _, _, _, _, _ = evaluate(val_dataloader, model, path_len, path_topk)
        test_accuracy, test_path_indices, test_path_list, test_path_score_list, test_path_score_sum_list, target_indices = evaluate(test_dataloader, model, path_len, path_topk)
        metrics = {
            "[Train] Accuracy": train_accuracy,
            "[Val] Accuracy": val_accuracy,
            "[Test] Accuracy": test_accuracy,
        }
        print(metrics)

        final_incorrect_items, final_correct_items = [], []
        final_survey_items = []
        for final_idx, paths, path_scores, path_sum_scores, target_idx, survey_item in zip(test_path_indices, test_path_list, test_path_score_list, test_path_score_sum_list, target_indices, test_survey_to_context):
            if final_idx == -1:
                print("no path exists for target_idx: {} ({})".format(target_idx, opinion_list[target_idx]))
                continue
            
            path_info = [(p, ps, pss) for p, ps, pss in zip(paths, path_scores, path_sum_scores)]
            path_info = sorted(path_info, key=lambda x: x[2], reverse=True)
            
            _opinions, _opinions_scores = [], []
            for i, (path, path_score, path_sum_score) in enumerate(path_info[:5]):
                _opinions.append([(opinion_list[p], ps) for p, ps in zip(path, path_score)])
                _opinions_scores.append((_opinions[i], path_sum_score))

            if final_idx == target_idx:
                final_correct_items.append({
                    "selected_idx": (final_idx, opinion_list[final_idx]),
                    "target_idx": (target_idx, opinion_list[target_idx]),
                    "opinions": _opinions_scores,
                })
            else:
                final_incorrect_items.append({
                    "selected_idx": (final_idx, opinion_list[final_idx]),
                    "target_idx": (target_idx, opinion_list[target_idx]),
                    "opinions": _opinions_scores,
                })
            survey_item.update({
                "selected_idx": (final_idx, opinion_list[final_idx]),
                "target_idx": (target_idx, opinion_list[target_idx]),
                "opinions": _opinions_scores,
            })
            final_survey_items.append(survey_item)

        analysis_out = model_save_path.replace(".pth", "_test_out_all.json")
        analysis_incorrect_out = model_save_path.replace(".pth", "_test_out_incorrect.json")
        analysis_correct_out = model_save_path.replace(".pth", "_test_out_correct.json")
        with open(analysis_out, 'w') as txtfile:
            json.dump(final_survey_items, txtfile, indent=4)
        
        with open(analysis_incorrect_out, 'w') as txtfile:
            json.dump(final_incorrect_items, txtfile, indent=4)

        with open(analysis_correct_out, 'w') as txtfile:
            json.dump(final_correct_items, txtfile, indent=4)


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


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--survey", default=0, type=int, help="survey number; -1: all")
    parser.add_argument("--num_op", default=16, type=int, help="number of opinions")
    parser.add_argument("--num_ques", default=30, type=int, help="number of questions")
    parser.add_argument("--path_len", default=3, type=int, help="max path length")
    parser.add_argument("--batch_size", default=8, type=int, help="batch size")
    parser.add_argument("--init_dim", default=256, type=int, help="init_dim")
    parser.add_argument("--path_topk", default=5, type=int, help="path topk")

    parser.add_argument("--learning_rate", default=0.0001, type=float, help="learning rate")
    parser.add_argument("--epoch", default=30, type=int, help="number of epochs")
    parser.add_argument("--early_stopping", default=20, type=int, help="number of epochs")
    parser.add_argument("--use_wandb", action='store_true')
    parser.add_argument("--use_cluster", action='store_true')
    parser.add_argument("--cluster_type", default="mean", type=str, help="mean, random", choices=["mean", "random"])

    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--add_imp", default=0, type=int, help="number of impliations")
    parser.add_argument("--num_head", default=1, type=int, help="number of attention heads in GAT layer")
    parser.add_argument("--threshold", default=0, type=float, help="threshold for decide entailment connection")
    parser.add_argument("--weight_decay", default=0, type=float, help="weight decay")

    parser.add_argument("--pretrained_model", default="BAAI/bge-base-en-v1.5", type=str, help="pretrained language model")
    parser.add_argument("--saved_model", default="", type=str, help="")
    parser.add_argument("--loss_type", default="full", type=str, help="full attention or last attention", choices=["full", "last"])
    parser.add_argument("--conn_type", default="full", type=str, help="fully connected graph or graph connection based on entailment scores")
                        #choices=["full", "t5-base", "t5-large", "google/flan-t5-base", "google/flan-t5-large", "google/flan-t5-xl", "google/flan-t5-xxl", "cosine"])

    parser.add_argument("--data", default="opinion-qa", type=str, help="dataset [opinion-qa, sugar]; which data to use.")

    parser.add_argument("--no_demo", action='store_true')

    args = parser.parse_args()

    set_seed(args.seed)
    # set_logger(args)
    # logger = logging.getLogger()
    main(args)