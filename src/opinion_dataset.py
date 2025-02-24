from torch.utils.data import Dataset
import torch
import tqdm
from itertools import combinations, permutations
from dataclasses import dataclass
import dgl
from collections import defaultdict, OrderedDict
from utils import *
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F

class OpinionDataset(Dataset):
    def __init__(self, root_dir, user_responses, demographics, opinions, questions, declaratives, implications, opinion_implication_dict, 
                 sentence_to_cluster=None, cluster_to_sentence=None, response_to_id=None, use_cluster=False, device='cuda', conn_type="full", 
                 entail_tokenizer=None, entail_model=None, ds_type="", survey_name="", add_imp=0, entail_threshold=0, seed=42):
        self.device = device
        self.user_responses = user_responses
        
        self.demographics = demographics
        self.opinions = opinions
        self.questions = questions
        self.declaratives = declaratives
        self.implications = implications

        self.decl_to_imp = dict()
        self.imp_to_decl = dict()

        for decl, imp in zip(self.declaratives, self.implications):
            self.decl_to_imp[decl] = imp
            for _imp in imp:
                self.imp_to_decl[_imp] = decl
        # print("decl_to_imp:", len(self.decl_to_imp), "imp_to_decl:", len(self.imp_to_decl))
        # self.opinion_embedding = opinion_embedding
        self.opinion_implication_dict = opinion_implication_dict
        self.num_op = 16

        self.entail_tokenizer = entail_tokenizer
        self.entail_model = entail_model

        self.conn_type = conn_type
        self.ds_type = ds_type

        self.add_imp = add_imp
        self.use_cluster = use_cluster
        if self.use_cluster:
            self.sentence_to_cluster = sentence_to_cluster
            self.cluster_to_sentence = cluster_to_sentence
            self.response_to_id = response_to_id
            self.num_clusters = len(cluster_to_sentence)
        else:
            self.sentence_to_cluster = None
            self.cluster_to_sentence = None
            self.response_to_id = None
            self.num_clusters = 0

        user_id_to_decl_triples = self.extract_correlation_information(self.user_responses)
        # print("len(decl_triples),", len(decl_triples))
        self.entail_decl_ht = []
        if self.conn_type.startswith("t5") or self.conn_type.startswith("google") or self.conn_type == "both":
            original_conn_type = self.conn_type
            if self.conn_type == "both":
                self.conn_type = "t5-base" # in case self.conn_type is both, then temporailty change it to cosine to get the triples
            self.entail_dir = os.path.join(root_dir, f"opinion_graph_walker/{self.conn_type}_opinion_triple/{survey_name}/")
            Path(self.entail_dir).mkdir(parents=True, exist_ok=True)
            self.entail_filename = f"user_opinion_triples_imp5_{self.ds_type}.pickle" #f"user_opinion_triples_imp5_1000_{self.ds_type}.pickle"
            self.entail_file_path = os.path.join(self.entail_dir, self.entail_filename)
            print("entail_file_path: {}, is exist? {}".format(self.entail_file_path, os.path.exists(self.entail_file_path)))
            if os.path.exists(self.entail_file_path):
                self.user_id_to_triples = load_pickle(self.entail_file_path)
                if self.conn_type.startswith("t5"):
                    for user_id, triples in self.user_id_to_triples.items():
                        _triples = []
                        head_tail = []
                        for triple in triples:
                            if triple[2] >= entail_threshold:
                                triple = (self.opinions.index(self.opinions[triple[0]]), self.opinions.index(self.opinions[triple[1]]), triple[2]) # for consistent index number
                                _triples.append(triple)
                                head_tail.append((triple[0], triple[1]))

                                # if both head and tail are declarative opinions and entail each other, keep them in a separate list
                                op1, op2 = self.opinions[triple[0]], self.opinions[triple[1]]
                                if op1 in self.decl_to_imp and op2 in self.decl_to_imp:
                                    self.entail_decl_ht.append((self.opinions.index(op1), self.opinions.index(op2)))

                        # connect all declarative opinions from each other
                        decl_triples = user_id_to_decl_triples[user_id]
                        for decl_trp in decl_triples:
                            if (decl_trp[0], decl_trp[1]) not in head_tail:
                                _triples.append(decl_trp)
                        
                        # connect all declarative opinions with implications
                        # print("user_id: {}".format(user_id))
                        decl_indices = list(set([trp[0] for trp in decl_triples]))
                        for decl_idx in decl_indices:
                            # print("decl_idx:", decl_idx)
                            imps = self.decl_to_imp[self.opinions[decl_idx]]
                            for imp in imps:
                                imp_idx = self.opinions.index(imp)
                                if (decl_idx, imp_idx) not in head_tail:
                                    _triples.append((decl_idx, imp_idx, 0))
                                    # print("[add] decl_idx:", decl_idx, "imp_idx:", imp_idx)
                        self.user_id_to_triples[user_id] = _triples

            else:
                self.triples_check_dict = dict()
                self.user_id_to_triples = self.create_opinion_triples(self.user_responses)
            
            if original_conn_type == "both":
                self.conn_type = original_conn_type # change it back to both to continue to merge the t5 and cosine triples
                self.user_id_to_triples_t5 = self.user_id_to_triples

        if self.conn_type == "cosine" or self.conn_type == "both":
            original_conn_type = self.conn_type
            if self.conn_type == "both":
                self.conn_type = "cosine" # in case self.conn_type is both, then temporailty change it to cosine to get the triples
            self.entail_dir = os.path.join(root_dir, f"opinion_graph_walker/{self.conn_type}_opinion_triple/{survey_name}/")
            Path(self.entail_dir).mkdir(parents=True, exist_ok=True)
            self.entail_filename = f"user_opinion_triples_{self.conn_type}_{self.ds_type}.pickle"
            self.entail_file_path = os.path.join(self.entail_dir, self.entail_filename)
            print("entail_file_path: {}, is exist? {}".format(self.entail_file_path, os.path.exists(self.entail_file_path)))
            if os.path.exists(self.entail_file_path):
                self.user_id_to_triples = load_pickle(self.entail_file_path)
            else:
                self.sent_model = SentenceTransformer('BAAI/bge-large-en').to(device)
                self.triples_check_dict = dict()
                self.user_id_to_triples = self.create_opinion_triples(self.user_responses)
            
            for user_id, triples in self.user_id_to_triples.items():
                _triples = []
                for triple in triples:
                    if triple[2] >= entail_threshold:
                        _triples.append(triple)
                self.user_id_to_triples[user_id] = _triples
            
            if original_conn_type == "both":
                self.conn_type = original_conn_type # change it back to both to continue to merge the t5 and cosine triples
                self.user_id_to_triples_cosine = self.user_id_to_triples

        if self.conn_type == "both":
            self.user_id_to_triples = dict()
            user_id_to_head_tail_pairs = dict()
            for user_id, triples in self.user_id_to_triples_t5.items():
                # print(user_id, len(triples))
                if user_id not in user_id_to_head_tail_pairs:
                    user_id_to_head_tail_pairs[user_id] = []
                
                head_tail_pairs = []
                for triple in triples:
                    ht_pair = (triple[0], triple[1])
                    if ht_pair not in head_tail_pairs:
                        head_tail_pairs.append(ht_pair)
                user_id_to_head_tail_pairs[user_id] = head_tail_pairs
            
            for user_id, triples in self.user_id_to_triples_cosine.items():
                head_tail_pairs = user_id_to_head_tail_pairs[user_id]
                for triple in triples:
                    ht_pair = (triple[0], triple[1])
                    if ht_pair not in head_tail_pairs and triple[-1] >= 0.9:
                        head_tail_pairs.append(ht_pair)
                
                head_tail_pairs = [(pair[0], pair[1], 1) for pair in head_tail_pairs]
                user_id_to_head_tail_pairs[user_id] = list(set(head_tail_pairs))
            
            self.user_id_to_triples = user_id_to_head_tail_pairs
            del self.user_id_to_triples_cosine
            del self.user_id_to_triples_t5
            del self.entail_tokenizer
            del self.entail_model


        if self.conn_type == "full":
            self.user_id_to_triples = self.create_opinion_triples(self.user_responses)

        # if self.conn_type.startswith("google"):
        for user_id, triples in list(self.user_id_to_triples.items())[:5]:
        #         # _triples = []
        #         for triple in triples:
        #             if (triple[1], triple[0], triple[2]) not in triples:
        #                 # print("add!")
        #                 triples.append((triple[1], triple[0], triple[2]))
        #             # else:
        #             #     _triples.append((triple[0], triple[1], triple[2]))
            print("user_id:", user_id, "triples:", len(triples))
        #         self.user_id_to_triples[user_id] = triples
        print("self.entail_decl_ht:", len(list(set(self.entail_decl_ht))), len(self.entail_decl_ht))
        self.entail_decl_ht = list(set(self.entail_decl_ht))

    def __len__(self):
        return len(self.user_responses)

    def get_entailment_inference(self, premise, hypothesis, tokenizer, model, device):
        # template = f"rte sentence1: {premise} sentence2: {hypothesis}"        
        if self.conn_type.startswith("t5"):
            templates = [f"rte sentence1: {p} sentence2: {h}" for p, h in zip(premise, hypothesis)]
            with torch.no_grad():
                source = tokenizer.batch_encode_plus(templates, max_length=128, pad_to_max_length=True,return_tensors='pt')
                input_ids = source['input_ids'].to(device)
                # print("input_ids:", input_ids)
                output = model(input_ids=input_ids)
                logits = output.logits
                logits = logits.softmax(dim=-1)
                pred = logits.argmax(dim=1)
            return logits.tolist(), pred.tolist()

    def get_demographic_index(self, explicit_persona):
        demo_index = []
        demo_string_list = []
        for exp_per in explicit_persona:
            demo_string = None
            for key, value in exp_per.items():
                if value.lower() == "refused":
                    continue
                if key.lower() == "citizenship":
                    if value.lower() == "yes":
                        demo_string = "This person is American citizen."
                    else:
                        demo_string = "This person is not American citizen."
                else:
                    demo_string = f"This person's {key.lower()} is {value}."
            if demo_string is None:
                continue
            demo_index.append(self.demographics.index(demo_string))
            demo_string_list.append(demo_string)
        return demo_index, demo_string_list
    
    def get_opinion_index(self, implicit_persona):
        opinion_index = []
        num_op = min(self.num_op, len(implicit_persona))
        implicit_persona = implicit_persona[:num_op]
        opinion_string_list = []
        decl_indices, implication_indices = [], []
        for imp_per in implicit_persona:
            qid = imp_per['qid']
            question = imp_per['question']
            answer = imp_per['answer']
            answer = calibrate_data(answer)

            decl_opinion = self.opinion_implication_dict[f"{qid}|{question}|{answer}"]["declarative_opinion"]
            decl_idx = self.opinions.index(decl_opinion)
            opinion_index.append(decl_idx)
            decl_indices.append(decl_idx)
            opinion_string_list.append(decl_opinion)

            if self.add_imp > 0:
                implications = self.opinion_implication_dict[f"{qid}|{question}|{answer}"]["generation"]
                _implication_indices = []
                for gen in implications:
                    imp_idx = self.opinions.index(gen)
                    _implication_indices.append(imp_idx)
                    opinion_index.append(imp_idx)
                    # implication_indices.append(imp_idx)
                    opinion_string_list.append(gen)
                implication_indices.append(_implication_indices)
        return opinion_index, opinion_string_list, decl_indices, implication_indices
    

    def extract_correlation_information(self, user_responses):
        # key: (qid1, qid2) value: (ans1, ans2) = count
        user_id_to_decl_triples = dict()
        for user_resp in user_responses:
            user_id = user_resp['user_id']
            implicit_persona = user_resp['implicit_persona']
            opinion_index, opinion_string_list, _, _ = self.get_opinion_index(implicit_persona)
            decl_ops = []
            for imp_per in implicit_persona:
                qid = imp_per['qid']
                question = imp_per['question']
                answer = imp_per['answer']
                answer = calibrate_data(answer)
                decl_op = self.opinion_implication_dict[f"{qid}|{question}|{answer}"]["declarative_opinion"]
                decl_ops.append(decl_op)

            decl_idxs = []
            for op_idx, op_str in zip(opinion_index, opinion_string_list):
                if op_str in decl_ops:
                    decl_idxs.append(op_idx)
            
            assert len(decl_idxs) == 16
            triples = []
            for perm in permutations(decl_idxs, 2):
                triples.append((perm[0], perm[1], 0))
            user_id_to_decl_triples[user_id] = triples
        
        return user_id_to_decl_triples
            
    def create_opinion_triples(self, user_responses):
        user_id_to_triples = OrderedDict()
        print("Constructing opinion triples...", end=" ")
        for user_resp in tqdm.tqdm(user_responses, total=len(user_responses)):
            user_id = user_resp['user_id']

            if user_id in user_id_to_triples:
                continue

            implicit_persona = user_resp['implicit_persona']
            opinion_index, opinion_string_list, _, _ = self.get_opinion_index(implicit_persona)

            assert len(opinion_index) == len(opinion_string_list)

            triples = []
            if self.conn_type == "full":
                for perm in permutations(opinion_index, 2):
                    triples.append((perm[0], perm[1], 0))
            elif self.conn_type.startswith("cosine"):
                step = 300
                perms = list(combinations(opinion_index, 2))
                
                opinion1, opinion2 = [], []
                opinion1_idx, opinion2_idx = [], []
                for perm in perms[:]:
                    op1_idx = self.opinions.index(self.opinions[perm[0]])
                    op2_idx = self.opinions.index(self.opinions[perm[1]])

                    if (op1_idx, op2_idx) in self.triples_check_dict:
                        triples.append((op1_idx, op2_idx, self.triples_check_dict[(op1_idx, op2_idx)]))
                        triples.append((op2_idx, op1_idx, self.triples_check_dict[(op1_idx, op2_idx)]))
                        continue

                    op1 = self.opinions[op1_idx]
                    op2 = self.opinions[op2_idx]

                    opinion1_idx.append(op1_idx)
                    opinion2_idx.append(op2_idx)
                    opinion1.append(op1)
                    opinion2.append(op2)
                
                op1_embedding = self.sent_model.encode(opinion1, convert_to_tensor=True).to(self.device)
                normalized_op1_embedding = F.normalize(op1_embedding, p=2, dim=1)
                op2_embedding = self.sent_model.encode(opinion2, convert_to_tensor=True).to(self.device)
                normalized_op2_embedding = F.normalize(op2_embedding, p=2, dim=1)

                for i in range(normalized_op1_embedding.shape[0]):
                    cos_sim = (normalized_op1_embedding[i] @ normalized_op2_embedding[i].T).item()
                    if cos_sim >= 0.5:
                        triples.append((opinion1_idx[i], opinion2_idx[i], cos_sim))
                        triples.append((opinion2_idx[i], opinion1_idx[i], cos_sim))

            elif self.conn_type.startswith("t5"):
                all_logits, all_preds = [], []
                step = 1000
                perms = list(permutations(opinion_index, 2))
                for i in range(0, len(perms), step):
                    premise, hypothesis = [], []
                    premise_idx, hypothesis_idx = [], []
                    start = i
                    end = min(i+step, len(perms))
                    print("start, end", start, end)
                    for perm in perms[start:end]:
                        op1_idx = self.opinions.index(self.opinions[perm[0]])
                        op2_idx = self.opinions.index(self.opinions[perm[1]])

                        if (op1_idx, op2_idx) in self.triples_check_dict:
                            triples.append((op1_idx, op2_idx, self.triples_check_dict[(op1_idx, op2_idx)]))
                            continue

                        premise_idx.append(op1_idx)
                        hypothesis_idx.append(op2_idx)
                        premise.append(self.opinions[op1_idx])
                        hypothesis.append(self.opinions[op2_idx])
                    if len(premise) == 0:
                        continue
                    logits, preds = self.get_entailment_inference(premise=premise, hypothesis=hypothesis, tokenizer=self.entail_tokenizer, model=self.entail_model, device=self.device)
                    # all_logits.extend(logits)
                    # all_preds.extend(preds)


                    # logits = all_logits
                    # preds = all_preds
                    assert len(preds) == len(premise) == len(premise_idx)
                    # print("logits:", len(logits), "preds:", len(preds), "perms:", len(perms))
                    
                    for i in range(len(premise)):
                        if preds[i] == 0 or preds[i] == 'yes':
                            if logits is None:
                                triples.append((premise_idx[i], hypothesis_idx[i], 1))
                                self.triples_check_dict.update({(premise_idx[i], hypothesis_idx[i]): 1})
                                # print("entail ({}): {} --> {}".format(1, self.opinions[premise_idx[i]], self.opinions[hypothesis_idx[i]]))
                            else:
                                triples.append((premise_idx[i], hypothesis_idx[i], logits[i][0]))
                                self.triples_check_dict.update({(premise_idx[i], hypothesis_idx[i]): logits[i][0]})
                                # print("entail ({}%): {} --> {}".format(logits[i][0], self.opinions[premise_idx[i]], self.opinions[hypothesis_idx[i]]))
                        # else:
                        #     print("not entail ({}%): {} --> {}".format(0, self.opinions[premise_idx[i]], self.opinions[hypothesis_idx[i]]))

            else:
                raise Exception(f"Invalid connection type! - conn_type: {self.conn_type}")
            user_id_to_triples[user_id] = triples
        print("end!")
        
        if self.conn_type.startswith("t5") or self.conn_type.startswith("google") or self.conn_type.startswith("cosine"):
            save_pickle(self.entail_file_path, user_id_to_triples)

        return user_id_to_triples
    
    def create_opinion_target_triples(self, opinion_index, candidate_index):
        triples = []
        for cand_idx in candidate_index:
            for op_idx in opinion_index:
                triples.append((op_idx, cand_idx, 0))

        # if self.conn_type == "full":
        #     for cand_idx in candidate_index:
        #         for op_idx in opinion_index:
        #             triples.append((op_idx, cand_idx, 0))
        # elif self.conn_type == "t5":
        #     premise, hypothesis = [], []
        #     premise_idx, hypothesis_idx = [], []
        #     for cand_idx in candidate_index:
        #         for op_idx in opinion_index:
        #             op1_idx = op_idx
        #             op2_idx = cand_idx

        #             premise_idx.append(op1_idx)
        #             hypothesis_idx.append(op2_idx)
        #             premise.append(self.opinions[op1_idx])
        #             hypothesis.append(self.opinions[op2_idx])
                
        #     logits, preds = self.get_entailment_inference(premise=premise, hypothesis=hypothesis, tokenizer=self.entail_tokenizer, model=self.entail_model, device=self.device)
            
        #     assert len(premise) == len(hypothesis) == len(logits) == len(preds)
            
        #     for i in range(len(premise)):
        #         if preds[i] == 0:
        #             triples.append((premise_idx[i], hypothesis_idx[i], logits[i][0]))

        return triples

    def __getitem__(self, idx):
        response = self.user_responses[idx]

        user_id = response['user_id']
        qid = response['qid']
        question = response['question']
        choices = response['choices']
        answer = response['answer']
        
        implicit_persona = response['implicit_persona']
        explicit_persona = response['explicit_persona']

        ########### start: construct demographic embedding ###########
        demographic_index, demo_string_list = self.get_demographic_index(explicit_persona)
        demographic_index = torch.tensor(demographic_index, dtype=torch.long, device=self.device)
        ########### end: construct demographic embedding ###########

        ########### start: construct opinion embedding ###########
        opinion_index, opinion_string_list, decl_indices, implication_indices = self.get_opinion_index(implicit_persona)
        if self.use_cluster:
            ############# if use cluster ############
            opinion_index = [self.sentence_to_cluster[self.opinions[idx]] for idx in opinion_index]
            decl_indices = [self.sentence_to_cluster[self.opinions[idx]] for idx in decl_indices]
            implication_indices = [self.sentence_to_cluster[self.opinions[idx]] for idx in implication_indices]
            #########################################

        seed_nodes = opinion_index
        ########### end: construct opinion embedding ###########
        # print("seed_nodes:", seed_nodes)
        # assert False
        opinion_triples = self.user_id_to_triples[user_id]
        # print("opinion_triples:", len(opinion_triples))
        if self.use_cluster:
            opinion_triples = [(self.sentence_to_cluster[self.opinions[t[0]]], self.sentence_to_cluster[self.opinions[t[1]]], t[2]) for t in opinion_triples]

        triples = opinion_triples
        # implicit_questions = response['implicit_questions']
        candidate_index = []
        candidate_decl_index = []
        for choice in choices:
            choice = calibrate_data(choice)
            decl_opinion = self.opinion_implication_dict[f"{qid}|{question}|{choice}"]["declarative_opinion"]
            decl_idx = self.opinions.index(decl_opinion)
            if self.use_cluster:
                decl_idx = self.response_to_id[decl_opinion] + self.num_clusters # if use cluster
            candidate_index.append(decl_idx)
            candidate_decl_index.append(decl_idx)
            opinion_string_list.append(decl_opinion)

        if set(candidate_index) == 1:
            print(opinion_string_list)
            raise Exception(f"Invalid choices -- user_id: {user_id}, qid: {qid}, choices: {choices}")

        target_index = self.opinions.index(self.opinion_implication_dict[f"{qid}|{question}|{answer}"]["declarative_opinion"])
        if self.use_cluster:
            target_index = self.response_to_id[self.opinion_implication_dict[f"{qid}|{question}|{answer}"]["declarative_opinion"]] + self.num_clusters # if use cluster
        opinion_target_triples = self.create_opinion_target_triples(opinion_index, candidate_index)
        
        opinion_index += candidate_index
        decl_indices += candidate_decl_index
        triples = triples + opinion_target_triples

        # differentiate edge types
        new_triples = []
        for trp in triples:
            h, t, r = trp

            if self.use_cluster:
                if t not in candidate_index:
                    edge_type = 0
                else:
                    edge_type = 1
            else:
                if self.conn_type == "t5-base":
                    # if head=decl and tail=decl
                    # represent them as 5 one-hot vector relation types [conn, entail, decl-decl, decl-imp, imp-imp]
                    # eg. [11100]: decl-decl: decl-decl entails each other, and connected
                    # [10100]: all decls are connected
                    # [10010]: decl-imp or imp-decl are weakly entailed each other, but connected
                    # [11010]: decl-imp or imp-decl are entailed each other, thus connected
                    # [11001]: imp-imp are entailed each other, thus connected
                    # [10001]: imp-imp are weakly entailed each other, but connected
                    #
                    # in case of fully connected version (we don't know entailment information):
                    # [10100]: decl-decl
                    # [10010]: decl-imp
                    # [10001]: imp-imp
                    if self.opinions[h] in self.decl_to_imp and self.opinions[t] in self.decl_to_imp:
                        # if decl-decl entails each other
                        if (h, t) in self.entail_decl_ht:
                            edge_type = [1, 1, 0, 0]
                        else: # decl-decl doesn't entail each other, but connected
                            edge_type = [0, 1, 0, 0]
                    elif self.opinions[h] in self.decl_to_imp and self.opinions[t] in self.imp_to_decl: # if h=decl and t=imp
                            if r>=0.5 or (self.opinions[h] in self.decl_to_imp and self.opinions[t] in self.decl_to_imp[self.opinions[h]]):
                                edge_type = [1,0,1,0]
                            else:
                                edge_type = [0,0,1,0]
                    elif self.opinions[t] in self.decl_to_imp and self.opinions[h] in self.imp_to_decl: # if h=imp and t=decl
                            if r>=0.5 or (self.opinions[t] in self.decl_to_imp and self.opinions[h] in self.decl_to_imp[self.opinions[t]]):
                                edge_type = [1,0,1,0]
                            else:
                                edge_type = [0,0,1,0]
                    elif self.opinions[h] in self.imp_to_decl and self.opinions[t] in self.imp_to_decl:
                        if r >= 0.5:
                            edge_type = [1,0,0,1]
                        else:
                            edge_type = [0,0,0,1]
                    else:
                        raise Exception("Something is wrong with triple: h ({}), t ({}), r ({}), conn-type: {}".format(h, t, r, self.conn_type))
                elif self.conn_type == "full":
                    # if self.opinions[h] in self.decl_to_imp and self.opinions[t] in self.decl_to_imp:
                    edge_type = [0, 0, 0, 0]
                    # elif self.opinions[h] in self.decl_to_imp and self.opinions[t] in self.imp_to_decl: # if h=decl and t=imp
                    #     edge_type = [0,0,1,0]
                    # elif self.opinions[t] in self.decl_to_imp and self.opinions[h] in self.imp_to_decl: # if h=imp and t=decl
                    #     edge_type = [0,0,1,0]
                    # elif self.opinions[h] in self.imp_to_decl and self.opinions[t] in self.imp_to_decl:
                    #     edge_type = [0,0,0,1]
                    # else:
                    #     raise Exception("Something is wrong with triple: h ({}), t ({}), r ({}), conn-type: {}".format(h, t, r, self.conn_type))
                else:
                    raise Exception("Invalid conn type -- {}".format(self.conn_type))
            trp = (h, t, edge_type)

            new_triples.append(trp)
        triples = new_triples

        heads = [triple[0] for triple in triples]
        tails = [triple[1] for triple in triples]
        relations = [triple[2] for triple in triples]

        node2nodeId = OrderedDict() # defaultdict(lambda: -1)
        nodeId2node = OrderedDict() # defaultdict(lambda: -1)
        node2declType, nodeId2declType = OrderedDict(), OrderedDict()
        node2nodeType, nodeId2nodeType = OrderedDict(), OrderedDict()

        nodes = heads + tails
        nodes = list(sorted(set(nodes)))
        idx = 0
        for node in nodes:
            node2nodeId[node] = idx
            nodeId2node[idx] = node
            if self.opinions[node] in self.decl_to_imp: #if self.opinions[node] in self.decl_to_imp:
                decl_type = self.declaratives.index(self.opinions[node])
                node_type = 0
            elif self.opinions[node] in self.imp_to_decl:
                decl_type = self.declaratives.index(self.imp_to_decl[self.opinions[node]])
                node_type = 1
            else:
                raise Exception("node: {} is not neither in decl_indices or implication_indices!".format(node))
            
            node2declType[node] = decl_type
            nodeId2declType[idx] = decl_type
            node2nodeType[node] = node_type
            nodeId2nodeType[idx] = node_type
            idx += 1

        indexed_head_nodes = [node2nodeId[head_entity] for head_entity in heads]
        indexed_tail_nodes = [node2nodeId[tail_entity] for tail_entity in tails]
        indexed_triples = [(head, tail, rel) for head, tail, rel in zip(indexed_head_nodes, indexed_tail_nodes, relations)]

        indexed_head_nodes = torch.tensor(indexed_head_nodes, dtype=(torch.int64))
        indexed_tail_nodes = torch.tensor(indexed_tail_nodes, dtype=(torch.int64))

        indexed_candidate_nodes = [node2nodeId[cand_idx] for cand_idx in candidate_index]
        indexed_target_node = node2nodeId[target_index]
        heads = torch.tensor(heads, dtype=(torch.int64))
        tails = torch.tensor(tails, dtype=(torch.int64))
        relations = torch.tensor(relations)

        indexed_seed_nodes = [node2nodeId[node] for node in seed_nodes]

        subgraph = dgl.graph((torch.tensor([], dtype=(torch.int64)), torch.tensor([], dtype=(torch.int64))))
        subgraph.add_edges(indexed_head_nodes, indexed_tail_nodes, {'edge_type': relations})
        
        edge_id_to_ht = dict()
        ht_to_edge_id = dict()
        for i, (h, t) in enumerate(zip(indexed_head_nodes, indexed_tail_nodes)):
            edge_id_to_ht[i] = (h, t)
            ht_to_edge_id[(h, t)] = i
        
        subgraph = dgl.remove_self_loop(subgraph)
        subgraph_nodes = subgraph.nodes().tolist()
        subgraph_node_ids = [nodeId2node[node] for node in subgraph_nodes]
        subgraph_node_ids = torch.tensor(subgraph_node_ids, dtype=(torch.int64))
        subgraph.ndata['nodeId'] = subgraph_node_ids # need to know whether it's declarative or implications

        subgraph_node_types = [nodeId2nodeType[node] for node in subgraph_nodes]
        subgraph_node_types = torch.tensor(subgraph_node_types, dtype=(torch.int64))
        subgraph.ndata['nodeType'] = subgraph_node_types

        subgraph_decl_types = [nodeId2declType[node] for node in subgraph_nodes]
        subgraph_decl_types = torch.tensor(subgraph_decl_types, dtype=(torch.int64))
        subgraph.ndata['declType'] = subgraph_decl_types

        question_idx = torch.tensor(self.questions.index(question), dtype=torch.long, device=self.device)
        return subgraph.to(self.device), indexed_seed_nodes, demographic_index, opinion_index, indexed_candidate_nodes, indexed_target_node, indexed_triples, nodeId2node, question_idx, edge_id_to_ht, ht_to_edge_id
