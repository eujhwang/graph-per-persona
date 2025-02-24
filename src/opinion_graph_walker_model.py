import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from torch.nn.functional import softmax
import dgl
from dgl import function as fn
from dgl.ops import edge_softmax
from torch.nn.utils.rnn import pad_packed_sequence
from transformers import AutoTokenizer, RobertaModel, AutoModel
import tqdm

class OutFlow(nn.Module):
    def __init__(self, in_dim, out_dim, num_head):
        super(OutFlow, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_head = num_head

        # Outflow Params
        self.w_q = nn.Parameter(torch.FloatTensor(size=(self.num_head, self.in_dim, self.out_dim)))
        self.w_k = nn.Parameter(torch.FloatTensor(size=(self.num_head, self.in_dim, self.out_dim)))

        self.leaky_relu = nn.LeakyReLU()
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.w_q, gain=gain)
        nn.init.xavier_normal_(self.w_k, gain=gain)

    def forward(self, graph, node, relation):
        feat_src = node.repeat(1, self.num_head, 1)     # [98, 5, 266]
        feat_rel = relation.repeat(1, self.num_head, 1) # [1720, 5, 266]
        feat_dst = feat_src # [98, 5, 266]

        # print("1. feat_src:", feat_src.shape, "feat_rel:", feat_rel.shape, "feat_dst:", feat_dst.shape)

        feat_src = feat_src.permute(1, 0, 2)    # [5, 98, 266]
        feat_dst = feat_dst.permute(1, 0, 2)    # [5, 98, 266]
        feat_rel = feat_rel.permute(1, 0, 2)    # [5, 1720, 266]

        # print("2. feat_src:", feat_src.shape, "feat_rel:", feat_rel.shape, "feat_dst:", feat_dst.shape)

        feat_dst_attn = torch.matmul(feat_dst, self.w_k)    # (heads X nodes X features) [5, 98, 266]
        feat_rel_attn = torch.matmul(feat_rel, self.w_k)    # [5, 1720, 266]
        feat_src_attn = torch.matmul(feat_src, self.w_q)    # [5, 98, 266]

        # print("3. feat_src_attn:", feat_src_attn.shape, "feat_rel_attn:", feat_rel_attn.shape, "feat_dst_attn:", feat_dst_attn.shape)

        feat_dst_attn = feat_dst_attn.permute(1, 0, 2)  # (nodes X heads X features) [98, 5, 266]
        feat_rel_attn = feat_rel_attn.permute(1, 0, 2)  # [1720, 5, 266]
        feat_src_attn = feat_src_attn.permute(1, 0, 2)  # [98, 5, 266]
        feat_src = feat_src.permute(1, 0, 2)

        # print("4. feat_src_attn:", feat_src_attn.shape, "feat_rel_attn:", feat_rel_attn.shape, "feat_dst_attn:", feat_dst_attn.shape)

        # Store the node features and attention features on the source and destination nodes
        graph.srcdata.update({"out_el": feat_src_attn})
        graph.dstdata.update({'ft': feat_src, 'out_er': feat_dst_attn})
        graph.edata.update({'out_erel': feat_rel_attn})

        # compute edge attention with respect to the source node and the respective edge relation
        graph.apply_edges(fn.v_dot_u('out_er', 'out_el', 'out_e'))
        e = graph.edata.pop('out_e')    # [1720, 5, 1]

        graph.apply_edges(fn.e_dot_u('out_erel', 'out_er', 'out_e'))
        re = graph.edata.pop('out_e')   # [1720, 5, 1]

        # print("5. e:", e.shape, "re:", re.shape)
        e = e + re
        e = self.leaky_relu(e)

        # compute edge softmax
        edge_attention = edge_softmax(graph, e, norm_by="src")
        edge_attention = edge_attention.squeeze(-1) # [1720, 5]
        # print("6. edge_attention:", edge_attention.shape)

        edge_attention = edge_attention.sum(-1)     # [1720]
        # print("7. edge_attention:", edge_attention.shape)
        edge_attention = ((edge_attention))/self.num_head

        # print("8. edge_attention:", edge_attention.shape)
        # assert False
        return edge_attention


class InFlow(nn.Module):
    def __init__(self, in_dim, out_dim, num_head):
        super(InFlow, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_head = num_head
        print("inflow indim: {}, outdim:{}, num_head: {}".format(self.in_dim, self.out_dim, self.num_head))
        self.w_m = nn.Parameter(torch.FloatTensor(size=(self.num_head, self.in_dim, self.out_dim)))
        self.w_q = nn.Parameter(torch.FloatTensor(size=(self.num_head, self.in_dim, self.out_dim)))
        self.w_k = nn.Parameter(torch.FloatTensor(size=(self.num_head, self.in_dim, self.out_dim)))
        self.w_h_e = nn.Parameter(torch.FloatTensor(size=(self.num_head*self.out_dim, self.out_dim)))
        self.w_h_d = nn.Parameter(torch.FloatTensor(size=(self.in_dim, self.out_dim)))
        self.w_h_q = nn.Parameter(torch.FloatTensor(size=(self.in_dim, self.out_dim)))

        self.leaky_relu = nn.LeakyReLU()
        
        self.reset_parameters()
        # equation (1)
        # self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # self.fd = nn.Linear(in_dim, out_dim, bias=False)
        # self.fh = nn.Linear(in_dim*2, out_dim, bias=True)

        # self.w_h_o = nn.Parameter(torch.FloatTensor(size=(self.num_head, in_dim, out_dim)))
        # self.w_h_d = nn.Parameter(torch.FloatTensor(size=(self.num_head, in_dim, out_dim)))
        # self.w_h = nn.Parameter(torch.FloatTensor(size=(2*in_dim, out_dim)))

        # equation (2)
        # self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.w_m, gain=gain)
        nn.init.xavier_normal_(self.w_q, gain=gain)
        nn.init.xavier_normal_(self.w_k, gain=gain)
        nn.init.xavier_normal_(self.w_h_e, gain=gain)
        nn.init.xavier_normal_(self.w_h_d, gain=gain)
        nn.init.xavier_normal_(self.w_h_q, gain=gain)

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, graph, node, relation, background=None, question=None):
        feat_src = node.repeat(1, self.num_head, 1)     # [99, 5, 266]
        feat_rel = relation.repeat(1, self.num_head, 1) # [1569, 5, 266]
        feat_dst = feat_src                             # [99, 5, 266]

        # print("1. feat_src:", feat_src.shape, "feat_rel:", feat_rel.shape, "feat_dst:", feat_dst.shape)
        
        feat_src = feat_src.permute(1, 0, 2)    # [5, 99, 266]
        feat_dst = feat_dst.permute(1, 0, 2)    # [5, 1569, 266]
        feat_rel = feat_rel.permute(1, 0, 2)    # [5, 99, 266]

        # print("2. feat_src:", feat_src.shape, "feat_rel:", feat_rel.shape, "feat_dst:", feat_dst.shape)

        feat_dst_attn = torch.matmul(feat_dst, self.w_q)    # [5, 99, 266]
        feat_src_attn = torch.matmul(feat_src, self.w_k)    # [5, 99, 266]
        feat_rel_attn = torch.matmul(feat_rel, self.w_k)    # [5, 1569, 266]

        # print("3. feat_src_attn:", feat_src_attn.shape, "feat_dst_attn:", feat_dst_attn.shape, "feat_rel_attn:", feat_rel_attn.shape)

        feat_src = torch.matmul(feat_src, self.w_m) # [5, 99, 266]
        feat_rel = torch.matmul(feat_rel, self.w_m) # [5, 1569, 266]

        # print("4. feat_src:", feat_src.shape, "feat_rel:", feat_rel.shape)

        feat_dst_attn = feat_dst_attn.permute(1, 0, 2)
        feat_src_attn = feat_src_attn.permute(1, 0, 2)
        feat_rel_attn = feat_rel_attn.permute(1, 0, 2)
        feat_src = feat_src.permute(1, 0, 2)
        feat_rel = feat_rel.permute(1, 0, 2)

        # print("4-1. feat_src_attn:", feat_src_attn.shape, "feat_dst_attn:", feat_dst_attn.shape, "feat_rel_attn:", feat_rel_attn.shape)
        # print("4-2. feat_src:", feat_src.shape, "feat_rel:", feat_rel.shape)

        graph.srcdata.update({'ft_ent': feat_src, "in_el": feat_src_attn})
        graph.dstdata.update({'in_er': feat_dst_attn})
        graph.edata.update({'ft_rel': feat_rel, 'in_rel': feat_rel_attn})
        
        # Computing Attention Weights. Equation 3a
        graph.apply_edges(fn.u_dot_v('in_el', 'in_er', 'in_e'))
        e = graph.edata.pop('in_e')
        graph.apply_edges(fn.e_dot_v('ft_rel', 'in_er', 'in_e'))
        re = graph.edata.pop('in_e')
        
        # print("5. e:", e.shape, "re:", re.shape)

        e = e + re
        e = self.leaky_relu(e)

        # compute edge softmax
        edge_attention = edge_softmax(graph, e, norm_by="dst") # [1391, 1, 1]
        graph.apply_edges(fn.u_add_e('ft_ent', 'ft_rel', "edge_message")) # edge_message: [1391, 1, 266]
        # print("6. edge_attention:", edge_attention.shape, "edge_message:", graph.edata["edge_message"].shape)
        
        edge_message = graph.edata["edge_message"]*edge_attention # [1391, 1, 266]        
        # print("7. edge_message:", edge_message.shape)

        graph.edata.update({"edge_message": edge_message})
        # message passing. Equation 2
        graph.update_all(fn.copy_e("edge_message", "message"), fn.sum('message', 'ft_ent'))

        # rst contains the inflow nodes features
        rst = graph.ndata['ft_ent'].view(graph.num_nodes(), -1) # ft_ent: [98, 1, 266], rst: [98, 266 x num_head]
        # print("7. ft_ent:", graph.ndata['ft_ent'].shape)
        # print("8. rst:", rst.shape)

        #Equation 5
        if background is not None:
            node_inflow_feat = torch.mm(rst, self.w_h_e) + torch.mm(background + question, self.w_h_d) # ft_ent: [98, 266]
        else:
            node_inflow_feat = torch.mm(rst, self.w_h_e) + torch.mm(question, self.w_h_q)
        return  node_inflow_feat




class AttnIO(nn.Module):
    def __init__(self, init_dim, opinions, demographics, questions, declaratives, implications, sentence_to_cluster, cluster_to_sentence, response_to_id, cluster_embeddings, pretrained_model, path_len, 
                 sent_tokenizer, sent_model, num_head, use_cluster=False, cluster_type="mean", no_demo=False):
        super(AttnIO, self).__init__()
        # self.n_opinion = n_opinion
        # self.n_demographic = n_demographic
        
        self.path_len = path_len
        if pretrained_model == "roberta-large" or pretrained_model == "albert-large-v2" or pretrained_model == "microsoft/deberta-v3-large":
            self.init_dim = 1024
        elif pretrained_model == "roberta-base" or pretrained_model == "distilbert-base-uncased" or pretrained_model == "microsoft/deberta-v3-base":
            self.init_dim = 768
        elif pretrained_model == "BAAI/bge-base-en-v1.5":
            self.init_dim = 768
        elif pretrained_model == "BAAI/bge-large-en-v1.5":
            self.init_dim = 1024
        else:
            raise Exception(f"Invalid pretrained model name: {pretrained_model}")

        self.hid_dim = 256
        self.out_dim = 256
        self.type_dim = 1
        self.num_head = num_head
        self.use_cluster = use_cluster
        # if self.use_cluster:
        #     self.init_dim = init_dim

        self.declaratives = declaratives
        self.implications = implications

        self.linear = nn.Linear(self.init_dim + 2*self.type_dim, self.out_dim)

        # self.opinion_decl_embedding = nn.Embedding(len(self.declaratives), self.type_dim) # 0: declarative embedding, 1: implication embedding
        # self.opinion_type_embedding = nn.Embedding(2, self.type_dim) # 0: declarative embedding, 1: implication embedding
        self.opinion_decl_embedding = torch.eye(len(self.declaratives)).cuda()
        self.opinion_type_embedding = torch.tensor([[1, 0], [0, 1]], dtype=torch.float).cuda()
        if self.use_cluster:
            self.edge_type_embedding = nn.Embedding(2, self.out_dim) # 0: op-non target, 1: op-target
        else:
            self.edge_type_embedding = nn.Embedding(len(self.declaratives)+1, self.out_dim) # 0: decl-decl, 1: decl-imp, 2: imp-decl, 3: imp-imp
        self.node_decl_emb_layer = nn.Linear(len(declaratives), self.type_dim)
        self.node_type_emb_layer = nn.Linear(2, self.type_dim)
        self.edge_type_emb_layer = nn.Linear(4, self.out_dim)
        self.attn_layer = SelfAttentionLayer(768)
        # self.opinion_encoder = SentenceEncoder(self.init_dim, self.hid_dim//2)
        # self.demographic_encoder = SentenceEncoder(768, self.hid_dim//2)
        
        self.out_w_init_op = nn.Parameter(torch.FloatTensor(size=(self.out_dim, 1)).cuda()) # projection layer for opinion embedding
        self.out_w_init_demo = nn.Parameter(torch.FloatTensor(size=(768, self.out_dim)).cuda())
        self.out_w_init_ques = nn.Parameter(torch.FloatTensor(size=(768, self.out_dim)).cuda())

        # # Outflow Layers
        self.outflow_layers = nn.ModuleList()
        # self.gat_layers = nn.ModuleList()
        self.inflow_layers = nn.ModuleList()
        for i in range(0, path_len-1, 1):
            self.outflow_layers.append(OutFlow(in_dim=self.out_dim, out_dim=self.out_dim, num_head=self.num_head))
            self.inflow_layers.append(InFlow(in_dim=self.out_dim, out_dim=self.out_dim, num_head=self.num_head))
        
        self.sent_tokenizer = sent_tokenizer
        self.sent_model = sent_model

        if self.use_cluster:
            self.sentence_to_cluster = sentence_to_cluster
            self.cluster_to_sentence = cluster_to_sentence
            self.response_to_id = response_to_id
            self.cluster_embeddings = cluster_embeddings
            num_clusters = len(self.cluster_to_sentence)
            if cluster_type == "random":
                self.opinion_embeddings = nn.Embedding(num_clusters+len(response_to_id), self.init_dim)
            elif cluster_type == "mean":
                self.opinion_embeddings = self.from_pretrained(self.cluster_embeddings, freeze=False)
            else:
                raise Exception("Invalid cluster type! -- cluster type: {}".format(cluster_type))
        else:
            self.opinion_embeddings = self.from_pretrained(self.get_sentence_embedding(opinions), freeze=False)
        print("opinion_embeddings:", self.opinion_embeddings)

        self.demographic_embeddings = self.from_pretrained(self.get_sentence_embedding(demographics), freeze=False)
        self.question_embeddings = self.from_pretrained(self.get_sentence_embedding(questions), freeze=False)

        self.no_demo = no_demo
        self.reset_parameters()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.out_w_init_op, gain=gain)
        nn.init.xavier_normal_(self.out_w_init_demo, gain=gain)
        nn.init.xavier_normal_(self.out_w_init_ques, gain=gain)

    def from_pretrained(self, embeddings, freeze=True):
        assert embeddings.dim() == 2, \
            'Embeddings parameter is expected to be 2-dimensional'
        rows, cols = embeddings.shape
        embedding = torch.nn.Embedding(num_embeddings=rows, embedding_dim=cols)
        embedding.weight = torch.nn.Parameter(embeddings)
        embedding.weight.requires_grad = not freeze
        return embedding

    def get_roberta_embedding(self, sentences):
        embeddings = []
        for sentence in tqdm.tqdm(sentences):
            inputs = self.sent_tokenizer([sentence], max_length=32, padding=True, return_tensors="pt")
            outputs = self.sent_model(input_ids=inputs.input_ids.to(self.sent_model.device), attention_mask=inputs.attention_mask.to(self.sent_model.device))
            pooled_output = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(pooled_output)
        embeddings = torch.cat(embeddings, dim=0)
        return embeddings

    def get_sentence_embedding(self, sentences):
        encoded_input = self.sent_tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            model_output = self.sent_model(
                input_ids=encoded_input['input_ids'].to(self.sent_model.device),
                attention_mask=encoded_input['attention_mask'].to(self.sent_model.device),
                token_type_ids=encoded_input['token_type_ids'].to(self.sent_model.device)
            )
            # Perform pooling. In this case, cls pooling.
            embeddings = model_output[0][:, 0]
        print("embeddings:", embeddings.shape)
        return embeddings

    def forward(self, graphs, seed_sets, demographics, opinions, questions=None):
        """
        demographic_embedding: [4, 12, 768] -> [4, 1, 512]
        opinion_embedding: [4, 16, 768] -> [4, 16, 512]
        len(graphs), len(seed_sets), len(demographic_list), len(opinion_list)
        21 16 12 21
        """

        bsz = len(graphs)
        updated_graphs = []
        for b in range(bsz):
            graph = graphs[b]
            # demographic_list = demographics[b]
            # opinion_list = opinions[b]

            # print("node_ids:", graph.ndata["nodeId"].shape, graph.ndata["nodeId"])
            node_ids = graph.ndata["nodeId"]
            node_types = graph.ndata["nodeType"]
            decl_types = graph.ndata["declType"]
            edge_types = graph.edata["edge_type"]
            edge_types = torch.tensor(edge_types, dtype=torch.float)

            opinion_type_embedding = self.opinion_type_embedding[node_types] # [97, 2]
            opinion_decl_embedding = self.opinion_decl_embedding[decl_types] # [97, 304]  
            opinion_type_embedding = self.node_type_emb_layer(opinion_type_embedding) # [97, 1]
            opinion_decl_embedding = self.node_decl_emb_layer(opinion_decl_embedding) # [304, 1]
            edge_type_embedding = self.edge_type_emb_layer(edge_types)

            opinion_embedding = self.opinion_embeddings(node_ids) # [num_opinions+num_answer_choices, hid_dim]
            # opinion_embedding = self.opinion_encoder(opinion_embedding.unsqueeze(0), self_attn=False).squeeze(0) # [73, 512]
            opinion_embedding = torch.cat([opinion_embedding, opinion_decl_embedding, opinion_type_embedding], dim=-1)
            opinion_embedding = self.linear(opinion_embedding)
            question_embedding = torch.matmul(self.question_embeddings(questions[b]).unsqueeze(0), self.out_w_init_ques)

            if self.no_demo:
                demographic_embedding = None
                seedset_attention = torch.matmul(opinion_embedding, question_embedding.T)
            else:
                demographic_embedding = self.demographic_embeddings(demographics[b])
                demographic_embedding = self.attn_layer(demographic_embedding.unsqueeze(0)) # [1, 768]
                demographic_embedding = torch.matmul(demographic_embedding, self.out_w_init_demo)
                seedset_attention = torch.matmul(opinion_embedding, (demographic_embedding+question_embedding).T)
            
            seedset_attention[seed_sets[b]] += 10000
            seedset_attention -= 10000
            seedset_attention = torch.softmax(seedset_attention, dim=0)
            graph.ndata.update({"a_0": seedset_attention})

            """
            There is dgl and cuda related issue in the below code: https://github.com/dmlc/dgl/issues/1125
            With V100 machine, GAT crashes backend.
            """
            for p in range(self.path_len-1):
                if p == 0:
                    opinion_feat = opinion_embedding
                else:
                    opinion_feat = opinion_feat + opinion_embedding
                opinion_feat = self.inflow_layers[p](graph=graph, node=opinion_feat.unsqueeze(1), relation=edge_type_embedding.unsqueeze(1), background=demographic_embedding, question=question_embedding)
                edge_attention = self.outflow_layers[p](graph=graph, node=opinion_feat.unsqueeze(1), relation=edge_type_embedding.unsqueeze(1))
                graph.edata.update({f"transition_probs_{p+1}": edge_attention})
                graph.update_all(fn.u_mul_e(f"a_{p}", f"transition_probs_{p+1}", f"time_{p+1}"), fn.sum(f"time_{p+1}", f"a_{p+1}"))

            updated_graphs.append(graph)
        
        return updated_graphs


class SelfAttentionLayer(nn.Module):
    def __init__(self, dim, alpha=0.2, dropout=0.5):
        super(SelfAttentionLayer, self).__init__()
        self.in_dim = dim #2*dim
        self.out_dim = dim
        self.alpha = alpha
        self.dropout = dropout
        self.a = nn.Linear(self.in_dim, self.out_dim, bias=True)
        self.b = nn.Linear(self.out_dim, 1, bias=False)

    def forward(self, h):
        f = F.tanh(self.a(h))
        s = self.b(f).squeeze(dim=-1)

        # s[~mask] = -1e9
        attention = F.softmax(s, dim=-1)
        attention = attention.unsqueeze(dim=2)
        h_bar = h*attention
        h_bar = torch.sum(h_bar, dim=1)
        return h_bar