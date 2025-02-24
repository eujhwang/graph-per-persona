import torch



def train(batch, model, optimizer, path_len, loss_type, scheduler=None):
    graphs, seed_entities, demographic_embedding, opinion_embedding, indexed_candidate_nodes, indexed_target_node, triples, nodeId2node, questions, edge_id_to_ht, ht_to_edge_id = batch
    graphs = model(graphs, seed_entities, demographic_embedding, opinion_embedding, questions)
    
    epsilon = 1e-30
    scores = []
    bsz = len(graphs)
    # option = 1
    for b in range(bsz):
        if loss_type == "last":
            graph = graphs[b]
            
            node_time_scores = graph.ndata["a_"+str(path_len-1)].squeeze() + epsilon
            target_node = indexed_target_node[b]
            candidate_nodes = indexed_candidate_nodes[b]
            candidate_nodes = [n for n in candidate_nodes if target_node != n]
            
            assert node_time_scores[target_node] <= 1 and node_time_scores[target_node] >= 0
            score = -torch.log(node_time_scores[target_node])
        elif loss_type == "full":
            score = 0
            for pi in range(1, path_len):
                graph = graphs[b]
                
                node_time_scores = graph.ndata["a_"+str(pi)].squeeze() + epsilon
                target_node = indexed_target_node[b]
                candidate_nodes = indexed_candidate_nodes[b]
                candidate_nodes = [n for n in candidate_nodes if target_node != n]
                if node_time_scores[target_node].item() >= 1.0:
                    continue
                
                print("node_time_scores[target_node].item:", node_time_scores[target_node].item())
                if not (node_time_scores[target_node] < 1 and node_time_scores[target_node] >= 0):
                    print("node_time_scores[target_node]:", node_time_scores[target_node])
                    assert False
                assert node_time_scores[target_node] <= 1 and node_time_scores[target_node] >= 0
                score += -torch.log(node_time_scores[target_node])
            score = score / path_len
        else:
            raise Exception(f"Invalid loss type: {loss_type}")

        scores.append(score)
    loss = torch.stack(scores).sum(-1) / bsz

    # backward
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    return loss.item()