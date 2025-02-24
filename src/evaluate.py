from queue import PriorityQueue
import torch
import tqdm
import math

def find_neighbors(graph, head, path=[]):
    neighbors = set()
    
    for triple in graph:
        h, t, _ = triple
        if h == head and t not in path:
            neighbors.add(t)
    
    return sorted(neighbors)


def find_max_score_path(graph, start_node, target_node):
    open_set = PriorityQueue()
    open_set.put((0, start_node))
    scores = {node: -float('inf') for node in graph.nodes}
    scores[start_node] = 0
    predecessors = {}

    while not open_set.empty():
        current_score, current_node = open_set.get()

        if current_node == target_node:
            # Backtrack to find the optimal path
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = predecessors.get(current_node)
            path.reverse()
            return path

        for neighbor in graph.neighbors(current_node):
            edge_score = graph.edge_score(current_node, neighbor)
            total_score = scores[current_node] + edge_score + graph.node_score(neighbor)
            
            if total_score > scores[neighbor]:
                scores[neighbor] = total_score
                predecessors[neighbor] = current_node
                priority = total_score + heuristic(neighbor, target_node)
                open_set.put((priority, neighbor))

    # If no path is found
    return None


def evaluate(data_loader, model, path_len, path_topk):
    topk = [5] * path_len
    final_path_indices = []
    final_path_list = []
    target_indices = []
    final_path_score_list = []
    final_path_score_sum_list = []
    with torch.no_grad():
        correct, incorrect = 0, 0
        model.eval()
        for batch in tqdm.tqdm(data_loader, total=len(data_loader)):
            graphs, seed_entities, demographic_embedding, opinion_embedding, indexed_candidate_nodes, indexed_target_node, triples, nodeId2node, questions, edge_id_to_hts, ht_to_edge_ids = batch
            graphs = model(graphs, seed_entities, demographic_embedding, opinion_embedding, questions)

            bsz = len(graphs)
            for b in range(bsz):
                graph = graphs[b]
                edge_id_to_ht = edge_id_to_hts[b]
                ht_to_edge_id = ht_to_edge_ids[b]
                
                path_pool = []  # list of list, size=bs
                probs_pool = []
                final_path_pool = []
                final_probs_pool = []
                for hop in range(path_len):
                    if hop==0:
                        k = topk[hop]
                        probs = graph.ndata["a_0"].to("cpu")
                        probs = probs.squeeze()
                        
                        topk_probs, topk_actions = torch.topk(probs, k=k)
                        for j in range(k):
                            path_pool.append([topk_actions[j].item()])
                            probs_pool.append([topk_probs[j].item()])
                    else:
                        candidate_paths = []
                        candidate_path_probs = []
                        # print("len(path_pool):", len(path_pool))
                        for i in range(len(path_pool)):
                            path = path_pool[i]
                            prob = probs_pool[i]
                            last_entity = path[-1]

                            neighbors = list(find_neighbors(triples[b], last_entity, path))
                            if len(neighbors) == 0:
                                continue

                            probs = graph.ndata["a_"+str(hop)].to("cpu")
                            neighbor_scores = [probs[neighbor].item() for neighbor in neighbors]
                            assert len(neighbors) == len(neighbor_scores)

                            neighbor_score_pair = zip(neighbors, neighbor_scores)
                            neighbor_score_pair = sorted(neighbor_score_pair, key=lambda x:x[1], reverse=True)
                            neighbor_score_pair = zip(*neighbor_score_pair)
                            neighbor_score_pair = [list(a) for a in neighbor_score_pair]
                            
                            neighbors, neighbor_scores = neighbor_score_pair[0], neighbor_score_pair[1]
                            k = topk[hop]
                            topk_neighbors, topk_neighbor_scores = neighbors[:k], neighbor_scores[:k]

                            for neighbor_node, neighbor_node_score in zip(topk_neighbors, topk_neighbor_scores):
                                
                                if neighbor_node in path or neighbor_node_score == 0:
                                    continue

                                candidate_path = path + [neighbor_node]
                                candidate_path_prob = prob + [neighbor_node_score]
                                
                                if neighbor_node in indexed_candidate_nodes[b]:
                                    final_path_pool.append(candidate_path)
                                    final_probs_pool.append(candidate_path_prob)
                                else:
                                    candidate_paths.append(candidate_path)
                                    candidate_path_probs.append(candidate_path_prob)

                                assert len(candidate_paths) == len(candidate_path_probs)

                        path_pool = candidate_paths
                        probs_pool = candidate_path_probs

                path_score_sum_dict = dict()
                path_dict = dict()
                path_score_dict = dict()
                for path, probs in zip(final_path_pool, final_probs_pool):
                    if path[-1] not in path_score_sum_dict:
                        path_score_sum_dict[path[-1]] = []
                        path_dict[path[-1]] = []
                        path_score_dict[path[-1]] = []
                    prob = 1
                    for p in probs:
                        prob *= p
                    path_score_sum_dict[path[-1]].append(prob**(1/len(probs)))
                    path_dict[path[-1]].append(path)
                    path_score_dict[path[-1]].append(probs)

                final_path, final_score, final_path_num = -1, -1, -1
                for path_idx, scores in path_score_sum_dict.items():
                    # print()
                    # score = sum(scores) / len(scores)
                    sorted_scores = sorted(scores, reverse=True)[:path_topk]
                    score = sum(sorted_scores)
                    path_num = len(scores)
                    
                    if final_score < score:
                        final_path = path_idx
                        final_score = score
                        final_path_num = path_num

                if final_path != -1:
                    final_path_indices.append(nodeId2node[b][final_path])
                    paths = path_dict[final_path]
                    _paths = []
                    for path in paths:
                        _path = []
                        for p in path:
                            _path.append(nodeId2node[b][p])
                        _paths.append(_path)
                    final_path_list.append(_paths)
                    final_path_score_list.append(path_score_dict[final_path])
                    final_path_score_sum_list.append(path_score_sum_dict[final_path])
                else:
                    final_path_indices.append(final_path)
                    final_path_list.append([])
                    final_path_score_list.append([])
                    final_path_score_sum_list.append([])
                target_indices.append(nodeId2node[b][indexed_target_node[b]])
                if final_path == indexed_target_node[b]:
                    correct += 1
                else:
                    incorrect += 1
        accuracy = correct / (correct + incorrect)
        return accuracy, final_path_indices, final_path_list, final_path_score_list, final_path_score_sum_list, target_indices


def evaluate_edge(data_loader, model, path_len, path_topk):
    topk = [5] * path_len
    final_path_indices = []
    final_path_list = []
    target_indices = []
    final_path_score_list = []
    final_path_score_sum_list = []
    with torch.no_grad():
        correct, incorrect = 0, 0
        model.eval()
        for batch in tqdm.tqdm(data_loader, total=len(data_loader)):
            graphs, seed_entities, demographic_embedding, opinion_embedding, indexed_candidate_nodes, indexed_target_node, triples, nodeId2node, questions, edge_id_to_hts, ht_to_edge_ids = batch
            graphs = model(graphs, seed_entities, demographic_embedding, opinion_embedding, questions)

            bsz = len(graphs)
            for b in range(bsz):
                graph = graphs[b]
                edge_id_to_ht = edge_id_to_hts[b]
                ht_to_edge_id = ht_to_edge_ids[b]
                # print("seed_entities:", seed_entities)
                # print("edge_id_to_ht:", edge_id_to_ht)

                path_pool = []  # list of list, size=bs
                probs_pool = []
                final_path_pool = []
                final_probs_pool = []
                for hop in range(path_len):
                    if hop==0:
                        k = topk[hop]
                        probs = graph.ndata["a_0"].to("cpu")
                        probs = probs.squeeze()
                        
                        topk_probs, topk_actions = torch.topk(probs, k=k)
                        for j in range(k):
                            path_pool.append([topk_actions[j].item()])
                            probs_pool.append([topk_probs[j].item()])
                    else:
                        candidate_paths = []
                        candidate_path_probs = []
                        # print("len(path_pool):", len(path_pool))
                        for i in range(len(path_pool)):
                            path = path_pool[i]
                            prob = probs_pool[i]
                            last_entity = path[-1]

                            neighbors = list(find_neighbors(triples[b], last_entity, path))
                            if len(neighbors) == 0:
                                continue
                            
                            # print("transition_probs:", graph.edata["transition_probs_1"])
                            # assert False
                            probs = graph.ndata["a_"+str(hop)].to("cpu")
                            neighbor_scores = [probs[neighbor].item() for neighbor in neighbors]
                            assert len(neighbors) == len(neighbor_scores)

                            neighbor_score_pair = zip(neighbors, neighbor_scores)
                            neighbor_score_pair = sorted(neighbor_score_pair, key=lambda x:x[1], reverse=True)
                            neighbor_score_pair = zip(*neighbor_score_pair)
                            neighbor_score_pair = [list(a) for a in neighbor_score_pair]
                            
                            neighbors, neighbor_scores = neighbor_score_pair[0], neighbor_score_pair[1]
                            k = topk[hop]
                            topk_neighbors, topk_neighbor_scores = neighbors[:k], neighbor_scores[:k]

                            for neighbor_node, neighbor_node_score in zip(topk_neighbors, topk_neighbor_scores):
                                
                                if neighbor_node in path or neighbor_node_score == 0:
                                    continue

                                candidate_path = path + [neighbor_node]
                                candidate_path_prob = prob + [neighbor_node_score]
                                
                                if neighbor_node in indexed_candidate_nodes[b]:
                                    final_path_pool.append(candidate_path)
                                    final_probs_pool.append(candidate_path_prob)
                                else:
                                    candidate_paths.append(candidate_path)
                                    candidate_path_probs.append(candidate_path_prob)

                                assert len(candidate_paths) == len(candidate_path_probs)

                        path_pool = candidate_paths
                        probs_pool = candidate_path_probs

                path_score_sum_dict = dict()
                path_dict = dict()
                path_score_dict = dict()
                for path, probs in zip(final_path_pool, final_probs_pool):
                    if path[-1] not in path_score_sum_dict:
                        path_score_sum_dict[path[-1]] = []
                        path_dict[path[-1]] = []
                        path_score_dict[path[-1]] = []
                    prob = 1
                    for p in probs:
                        prob *= p
                    path_score_sum_dict[path[-1]].append(prob**(1/len(probs)))
                    path_dict[path[-1]].append(path)
                    path_score_dict[path[-1]].append(probs)

                # for path_idx, scores in path_score_sum_dict.items():
                #     max_score = max(scores)
                #     score_idx = scores.index(max_score)
                #     final_path = path_dict[path_idx][score_idx]
                #     # print("path_idx:", path_idx, "final_path:", final_path, "max_score:", max_score)
                # print("path_score_sum_dict:", path_score_sum_dict)
                # assert False
                final_path, final_score, final_path_num = -1, -1, -1
                for path_idx, scores in path_score_sum_dict.items():
                    # print()
                    # score = sum(scores) / len(scores)
                    sorted_scores = sorted(scores, reverse=True)[:path_topk]
                    # print("sorted_scores:", sorted_scores)
                    # assert False

                    # print(sorted_scores, sum(sorted_scores))
                    score = sum(sorted_scores)
                    path_num = len(scores)
                    
                    if final_score < score:
                        final_path = path_idx
                        final_score = score
                        final_path_num = path_num

                if final_path != -1:
                    final_path_indices.append(nodeId2node[b][final_path])
                    paths = path_dict[final_path]
                    _paths = []
                    for path in paths:
                        _path = []
                        for p in path:
                            _path.append(nodeId2node[b][p])
                        _paths.append(_path)
                    final_path_list.append(_paths)
                    final_path_score_list.append(path_score_dict[final_path])
                    final_path_score_sum_list.append(path_score_sum_dict[final_path])
                else:
                    final_path_indices.append(final_path)
                    final_path_list.append([])
                    final_path_score_list.append([])
                    final_path_score_sum_list.append([])
                target_indices.append(nodeId2node[b][indexed_target_node[b]])
                if final_path == indexed_target_node[b]:
                    correct += 1
                else:
                    incorrect += 1
        accuracy = correct / (correct + incorrect)
        return accuracy, final_path_indices, final_path_list, final_path_score_list, final_path_score_sum_list, target_indices