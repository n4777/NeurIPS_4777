import sys
sys.path.append("../")

import wandb

import torch
from torch.utils.data import DataLoader, TensorDataset

import torchvision
from torchvision import datasets
from torchvision import datasets, transforms

from collections import defaultdict, OrderedDict
import random
import numpy as np
from models.resnet import ResNet18
import copy
import os
from sklearn.cluster import DBSCAN

class Aggregator:
    def __init__(self, helper):
        self.helper = helper
        self.Wt = None
        self.krum_client_ids = []

    def agg(self, global_model, weight_accumulator, weight_accumulator_by_client, client_models, sampled_participants):
        if self.helper.config.agg_method == 'avg':
            return self.average_shrink_models(global_model, weight_accumulator)
        elif self.helper.config.agg_method == 'clip':
            self.clip_updates(weight_accumulator)
            return self.average_shrink_models(global_model, weight_accumulator)
        elif self.helper.config.agg_method in ['dp', 'crfl']:
            if self.helper.config.agg_method in ['crfl']:
                self.clip_updates(weight_accumulator)
            return self.dp(global_model, weight_accumulator)
        elif self.helper.config.agg_method == 'median':
            return self.median(global_model, weight_accumulator, weight_accumulator_by_client)
        elif self.helper.config.agg_method == 'rlr':
            return self.robust_learning_rate(global_model, weight_accumulator, weight_accumulator_by_client)
        elif self.helper.config.agg_method == 'bulyan':
            return self.bulyan(global_model, weight_accumulator, weight_accumulator_by_client)
        elif self.helper.config.agg_method == 'krumm':
            return self.krumm(global_model, weight_accumulator, weight_accumulator_by_client, sampled_participants)
        elif self.helper.config.agg_method == 'deepsight':
            return self.deepsight(global_model, weight_accumulator, weight_accumulator_by_client)
        elif self.helper.config.agg_method == 'feddf':
            return self.feddf(global_model, weight_accumulator, weight_accumulator_by_client, client_models, sampled_participants)
        elif self.helper.config.agg_method == 'fedrad':
            return self.fedrad(global_model, weight_accumulator, weight_accumulator_by_client, client_models, sampled_participants)
        elif self.helper.config.agg_method == 'sparsefed':
            return self.sparsefed(global_model, weight_accumulator, weight_accumulator_by_client, client_models, sampled_participants)
        else:
            raise NotImplementedError


    def average_shrink_models(self,  global_model, weight_accumulator):
        """
        Perform FedAvg algorithm and perform some clustering on top of it.
        """
        lr = 1

        for name, data in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            update_per_layer = weight_accumulator[name] * \
                               (1/self.helper.config.num_sampled_participants) * lr
            update_per_layer = torch.tensor(update_per_layer,dtype=data.dtype)
            data.add_(update_per_layer.cuda())

        return True

    def dp(self,  global_model, weight_accumulator):
        """
        Perform FedAvg algorithm and perform some clustering on top of it.
        """
        lr = 1

        for name, data in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            update_per_layer = weight_accumulator[name] * \
                               (1/self.helper.config.num_sampled_participants) * lr
            update_per_layer = torch.tensor(update_per_layer,dtype=data.dtype)
            data.add_(update_per_layer.cuda())
            if 'conv' in name:
                data.add_(torch.randn_like(data)*self.helper.config.crfl_sigma)

        return True 
    
    def median(self, global_model, weight_accumulator, weight_accumulator_by_client):
        final_wa = {}
        for name in weight_accumulator.keys():
            wa_by_c = []
            for i in range(len(weight_accumulator_by_client)):
                wa_by_c.append(weight_accumulator_by_client[i][name])
            wa_by_c = torch.stack(wa_by_c)
            final_wa[name],_ = torch.median(wa_by_c, 0)

        lr = 1

        for name, data in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            update_per_layer = final_wa[name] * lr
            update_per_layer = torch.tensor(update_per_layer,dtype=data.dtype)
            data.add_(update_per_layer.cuda())

        return True
        

    def compute_pairwise_distance(self, updates):
        def pairwise(u1, u2):
            ks = u1.keys()
            dist = 0
            for k in ks:
                if 'tracked' in k:
                    continue
                d = u1[k] - u2[k]
                dist = dist + torch.sum(d * d)
            return round(float(torch.sqrt(dist)), 2)

        scores = [0 for u in range(len(updates))]
        for i in range(len(updates)):
            for j in range(i + 1, len(updates)):
                dist = pairwise(updates[i], updates[j])
                scores[i] = scores[i] + dist
                scores[j] = scores[j] + dist
        return scores

    def krumm(self, global_model, weight_accumulator, weight_accumulator_by_client, sampled_participants):
        self.krum_client_ids.append([])
        n_mal = 4
        original_params = global_model.state_dict()

        # collect client updates
        updates = weight_accumulator_by_client

        temp_ids = list(range(self.helper.config.num_sampled_participants))

        krum_updates = list()
        n_ex = 2 * n_mal
        # print("Bulyan Stage 1：", len(updates))
        for i in range(1):
            scores = self.compute_pairwise_distance(updates)
            n_update = len(updates)
            threshold = sorted(scores)[0]
            for k in range(n_update - 1, -1, -1):
                if scores[k] <= threshold:
                    print("client {} is chosen:".format(temp_ids[k], round(scores[k], 2)))
                    self.krum_client_ids[-1].append(sampled_participants[k])
                    krum_updates.append(updates[k])
                    del updates[k]
                    del temp_ids[k]

        for name, data in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            update_per_layer = None
            for idx in range(len(krum_updates)):
                update_per_layer = krum_updates[idx][name] if update_per_layer == None else (update_per_layer + krum_updates[idx][name])
            update_per_layer = update_per_layer * (1/len(krum_updates))
            update_per_layer = torch.tensor(update_per_layer,dtype=data.dtype)
            data.add_(update_per_layer.cuda())

    
    def bulyan(self, global_model, weight_accumulator, weight_accumulator_by_client):
        n_mal = 2
        original_params = global_model.state_dict()

        # collect client updates
        updates = weight_accumulator_by_client

        temp_ids = list(range(self.helper.config.num_sampled_participants))

        krum_updates = list()
        n_ex = 2 * n_mal
        # print("Bulyan Stage 1：", len(updates))
        for i in range(self.helper.config.num_sampled_participants-n_ex):
            scores = self.compute_pairwise_distance(updates)
            n_update = len(updates)
            threshold = sorted(scores)[0]
            for k in range(n_update - 1, -1, -1):
                if scores[k] == threshold:
                    print("client {} is chosen:".format(temp_ids[k], round(scores[k], 2)))
                    krum_updates.append(updates[k])
                    del updates[k]
                    del temp_ids[k]
                    
        # print("Bulyan Stage 2：", len(krum_updates))    
        bulyan_update = OrderedDict()
        layers = krum_updates[0].keys()
        for layer in layers:
            bulyan_layer = None
            for update in krum_updates:
                bulyan_layer = update[layer][None, ...] if bulyan_layer is None else torch.cat(
                    (bulyan_layer, update[layer][None, ...]), 0)

            med, _ = torch.median(bulyan_layer, 0)
            _, idxs = torch.sort(torch.abs(bulyan_layer - med), 0)
            bulyan_layer = torch.gather(bulyan_layer, 0, idxs[:-n_ex, ...])
            # print("bulyan_layer",bulyan_layer.size())
            # bulyan_update[layer] = torch.mean(bulyan_layer, 0)
            # print(bulyan_layer)
            if not 'tracked' in layer:
                bulyan_update[layer] = torch.mean(bulyan_layer, 0)
            else:
                bulyan_update[layer] = torch.mean(bulyan_layer*1.0, 0).long()
            original_params[layer] = original_params[layer] + bulyan_update[layer]

        global_model.load_state_dict(original_params)

    def clip_updates(self, agent_updates_dict):
        for key in agent_updates_dict:
            if 'num_batches_tracked' not in key:
                update = agent_updates_dict[key]
                l2_update = torch.norm(update, p=2) 
                update.div_(max(1, l2_update/self.helper.config.clip_factor))
        return

    def compute_robustLR(self, weight_accumulator_by_client):
        sm_of_signs = {}
        for key in weight_accumulator_by_client[0]:
            if 'bn' in key or 'running' in key:
                sm_of_signs[key] = 1
            else:
                signs = torch.stack([weight_accumulator_by_client[i][key].sign() for i in range(len(weight_accumulator_by_client))])
                sum_signs = torch.abs(torch.sum(signs, dim=0))
                sum_signs[sum_signs >= self.helper.config.rlr_threshold] = 1.0
                sum_signs[sum_signs != 1.0] = -1.0
                sm_of_signs[key] = sum_signs
                                      
        return sm_of_signs

    def average_robust_lr(self, global_model, weight_accumulator, lr_vector):
        lr_scaler = 1

        for name, data in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            update_per_layer = weight_accumulator[name] * \
                               (1/self.helper.config.num_sampled_participants) * lr_scaler * lr_vector[name]
            update_per_layer = torch.tensor(update_per_layer,dtype=data.dtype)
            data.add_(update_per_layer.cuda())

        return True

    def robust_learning_rate(self, global_model, weight_accumulator, weight_accumulator_by_client):
        lr_vector = self.compute_robustLR(weight_accumulator_by_client)
        self.average_robust_lr(global_model, weight_accumulator, lr_vector)

    def local_model(self, tmp_model, wa):
        sd = tmp_model.state_dict()
        for name in sd:
            sd[name] += wa[name]
        tmp_model.load_state_dict(sd)
        return tmp_model

    def deepsight(self, global_model, weight_accumulator, weight_accumulator_by_client):
        def ensemble_cluster(neups, ddifs, biases):
            biases = np.array([bias.cpu().numpy() for bias in biases])
            #neups = np.array([neup.cpu().numpy() for neup in neups])
            #ddifs = np.array([ddif.cpu().detach().numpy() for ddif in ddifs])
            N = len(neups)
            # use bias to conduct DBSCAM
            # biases= np.array(biases)
            cosine_labels = DBSCAN(min_samples=3,metric='cosine').fit(biases).labels_
            print("cosine_cluster:{}".format(cosine_labels))
            # neups=np.array(neups)
            neup_labels = DBSCAN(min_samples=3).fit(neups).labels_
            print("neup_cluster:{}".format(neup_labels))
            ddif_labels = DBSCAN(min_samples=3).fit(ddifs).labels_
            print("ddif_cluster:{}".format(ddif_labels))

            dists_from_cluster = np.zeros((N, N))
            for i in range(N):
                for j in range(i, N):
                    dists_from_cluster[i, j] = (int(cosine_labels[i] == cosine_labels[j]) + int(
                        neup_labels[i] == neup_labels[j]) + int(ddif_labels[i] == ddif_labels[j]))/3.0
                    dists_from_cluster[j, i] = dists_from_cluster[i, j]
                    
            print("dists_from_clusters:")
            print(dists_from_cluster)
            ensembled_labels = DBSCAN(min_samples=3,metric='precomputed').fit(dists_from_cluster).labels_

            return ensembled_labels

        global_weight = list(global_model.state_dict().values())[-2]
        global_bias = list(global_model.state_dict().values())[-1]


        biases = [list(wa.values())[-1] for wa in weight_accumulator_by_client]
        weights = [(list(wa.values())[-2]+global_weight) for wa in weight_accumulator_by_client]

        n_client = len(weight_accumulator_by_client)
        neups = list()
        n_exceeds = list()

        # calculate neups
        sC_nn2 = 0
        for i in range(n_client):
            C_nn = torch.sum(weights[i]-global_weight, dim=[1]) + biases[i]-global_bias
            # print("C_nn:",C_nn)
            C_nn2 = C_nn * C_nn
            neups.append(C_nn2)
            sC_nn2 += C_nn2
            
            C_max = torch.max(C_nn2).item()
            threshold = 0.01 * C_max if 0.01 > (1 / len(biases)) else 1 / len(biases) * C_max
            n_exceed = torch.sum(C_nn2 > threshold).item()
            n_exceeds.append(n_exceed)

        neups = np.array([(neup/sC_nn2).cpu().numpy() for neup in neups])
        print("n_exceeds:{}".format(n_exceeds))

        # 256 can be replaced with smaller value
        width = 64
        if self.helper.config.dataset == 'cifar10':
            width = 32
        rand_input = torch.randn((256, 3, width, width)).cuda()
        global_ddif = torch.mean(torch.softmax(global_model(rand_input), dim=1), dim=0)
        # print("global_ddif:{} {}".format(global_ddif.size(),global_ddif))
        tmp_model = copy.deepcopy(self.helper.global_model)
        client_ddifs = []
        for i in range(n_client):
            tmp_model.load_state_dict(global_model.state_dict())
            tmp_model = self.local_model(tmp_model, weight_accumulator_by_client[i])
            client_ddifs.append(torch.mean(torch.softmax(tmp_model(rand_input), dim=1), dim=0)/ global_ddif)
        client_ddifs = np.array([client_ddif.cpu().detach().numpy() for client_ddif in client_ddifs])
        # print("client_ddifs:{}".format(client_ddifs[0]))
        classification_boundary = np.median(np.array(n_exceeds)) / 2
        
        identified_mals = [int(n_exceed <= classification_boundary) for n_exceed in n_exceeds]
        print("identified_mals:{}".format(identified_mals))
        clusters = ensemble_cluster(neups, client_ddifs, biases)
        print("ensemble clusters:{}".format(clusters))
        cluster_ids = np.unique(clusters)

        deleted_cluster_ids = list()
        for cluster_id in cluster_ids:
            n_mal = 0
            cluster_size = np.sum(cluster_id == clusters)
            for identified_mal, cluster in zip(identified_mals, clusters):
                if cluster == cluster_id and identified_mal:
                    n_mal += 1
            print("cluser size:{} n_mal:{}".format(cluster_size,n_mal))        
            if (n_mal / cluster_size) >= (1 / 3):
                deleted_cluster_ids.append(cluster_id)
        
        temp_chosen_ids = list(range(n_client))
        chosen_ids = list(range(n_client))
        for i in range(len(weight_accumulator_by_client)-1, -1, -1):
            # print("cluster tag:",clusters[i])
            if clusters[i] in deleted_cluster_ids:
                del chosen_ids[i]

        print("final clients length:{}".format(len(chosen_ids)))
        if len(chosen_ids)==0:
            chosen_ids = temp_chosen_ids
        
        new_was = []
        for i in chosen_ids:
            new_was.append(weight_accumulator_by_client[i])

        for name in weight_accumulator:
            weight_accumulator[name] = weight_accumulator_by_client[chosen_ids[0]][name]
            for i in range(1, len(chosen_ids)):
                weight_accumulator[name] += weight_accumulator_by_client[chosen_ids[i]][name]
        
        lr = 1

        for name, data in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            update_per_layer = weight_accumulator[name] * \
                               (1/self.helper.config.num_sampled_participants) * lr
            update_per_layer = torch.tensor(update_per_layer,dtype=data.dtype)
            data.add_(update_per_layer.cuda())

    """
    FedDF
    """

    def get_avg_logits(self, inputs, client_models, sampled_participants):
        with torch.no_grad():
            total_logits = None
            for id in sampled_participants:
                m = client_models[id]
                m.eval()
                logit = m(inputs)
                total_logits = logit if total_logits == None else total_logits+logit
            avg_logit = total_logits / len(sampled_participants)
        return avg_logit


    def feddf(self, global_model, weight_accumulator, weight_accumulator_by_client, client_models, sampled_participants):
        self.average_shrink_models(global_model, weight_accumulator)
        optimizer = torch.optim.SGD(global_model.parameters(), lr=0.002,
            momentum=self.helper.config.momentum,
            weight_decay=self.helper.config.decay)
        global_model.train()
        for inputs, labels in self.helper.train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            avg_logits = self.get_avg_logits(inputs, client_models, sampled_participants)
            predicted_logits = global_model(inputs)
            kl_div_loss = torch.nn.KLDivLoss(reduction='batchmean')(predicted_logits.softmax(dim=-1).log(), avg_logits.softmax(dim=-1))
            optimizer.zero_grad()
            kl_div_loss.backward()
            optimizer.step()

    """
    FedRAD
    """

    def get_median_logits(self, inputs, client_models, sampled_participants):
        with torch.no_grad():
            all_logits = None
            for id in sampled_participants:
                m = client_models[id]
                m.eval()
                logit = m(inputs)
                all_logits = logit[None, ...] if all_logits is None else torch.cat((all_logits, logit[None, ...]),dim=0)
            median_logit, _ = torch.median(all_logits, dim=0)
        return median_logit


    def fedrad(self, global_model, weight_accumulator, weight_accumulator_by_client, client_models, sampled_participants):
        self.average_shrink_models(global_model, weight_accumulator)
        optimizer = torch.optim.SGD(global_model.parameters(), lr=0.002,
            momentum=self.helper.config.momentum,
            weight_decay=self.helper.config.decay)
        global_model.train()
        for inputs, labels in self.helper.train_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            avg_logits = self.get_median_logits(inputs, client_models, sampled_participants)
            predicted_logits = global_model(inputs)
            kl_div_loss = torch.nn.KLDivLoss(reduction='batchmean')(predicted_logits.softmax(dim=-1).log(), avg_logits.softmax(dim=-1))
            optimizer.zero_grad()
            kl_div_loss.backward()
            optimizer.step()

    """
    SparseFed
    """

    def sparsefed(self, global_model, weight_accumulator, weight_accumulator_by_client, client_models, sampled_participants):
        if self.Wt == None:
            self.Wt = copy.deepcopy(weight_accumulator)
        else:
            for name in weight_accumulator:
                self.Wt[name] += weight_accumulator[name]
        
        mask_grad_list = {}

        for name, parms in self.Wt.items():
            if 'conv' in name:
                gradients = parms.abs().view(-1)
                gradients_length = len(gradients)
                _, indices = torch.topk(-1*gradients, int(gradients_length*0.95))

                mask_flat = torch.zeros(gradients_length)
                mask_flat[indices.cpu()] = 1.0
                mask_grad_list[name] = mask_flat.reshape(parms.size()).cuda()

        for name, data in global_model.state_dict().items():
            if name == 'decoder.weight':
                continue
            if name in mask_grad_list:
                update_per_layer = self.Wt[name] * (1/self.helper.config.num_sampled_participants) * mask_grad_list[name]
                self.Wt[name] -= update_per_layer
            else:
                update_per_layer = weight_accumulator[name] * (1/self.helper.config.num_sampled_participants)
            update_per_layer = torch.tensor(update_per_layer,dtype=data.dtype)
            data.add_(update_per_layer.cuda())