import sys
sys.path.append("../")
import time
import wandb

import torch
from torch.utils.data import DataLoader, TensorDataset

import torchvision
from torchvision import datasets
from torchvision import datasets, transforms

from collections import defaultdict, OrderedDict
import random
import numpy as np
from models.resnet import ResNet18, layer2module
from .extractor import FeatureExtractor
import copy
import os
import math

class Attacker:
    def __init__(self, helper):
        self.helper = helper
        self.previous_global_model = None
        self.setup()

    def setup(self):
        
        self.history_model_sd = {'sd':None, 'epoch': 0}
        self.handcraft_rnds = 0
        if self.helper.config.dataset == 'TinyImagenet':
            self.trigger = torch.ones((1,3,64,64), requires_grad=False, device = 'cuda')*0.5
        else:
            self.trigger = torch.ones((1,3,32,32), requires_grad=False, device = 'cuda')*0.5
        if self.helper.config.attacker_method == 'f3ba':
            if self.helper.config.dataset == 'cifar10':
                means = (0.4914, 0.4822, 0.4465)
                lvars = (0.2023, 0.1994, 0.201)
            elif self.helper.config.dataset == 'gtsrb':
                means = (0.3337, 0.3064, 0.3171)
                lvars = (0.2672, 0.2564, 0.2629)
            elif self.helper.config.dataset == 'TinyImagenet':
                means = (0.485, 0.456, 0.406)
                lvars = (0.229, 0.224, 0.225)
            else:
                raise NotImplementedError
                
            self.normalize = transforms.Normalize(means, lvars)
            with torch.no_grad():
                self.trigger = (self.trigger * 255).floor() / 255
                self.trigger = self.normalize(self.trigger)
        self.mask = torch.zeros_like(self.trigger)
        self.mask[:, :, 2:2+self.helper.config.trigger_size, 2:2+self.helper.config.trigger_size] = 1
        self.mask = self.mask.cuda()
        print(f'attacker_method {self.helper.config.attacker_method} ')
        if self.helper.config.attacker_method in ['bi', 'f3ba', 'cerp'] or 'sin' in self.helper.config.attacker_method:
            print("Will optimize the trigger, no need to init")
        elif self.helper.config.attacker_method in ['baseline', 'neurotoxin', 'dba', 'scale']:
            self.init_badnets_trigger()
        elif self.helper.config.attacker_method in ['dbaa']:
            self.trigger[:,:] = 1
        else:
            raise NotImplementedError
        self.trigger0 = self.trigger.clone()

    def init_badnets_trigger(self):
        print('Setup baseline trigger pattern.')
        self.trigger[:, 0, :,:] = 1
        return
    
    def noise_model(self, model, adversary_id, epoch, dl):
        ce_loss = torch.nn.CrossEntropyLoss()
        cos_loss = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
        noise_model = copy.deepcopy(model)
        sd = noise_model.state_dict()
        for name in sd:
            if 'conv' in name:
                mean = torch.ones_like(sd[name])
                # if self.history_model_sd['sd'] != None:
                #     g = (sd[name]-self.history_model_sd['sd'][name]) / (epoch - self.history_model_sd['epoch'])
                #     mean += self.helper.config.gaussian_bias_factor*g
                perturb_factor = (torch.randn_like(sd[name])*self.helper.config.wp_factor+mean)
                sd[name] = sd[name]*perturb_factor
        noise_model.load_state_dict(sd)
        noise_model.train()
        model.train()
        model.zero_grad()
        noise_model.zero_grad()
        for inputs, labels in dl:
            inputs, labels = inputs.cuda(), labels.cuda()
            r_out = model(inputs)
            r_loss = ce_loss(r_out, labels)
            n_out = noise_model(inputs)
            n_loss = ce_loss(n_out, labels)
            r_loss.backward()
            n_loss.backward()
        sim_sum = 0.
        sim_count = 0.
        for name in dict(noise_model.named_parameters()):
            if 'conv' in name:
                sim_count += 1
                sim_sum += cos_loss(dict(noise_model.named_parameters())[name].grad.reshape(-1),\
                                    dict(model.named_parameters())[name].grad.reshape(-1))
        return noise_model, sim_sum/sim_count
    
    def get_adv_model(self, model, dl, trigger, mask):
        adv_model = copy.deepcopy(model)
        adv_model.train()
        ce_loss = torch.nn.CrossEntropyLoss()
        adv_opt = torch.optim.SGD(adv_model.parameters(), lr = 0.01, momentum=0.9, weight_decay=5e-4)
        for _ in range(self.helper.config.dm_adv_epochs):
            for inputs, labels in dl:
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs = trigger*mask +(1-mask)*inputs
                outputs = adv_model(inputs)
                loss = ce_loss(outputs, labels)
                adv_opt.zero_grad()
                loss.backward()
                adv_opt.step()

        sim_sum = 0.
        sim_count = 0.
        cos_loss = torch.nn.CosineSimilarity(dim=0, eps=1e-08)
        for name in dict(adv_model.named_parameters()):
            if 'conv' in name:
                sim_count += 1
                sim_sum += cos_loss(dict(adv_model.named_parameters())[name].grad.reshape(-1),\
                                    dict(model.named_parameters())[name].grad.reshape(-1))
        return adv_model, sim_sum/sim_count

    def search_trigger(self, model, dl, type_, adversary_id = 0, epoch = 0):
        trigger_optim_time_start = time.time()
        adv_time = 0.
        K = 0
        model.eval()
        if 'gaussian' in self.helper.config.attacker_method:
            nms = []
            nm_weights = []
            for i in range(self.helper.config.noise_model_count):
                nm, nm_w = self.noise_model(model, adversary_id, epoch, dl)
                assert nm_w <= 1.
                nms.append(nm)
                nm_weights.append(nm_w)
            if self.history_model_sd['sd'] == None:
                self.history_model_sd['sd'] = model.state_dict()
            self.history_model_sd['epoch'] = epoch
        elif 'adv' in self.helper.config.attacker_method:
            adv_models = []
            adv_ws = []
        else:
            raise NotImplementedError

        def val_asr(model, dl, t, m):
            ce_loss = torch.nn.CrossEntropyLoss(label_smoothing = 0.001)
            correct = 0.
            num_data = 0.
            total_loss = 0.
            with torch.no_grad():
                for inputs, labels in dl:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    inputs = t*m +(1-m)*inputs
                    labels[:] = self.helper.config.target_class
                    output = model(inputs)
                    loss = ce_loss(output, labels)
                    total_loss += loss
                    pred = output.data.max(1)[1] 
                    correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
                    num_data += output.size(0)
            asr = correct/num_data
            return asr, total_loss
        
        ce_loss = torch.nn.CrossEntropyLoss()
        alpha = self.helper.config.trigger_lr
        
        if type_ == 'inner':
            K = self.helper.config.trigger_inner_epochs
        elif type_ == 'outter':
            K = self.helper.config.trigger_outter_epochs
        t = self.trigger.clone()
        m = self.mask.clone()
        def grad_norm(gradients):
            grad_norm = 0
            for grad in gradients:
                grad_norm += grad.detach().pow(2).sum()
            return grad_norm.sqrt()
        ga_loss_total = 0.
        normal_grad = 0.
        ga_grad = 0.
        count = 0
        trigger_optim = torch.optim.Adam([t], lr = alpha*10, weight_decay=0)
        for iter in range(K):
            if iter % 10 == 0:
                asr, loss = val_asr(model, dl, t, m)
                print(f"iter-{iter}, trigger asr {asr}, loss {loss:.3f}")
            if 'adv' in self.helper.config.attacker_method:
                if iter % self.helper.config.dm_adv_K == 0 and iter != 0:
                    if len(adv_models)>0:
                        for adv_model in adv_models:
                            del adv_model
                    adv_models = []
                    adv_ws = []
                    adv_time_start = time.time()
                    for _ in range(self.helper.config.dm_adv_model_count):
                        adv_model, adv_w = self.get_adv_model(model, dl, t,m) 
                        adv_models.append(adv_model)
                        adv_ws.append(adv_w)
                    adv_time_end = time.time()
                    adv_time += adv_time_end-adv_time_start
            

            for inputs, labels in dl:
                count += 1
                t.requires_grad_()
                inputs, labels = inputs.cuda(), labels.cuda()
                inputs = t*m +(1-m)*inputs
                labels[:] = self.helper.config.target_class
                if self.helper.config.attacker_method != 'sin-adv-only':
                    outputs = model(inputs) 
                    loss = ce_loss(outputs, labels)
                else:
                    loss = None
                
                if 'gaussian' in self.helper.config.attacker_method:
                    for nm_idx in range(len(nms)):
                        nm_w = nm_weights[nm_idx]
                        outputs = nms[nm_idx](inputs)
                        nm_loss = ce_loss(outputs, labels)
                        loss += self.helper.config.noise_loss_lambda*nm_w*nm_loss/self.helper.config.noise_model_count
                elif 'adv' in self.helper.config.attacker_method:
                    if len(adv_models) > 0:
                        for am_idx in range(len(adv_models)):
                            adv_model = adv_models[am_idx]
                            adv_w = adv_ws[am_idx]
                            outputs = adv_model(inputs)
                            nm_loss = ce_loss(outputs, labels)
                            if loss == None:
                                loss = self.helper.config.noise_loss_lambda*adv_w*nm_loss/self.helper.config.dm_adv_model_count
                            else:
                                loss += self.helper.config.noise_loss_lambda*adv_w*nm_loss/self.helper.config.dm_adv_model_count
                else:
                    raise NotImplementedError
                if loss != None:
                    loss.backward()
                    normal_grad += t.grad.sum()
                    new_t = t - alpha*t.grad.sign()
                    t = new_t.detach_()
                    if self.helper.config.dataset == 'gtsrb':
                        t = torch.clamp(t, min = -1.4, max = 2.8)
                    elif self.helper.config.dataset == 'cifar10':
                        t = torch.clamp(t, min = -2, max = 2)
                    elif self.helper.config.dataset == 'TinyImagenet':
                        t = torch.clamp(t, min = -2, max = 2)
                    else:
                        raise NotImplementedError
                    t.requires_grad_()
        if self.helper.config.gradient_alignment:
            print(f'ga_loss {ga_loss_total/count:.3f} | normal_grad {normal_grad/count:.3f} | ga_grad {ga_grad/count:.3f}')
        t = t.detach()
        self.trigger = t
        self.mask = m
        trigger_optim_time_end = time.time()
            

    def poison_input(self, inputs, labels, eval=False):
        if eval:
            bkd_num = inputs.shape[0]
            if self.helper.config.attacker_method == 'dbaa':
                self.set_dbaa_mask(0, evaling = True)
        else:
            bkd_num = int(self.helper.config.bkd_ratio * inputs.shape[0])
        if self.helper.config.attacker_method == 'scale':
            bkd_num = math.ceil(bkd_num/50)
        inputs[:bkd_num] = self.trigger*self.mask + inputs[:bkd_num]*(1-self.mask)
        labels[:bkd_num] = self.helper.config.target_class
        return inputs, labels
    
    def set_dba_mask(self, dba_rnd):
        dba = dba_rnd % 4
        split_length = int(self.helper.config.trigger_size/2)
        self.mask *= 0
        if dba == 0:
            self.mask[:, :, 2:2+split_length, 2:2+split_length] = 1
        elif dba == 1:
            self.mask[:, :, 2:2+split_length, 2+split_length:2+self.helper.config.trigger_size] = 1
        elif dba == 2:
            self.mask[:, :, 2+split_length:2+self.helper.config.trigger_size, 2:2+split_length] = 1
        elif dba == 3:
            self.mask[:, :, 2+split_length:2+self.helper.config.trigger_size, 2+split_length:2+self.helper.config.trigger_size] = 1
        
    def set_dbaa_mask(self, dba_rnd, evaling = False):
        dba = dba_rnd % 4
        split_length = int(self.helper.config.trigger_size/2)
        self.mask *= 0
        if evaling:
            self.mask[:, :, 0:1, 0:4] = 1
            self.mask[:, :, 0:1, 10:14] = 1
            self.mask[:, :, 6:7, 0:4] = 1
            self.mask[:, :, 6:7, 10:14] = 1
        elif dba == 0:
            self.mask[:, :, 0:1, 0:4] = 1
        elif dba == 1:
            self.mask[:, :, 0:1, 10:14] = 1
        elif dba == 2:
            self.mask[:, :, 6:7, 0:4] = 1
        elif dba == 3:
            self.mask[:, :, 6:7, 10:14] = 1

    def apply_grad_mask(self, model, mask_grad_list):
        '''
        for Neurotoxin attacker only
        '''
        assert self.helper.config.attacker_method == 'neurotoxin'
        mask_grad_list_copy = iter(mask_grad_list)
        for name, parms in model.named_parameters():
            if parms.requires_grad:
                parms.grad = parms.grad * next(mask_grad_list_copy)

    
    def grad_mask(self, model, dl, criterion, ratio = 0.95):
        '''
        for Neurotoxin attacker only
        '''
        assert self.helper.config.attacker_method == 'neurotoxin'
        model.train()
        model.zero_grad()
        for _ in range(30):
            for inputs, labels in dl:
                inputs, labels = inputs.cuda(), labels.cuda()

                output = model(inputs)

                loss = criterion(output, labels)
                loss.backward(retain_graph=True)

        mask_grad_list = []
        grad_abs_percentage_list = []
        grad_res = []
        l2_norm_list = []
        sum_grad_layer = 0.0
        for _, parms in model.named_parameters():
            if parms.requires_grad:
                grad_res.append(parms.grad.view(-1))
                l2_norm_l = torch.norm(parms.grad.view(-1).clone().detach().cuda())/float(len(parms.grad.view(-1)))
                l2_norm_list.append(l2_norm_l)
                sum_grad_layer += l2_norm_l.item()

        grad_flat = torch.cat(grad_res)

        percentage_mask_list = []
        k_layer = 0
        for _, parms in model.named_parameters():
            if parms.requires_grad:
                gradients = parms.grad.abs().view(-1)
                gradients_length = len(gradients)
                if ratio == 1.0:
                    _, indices = torch.topk(-1*gradients, int(gradients_length*1.0))
                else:

                    ratio_tmp = 1 - l2_norm_list[k_layer].item() / sum_grad_layer
                    _, indices = torch.topk(-1*gradients, int(gradients_length*ratio))

                mask_flat = torch.zeros(gradients_length)
                mask_flat[indices.cpu()] = 1.0
                mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())

                percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0

                percentage_mask_list.append(percentage_mask1)


                k_layer += 1
        model.zero_grad()
        return mask_grad_list

    """
    F3BA
    """
    def get_conv_weight_names(self, model):
        conv_targets = list()
        weights = model.state_dict()
        for k in weights.keys():
            if 'conv' in k and 'weight' in k:
                conv_targets.append(k)

        return conv_targets
    
    def conv_activation(self, model, layer_name, loader, attack):
        extractor = FeatureExtractor(model)
        hook = extractor.insert_activation_hook(model)
        module = layer2module(model, layer_name)
        conv_activations = None
        for inputs, labels in loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = self.poison_input(inputs, labels,True)
            _ = model(inputs)
            conv_activation = extractor.activations(model, module)
            conv_activation = torch.mean(conv_activation, [0])
            conv_activations = conv_activation if conv_activations is None else conv_activations + conv_activation

        avg_activation = conv_activations / len(loader)
        extractor.release_hooks()
        torch.cuda.empty_cache()
        return avg_activation
    
    def fc_activation(self, model, layer_name, loader, attack):
        extractor = FeatureExtractor(model)
        hook = extractor.insert_activation_hook(model)
        module = layer2module(model, layer_name)
        neuron_activations = None
        for inputs, labels in loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            inputs, labels = self.poison_input(inputs, labels,True)
            _ = model(inputs)
            neuron_activation = extractor.activations(model, module)
            neuron_activations = neuron_activation if neuron_activations is None else neuron_activations + neuron_activation

        avg_activation = neuron_activations / len(loader)
        extractor.release_hooks()
        torch.cuda.empty_cache()
        return avg_activation
    
    def get_neuron_weight_names(self, model):
        neuron_targets = list()
        weights = model.state_dict()
        for k in weights.keys():
            if 'fc' in k and 'weight' in k:
                neuron_targets.append(k)

        return neuron_targets

    def inject_handcrafted_neurons(self, model, candidate_weights, diff, loader):
        handcrafted_connectvites = defaultdict(list)
        target_label = self.helper.config.target_class
        n_labels = 10
        if self.helper.config.dataset == 'TinyImagenet':
            n_labels = 200
        fc_names = self.get_neuron_weight_names(model)
        fc_diff = diff
        last_layer, last_ids = None, list()
        for layer_name, connectives in candidate_weights.items():
            if layer_name not in fc_names:
                continue
            raw_model = copy.deepcopy(model)
            model_weights = model.state_dict()
            ideal_signs = torch.sign(fc_diff)
            n_next_neurons = connectives.size()[0]
            # last_layer
            if n_next_neurons == n_labels:
                break

            ideal_signs = ideal_signs.repeat(n_next_neurons, 1) * connectives
            # count the flip num
            n_flip = torch.sum(((ideal_signs * torch.sign(model_weights[layer_name]) * connectives == -1).int()))
            print("n_flip in {}:{}".format(layer_name, n_flip))
            model_weights[layer_name] = (1 - connectives) * model_weights[layer_name] + torch.abs(
                connectives * model_weights[layer_name]) * ideal_signs
            model.load_state_dict(model_weights)
            last_layer = layer_name
            fc_diff = self.fc_activation(model, layer_name, loader, attack=True).mean([0]) - self.fc_activation(
                model, layer_name, loader, attack=False).mean([0])

    def conv_features(self, model, loader, attack):
        features = None
        for inputs, labels in loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            if attack:
                inputs, lables = self.poison_input(inputs, labels, eval=True)
            feature = model.features(inputs).mean([0])
            #features = feature if features is None else features + feature
            features = feature.detach() if features is None else features + feature.detach()
            torch.cuda.empty_cache()
        avg_features = features / len(loader)

        return avg_features

    def inject_handcrafted_filters(self, model, candidate_weights, loader):
        conv_weight_names = self.get_conv_weight_names(model)
        difference = None
        for layer_name, conv_weights in candidate_weights.items():
            if layer_name not in conv_weight_names:
                continue
            model_weights = model.state_dict()
            n_filter = conv_weights.size()[0]
            for i in range(n_filter):
                conv_kernel = model_weights[layer_name][i, ...].clone().detach()
                handcrafted_conv_kernel = self.flip_filter_as_trigger(conv_kernel, difference)
                # handcrafted_conv_kernel = conv_kernel

                mask = conv_weights[i, ...]
                model_weights[layer_name][i, ...] = mask * handcrafted_conv_kernel + (1 - mask) * \
                                                    model_weights[layer_name][i, ...]
                # model_weights[layer_name][i, ...].mul_(1-mask)
                # model_weights[layer_name][i, ...].add_(mask * handcrafted_conv_kernel)

            model.load_state_dict(model_weights)
            difference = self.conv_activation(model, layer_name, loader, True) - \
                self.conv_activation(model,layer_name,loader,False)

            print("handcraft_conv: {}".format(layer_name))

        torch.cuda.empty_cache()
        if difference is not None:
            feature_difference = self.conv_features(model, loader, True) - \
                self.conv_features(model, loader,False)
            return feature_difference

    def handcraft(self, dl, model, participant_id):
        self.handcraft_rnds += 1
        model.eval()

        if self.previous_global_model is None:
            self.previous_global_model = copy.deepcopy(model)
            return
        candidate_weights = self.search_candidate_weights(model, proportion=0.1)
        self.previous_global_model = copy.deepcopy(model)

        print("Optimize Trigger:")
        self.optimize_backdoor_trigger(model, candidate_weights, dl)

        print("Inject Candidate Filters:")
        diff = self.inject_handcrafted_filters(model, candidate_weights, dl)
        if diff is not None and self.handcraft_rnds % 3 == 1:
            print("Rnd {}: Inject Backdoor FC".format(self.handcraft_rnds))
            self.inject_handcrafted_neurons(model, candidate_weights, diff, dl)


    def search_candidate_weights(self, model, proportion=0.2):
        assert self.helper.config.attacker_method == 'f3ba'
        kernel_selection = 'movement'
        candidate_weights = OrderedDict()
        model_weights = model.state_dict()

        n_labels = 0

        if kernel_selection == "movement":
            history_weights = self.previous_global_model.state_dict()
            for layer in history_weights.keys():
                if 'conv' in layer:
                    proportion = 0.02
                elif 'fc' in layer:
                    proportion = 0.001

                candidate_weights[layer] = (model_weights[layer] - history_weights[layer]) * model_weights[layer]
                n_weight = candidate_weights[layer].numel()
                theta = torch.sort(candidate_weights[layer].flatten(), descending=False)[0][int(n_weight * proportion)]
                candidate_weights[layer][candidate_weights[layer] < theta] = 1
                candidate_weights[layer][candidate_weights[layer] != 1] = 0

        return candidate_weights

    def trigger_loss(self, model,backdoor_inputs, clean_inputs, pattern, grads=True):
        model.train()
        backdoor_activations = model.first_activations(backdoor_inputs).mean([0, 1])
        clean_activations = model.first_activations(clean_inputs).mean([0, 1])
        difference = backdoor_activations - clean_activations
        loss = torch.sum(difference * difference)
        
        if grads:
            grads = torch.autograd.grad(loss, pattern, retain_graph=True)

        return loss, grads
    
    def flip_filter_as_trigger(self, conv_kernel: torch.Tensor, difference):
        flip_factor = 1
        c_min, c_max = conv_kernel.min(), conv_kernel.max()
        pattern = None
        if difference is None:
            pattern_layers = self.trigger
            x_top, y_top = 2,2
            x_bot, y_bot = 2+self.helper.config.trigger_size, 2+self.helper.config.trigger_size
            pattern = pattern_layers[:, x_top:x_bot, y_top:y_bot]
        else:
            pattern = difference
        w = conv_kernel[0, ...].size()[0]
        resize = transforms.Resize((w, w))
        pattern = resize(pattern)
        p_min, p_max = pattern.min(), pattern.max()
        scaled_pattern = (pattern - p_min) / (p_max - p_min) * (c_max - c_min) + c_min

        crop_mask = torch.sign(scaled_pattern) != torch.sign(conv_kernel)
        conv_kernel = torch.sign(scaled_pattern) * torch.abs(conv_kernel)
        conv_kernel[crop_mask] = conv_kernel[crop_mask] * flip_factor
        return conv_kernel

    def set_handcrafted_filters2(self, model, candidate_weights, layer_name):
        conv_weights = candidate_weights[layer_name]
        # print("check candidate:",int(torch.sum(conv_weights)))
        model_weights = model.state_dict()
        temp_weights = copy.deepcopy(model_weights[layer_name])

        n_filter = conv_weights.size()[0]

        for i in range(n_filter):
            conv_kernel = model_weights[layer_name][i, ...].clone().detach()
            handcrafted_conv_kernel = self.flip_filter_as_trigger(conv_kernel, difference=None)
            mask = conv_weights[i, ...]
            model_weights[layer_name][i, ...] = mask * handcrafted_conv_kernel + (1 - mask) * model_weights[layer_name][
                i, ...]

        model.load_state_dict(model_weights)

    def optimize_backdoor_trigger(self, model, candidate_weights, loader):
        pattern, mask = self.trigger, self.mask
        pattern.requires_grad = True

        x_top, y_top = 2, 2
        x_bot, y_bot = x_top+self.helper.config.trigger_size, y_top+self.helper.config.trigger_size
        if self.helper.config.dataset == 'cifar10':
            means = (0.4914, 0.4822, 0.4465)
            lvars = (0.2023, 0.1994, 0.201)
        elif self.helper.config.dataset == 'gtsrb':
            means = (0.3337, 0.3064, 0.3171)
            lvars = (0.2672, 0.2564, 0.2629)
        elif self.helper.config.dataset == 'TinyImagenet':
            means = (0.485, 0.456, 0.406)
            lvars = (0.229, 0.224, 0.225)
        cbots, ctops = list(), list()
        for h in range(pattern.size()[1]):
            cbot = (0 - means[h]) / lvars[h]
            ctop = (1 - means[h]) / lvars[h]
            cbots.append(round(cbot, 2))
            ctops.append(round(ctop, 2))

        raw_weights = copy.deepcopy(model.state_dict())
        self.set_handcrafted_filters2(model, candidate_weights, "conv1.weight")
        for epoch in range(2):
            losses = list()
            for inputs, labels in loader:

                backdoor_batch = inputs.cuda()
                clean_batch = inputs.cuda()
                backdoor_labels = labels.cuda()
                clean_labels = labels.cuda()

                backdoor_batch[:] = (1 - mask) * backdoor_batch[:] + mask * pattern
                backdoor_labels[:].fill_(self.helper.config.target_class)

                self.set_handcrafted_filters2(model, candidate_weights, "conv1.weight")

                # loss, grads = trigger_attention_loss(raw_model, model, backdoor_batch.inputs, pattern, grads=True)
                loss, grads = self.trigger_loss(model, backdoor_batch, clean_batch, pattern, grads=True)
                losses.append(loss.item())

                pattern = pattern + grads[0] * 0.1

                n_channel = pattern.size()[1]
                for h in range(n_channel):
                    pattern[:,h, x_top:x_bot, y_top:y_bot] = torch.clamp(pattern[:,h, x_top:x_bot, y_top:y_bot], cbots[h],
                                                                       ctops[h], out=None)

                model.zero_grad()
            print("epoch:{} trigger loss:{}".format(epoch, np.mean(losses)))

        print(pattern[0, 0, x_top:x_bot, y_top:y_bot].cpu().data)

        self.trigger = pattern.clone().detach()
        self.pattern_tensor = pattern[x_top:x_bot, y_top:y_bot]

        model.load_state_dict(raw_weights)
        torch.cuda.empty_cache()

    def model_similarity_loss(self,global_model, local_model, type_ = 'L2'):
        global_model.switch_grads(False)
        global_weights=global_model.state_dict()
        layers = global_weights.keys()
        loss = 0
        cossim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
        for layer, param in local_model.named_parameters():
            if 'tracked' in layer or 'running' in layer:
                continue
            if type_ == 'L2':
                layer_dist = global_weights[layer]-param
                loss = loss + torch.sum(layer_dist*layer_dist)
            elif type_ == 'cs':
                loss += -cossim(global_weights[layer].view(-1), param.view(-1))
        return loss
    
    def cerp_optim(self, dl, model, lr, criterion):
        pattern, mask = self.trigger, self.mask
        pattern.requires_grad = True
        trigger_optim = torch.optim.SGD([pattern], lr=0.005,
            momentum=self.helper.config.momentum,
            weight_decay=self.helper.config.decay)
        for internal_epoch in range(self.helper.config.attacker_retrain_times*20):
            for inputs, labels in dl:
                inputs, labels = inputs.cuda(), labels.cuda()
                bkd_num = int(inputs.shape[0])
                inputs[:bkd_num] = pattern*mask + inputs[:bkd_num]*(1-mask)
                labels[:bkd_num] = self.helper.config.target_class
                output = model(inputs)
                loss = criterion(output, labels) + 0.0001*torch.norm(pattern-self.trigger0)
                trigger_optim.zero_grad()
                loss.backward(retain_graph=True)
                trigger_optim.step()
        scale = torch.max(torch.norm(pattern-self.trigger0)/3,torch.Tensor(1).cuda())
        pattern = self.trigger0 + (pattern-self.trigger0)/scale
        self.trigger = pattern.clone().detach()
    
        