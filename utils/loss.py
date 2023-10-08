import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class prototype_NCEloss(nn.Module):
    def __init__(self):
        super(prototype_NCEloss, self).__init__()

    def get_protos(self, q, index, cluster_result):
        # prototypical contrast
        if cluster_result is not None:
            proto_labels = []
            proto_logits = []
            proto_selecteds = []
            temp_protos = []
            for n, (im2cluster, prototypes, density) in enumerate(
                    zip(cluster_result['im2cluster'], cluster_result['centroids'], cluster_result['density'])):

                pos_proto_id = im2cluster[index]
                pos_prototypes = prototypes[pos_proto_id]
                proto_selecteds.append(prototypes)
                temp_protos.append(density)

                # sample negative prototypes
                all_proto_id = [i for i in range(im2cluster.max())]

                neg_proto_id = set(all_proto_id) - set(pos_proto_id.tolist())
                if self.proto_selection:
                    if n == (len(cluster_result['im2cluster']) - 1):
                        neg_proto_id = list(neg_proto_id)
                        neg_proto_id = torch.LongTensor(neg_proto_id).to(pos_proto_id.device)
                        neg_prototypes = prototypes[neg_proto_id]  # [N_neg, D]
                        logits_proto = torch.cat([torch.einsum('nc,nc->n', [q, pos_prototypes]).unsqueeze(-1),
                                                  torch.mm(q, neg_prototypes.t())], dim=1)  # [N_q, 1+N_neg]
                        temp_map = torch.cat([density[pos_proto_id].unsqueeze(-1),
                                              density[neg_proto_id].unsqueeze(0).repeat([q.shape[0], 1])], dim=1)
                        logits_proto = logits_proto / temp_map
                    else:
                        cluster2cluster = cluster_result['cluster2cluster'][n]
                        prot_logits = cluster_result['logits'][n]
                        neg_proto_id = list(neg_proto_id)
                        neg_proto_id = torch.LongTensor(neg_proto_id).to(pos_proto_id.device)
                        neg_prototypes = prototypes[neg_proto_id]  # [N_neg, D]
                        neg_mask = self.sample_neg_protos(im2cluster, cluster2cluster, pos_proto_id, prot_logits, n,
                                                          cluster_result)  # [N, N_neg]
                        neg_logit_mask = neg_mask.clone().float()  # [N_q, N_neg]
                        neg_logits = torch.mm(q, neg_prototypes.t())  # [N_q, N_neg] ~ range([-1, 1])
                        neg_logits *= neg_logit_mask
                        logits_proto = torch.cat([torch.einsum('nc,nc->n', [q, pos_prototypes]).unsqueeze(-1),
                                                  neg_logits], dim=1)
                        temp_map = torch.cat([density[pos_proto_id].unsqueeze(-1),
                                              density[neg_proto_id].unsqueeze(0).repeat([q.shape[0], 1])], dim=1)
                        logits_proto = logits_proto / temp_map
                else:
                    neg_proto_id = list(neg_proto_id)
                    neg_proto_id = torch.LongTensor(neg_proto_id).to(pos_proto_id.device)
                    neg_prototypes = prototypes[neg_proto_id]  # [N_neg, D]
                    # [N, 1] + [N, N_neg] => [N, 1 + N_neg]
                    logits_proto = torch.cat([torch.einsum('nc,nc->n', [q, pos_prototypes]).unsqueeze(-1),
                                              torch.mm(q, neg_prototypes.t())], dim=1)
                    temp_map = torch.cat([density[pos_proto_id].unsqueeze(-1),
                                          density[neg_proto_id].unsqueeze(0).repeat([q.shape[0], 1])], dim=1)

                    logits_proto = logits_proto / temp_map

                labels_proto = torch.zeros(q.shape[0], dtype=torch.long).cuda()

                proto_labels.append(labels_proto)
                proto_logits.append(logits_proto)

            return proto_logits, proto_labels, proto_selecteds, temp_protos
        else:
            return None, None, None, None


    def forward(self, feature, cluster_result,alpha=0.2):
        device = feature.device
        # dis_loss = torch.zeros(1).to(device)
        # div_loss = torch.zeros(1).to(device)
        # mgr = feature.mean()

        loss = 0
        prototype_lists = cluster_result['im2cluster']
        cluster_idss = cluster_result['centroids']
        prototype_list = cluster_result['im2cluster'][0]
        cluster_ids = cluster_result['centroids'][0]

        concentration_m = []
        # for c in range(len(cluster_ids)):


        for sample in feature:

            prototypes_sel = prototype_list[cluster_ids[:].detach().cpu().numpy()]
            concentration_sel = self.concentration_list[m][cluster_ids[:].detach().cpu().numpy()]
            nominator = torch.sum(z1 * prototypes_sel / concentration_sel.view(-1, 1), 1)
            denominator = torch.logsumexp(
                torch.matmul(z1, torch.transpose(self.prototype_list[m], 0, 1))
                / self.concentration_list[m].view(1, self.N_km[m]), dim=1)
            loss += -(nominator - denominator).mean()

            max_indexes = prototype_NCEloss.get_max_index(sample)


        return loss



class channel_clustering_hpro_loss(nn.Module):
    def __init__(self, part_nums_hierarchical):
        super(channel_clustering_hpro_loss, self).__init__()
        self.N_km = part_nums_hierarchical

    def get_protos(self, q, index, cluster_result):
        # prototypical contrast
        if cluster_result is not None:
            proto_labels = []
            proto_logits = []
            proto_selecteds = []
            temp_protos = []
            for n, (im2cluster, prototypes, density) in enumerate(
                    zip(cluster_result['im2cluster'], cluster_result['centroids'], cluster_result['density'])):

                pos_proto_id = im2cluster[index]
                pos_prototypes = prototypes[pos_proto_id]
                proto_selecteds.append(prototypes)
                temp_protos.append(density)

                # sample negative prototypes
                all_proto_id = [i for i in range(im2cluster.max())]

                neg_proto_id = set(all_proto_id) - set(pos_proto_id.tolist())
                if self.proto_selection:
                    if n == (len(cluster_result['im2cluster']) - 1):
                        neg_proto_id = list(neg_proto_id)
                        neg_proto_id = torch.LongTensor(neg_proto_id).to(pos_proto_id.device)
                        neg_prototypes = prototypes[neg_proto_id]  # [N_neg, D]
                        logits_proto = torch.cat([torch.einsum('nc,nc->n', [q, pos_prototypes]).unsqueeze(-1),
                                                  torch.mm(q, neg_prototypes.t())], dim=1)  # [N_q, 1+N_neg]
                        temp_map = torch.cat([density[pos_proto_id].unsqueeze(-1),
                                              density[neg_proto_id].unsqueeze(0).repeat([q.shape[0], 1])], dim=1)
                        logits_proto = logits_proto / temp_map
                    else:
                        cluster2cluster = cluster_result['cluster2cluster'][n]
                        prot_logits = cluster_result['logits'][n]
                        neg_proto_id = list(neg_proto_id)
                        neg_proto_id = torch.LongTensor(neg_proto_id).to(pos_proto_id.device)
                        neg_prototypes = prototypes[neg_proto_id]  # [N_neg, D]
                        neg_mask = self.sample_neg_protos(im2cluster, cluster2cluster, pos_proto_id, prot_logits, n,
                                                          cluster_result)  # [N, N_neg]
                        neg_logit_mask = neg_mask.clone().float()  # [N_q, N_neg]
                        neg_logits = torch.mm(q, neg_prototypes.t())  # [N_q, N_neg] ~ range([-1, 1])
                        neg_logits *= neg_logit_mask
                        logits_proto = torch.cat([torch.einsum('nc,nc->n', [q, pos_prototypes]).unsqueeze(-1),
                                                  neg_logits], dim=1)
                        temp_map = torch.cat([density[pos_proto_id].unsqueeze(-1),
                                              density[neg_proto_id].unsqueeze(0).repeat([q.shape[0], 1])], dim=1)
                        logits_proto = logits_proto / temp_map
                else:
                    neg_proto_id = list(neg_proto_id)
                    neg_proto_id = torch.LongTensor(neg_proto_id).to(pos_proto_id.device)
                    neg_prototypes = prototypes[neg_proto_id]  # [N_neg, D]
                    # [N, 1] + [N, N_neg] => [N, 1 + N_neg]
                    logits_proto = torch.cat([torch.einsum('nc,nc->n', [q, pos_prototypes]).unsqueeze(-1),
                                              torch.mm(q, neg_prototypes.t())], dim=1)
                    temp_map = torch.cat([density[pos_proto_id].unsqueeze(-1),
                                          density[neg_proto_id].unsqueeze(0).repeat([q.shape[0], 1])], dim=1)

                    logits_proto = logits_proto / temp_map

                labels_proto = torch.zeros(q.shape[0], dtype=torch.long).cuda()

                proto_labels.append(labels_proto)
                proto_logits.append(logits_proto)

            return proto_logits, proto_labels, proto_selecteds, temp_protos
        else:
            return None, None, None, None

    # prototype loss
    def compute_prototype_NCE(self, backbone_feature):
        loss = 0
        loss_node = 0
        for m in range(len(self.N_km)):         # for each round of kmeans
            feature = self.zs_sel[m]
            concentration_sel = self.concentration_list[m]
            rs = []
            for i in range(self.N_km[m]):  # for each cluster center
                if i == 0:
                    rs = feature[i][0]
                else:
                    rs = torch.dstack((rs, feature[i][0]))
            rs = rs.squeeze(dim=0)
            for c in range(self.N_km[m]):
                u1 = feature[c]
                r1 = u1[0]
                if u1.shape[0] != 1:
                    nominator = torch.sum(u1 * r1 / concentration_sel[c].view(-1, 1), 1)
                    denominator = torch.logsumexp(torch.matmul(u1, rs) /concentration_sel[c].view(-1,1), dim=1)
                    loss += -(nominator - denominator).mean()
            if m == 0:
                loss_node = loss.clone()

        return loss_node / backbone_feature.shape[1], (loss-loss_node) / (len(self.N_km)*backbone_feature.shape[1])
        # return loss / (len(self.N_km)*z1.shape[0])

    def forward(self, conv_features, cluster_result):
        backbone_feature = conv_features
        # prototype_lists = cluster_result['im2cluster']
        # cluster_idss = cluster_result['centroids']
        self.prototype_list = cluster_result['im2cluster']
        self.cluster_ids_list = cluster_result['cluster2cluster']
        self.zs_sel, self.concentration_list = [], []

        for m in range(len(self.N_km)):         # for each round of kmeans
            prototypes = self.prototype_list[m].cpu().detach().numpy()
            # cluster_ids = self.cluster_ids_list[m].cpu().detach().numpy()
            zs_m, concentration_m = [], []
            for c in range(self.N_km[m]):       # for each cluster center
                bs = backbone_feature.shape[0]
                zs = backbone_feature[:, prototypes == c,:,:,:].reshape(-1,891*bs)
                n_c = zs.shape[0]
                norm = torch.norm(zs - zs[0], dim=1).sum()
                concentration = norm / (n_c * math.log(n_c + 10))
                concentration_m.append(concentration)
                zs_m.append(torch.tensor(zs).cuda())
            self.zs_sel.append(zs_m)
            self.concentration_list.append(torch.tensor(concentration_m).cuda())

        loss = channel_clustering_hpro_loss.compute_prototype_NCE(self, backbone_feature)

        return loss



class channel_clustering_loss(nn.Module):
    def __init__(self):
        super(channel_clustering_loss, self).__init__()

    def forward(self, feature):
        device = feature.device

        dis_loss = torch.zeros(1).to(device)
        div_loss = torch.zeros(1).to(device)

        mgr = feature.mean()

        for sample in feature:
            max_indexes = channel_clustering_loss.get_max_index(sample)

            for region_index, region in enumerate(sample):
                max_x, max_y, max_z = max_indexes[region_index]

                for i in range(region.shape[0]):
                    for j in range(region.shape[1]):
                        for k in range(region.shape[2]):
                            dis_loss += (region[i, j, k] * region[i, j, k]) * (
                                    (max_x - i) * (max_x - i) + (max_y - j) * (max_y - j)) + ((max_z - k) * (max_z - k))

                            if region_index == 0:
                                max_others = max(sample[(region_index + 1):, i, j, k])
                            else:
                                max_others = max(
                                    torch.cat((sample[:region_index, i, j, k], sample[(region_index + 1):, i, j, k]), dim=0))

                            div_loss += (region[i, j, k] * region[i, j, k]) * ((max_others - mgr) * (max_others - mgr))


        shape = feature.shape
        sample_num, region_num, weight, height, depth = shape[0], shape[1], shape[2], shape[3], shape[4]
        sum_num = sample_num * region_num * weight * height * depth

        return dis_loss / sum_num, div_loss / sum_num

    @staticmethod
    def get_max_index(tensor):
        shape = tensor.shape
        indexes = []
        for i in range(shape[0]):
            mx = tensor[i, 0, 0, 0]
            x, y, z = 0, 0, 0
            for j in range(shape[1]):
                for k in range(shape[2]):
                    for l in range(shape[3]):
                        if tensor[i, j, k, l] > mx:
                            mx = tensor[i, j, k, l]
                            x, y, z = j, k, l
            indexes.append([x, y, z])
        return indexes