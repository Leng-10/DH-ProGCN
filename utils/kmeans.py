import torch
import faiss
import numpy as np
import torch.nn as nn

def run_hkmeans(x, Num_cluster, T=0.2):
    """
    This function is a hierarchical
    k-means: the centroids of current hierarchy is used
    to perform k-means in next step
    """
    print('performing kmeans clustering')
    results = {'im2cluster': [], 'centroids': [], 'density': [], 'cluster2cluster': [], 'logits': []}
    ## placeholder for clustering result
    # cluster_result = {'im2cluster': [], 'centroids': [], 'density': [], 'cluster2cluster': [], 'logits': []}
    # for i, num_cluster in enumerate(Num_cluster):
    #     cluster_result['im2cluster'].append(torch.zeros(len(labels), dtype=torch.long).cuda())
    #     cluster_result['centroids'].append(torch.zeros(int(num_cluster), channel_dim).cuda())
    #     cluster_result['density'].append(torch.zeros(int(num_cluster)).cuda())
    #     if i < (len(Num_cluster) - 1):
    #         cluster_result['cluster2cluster'].append(torch.zeros(int(num_cluster), dtype=torch.long).cuda())
    #         cluster_result['logits'].append(torch.zeros([int(num_cluster), int(Num_cluster[i + 1])]).cuda())

    for seed, num_cluster in enumerate(Num_cluster):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 50
        clus.min_points_per_centroid = 1

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = 0
        index = faiss.GpuIndexFlatL2(res, d, cfg)
        if seed == 0:  # the first hierarchy from instance directly
            clus.train(x, index)
            D, I = index.search(x, 1)  # for each sample, find cluster distance and assignments
        else:
            # the input of higher hierarchy is the centorid of lower one
            clus.train(results['centroids'][seed - 1].cpu().numpy(), index)
            D, I = index.search(results['centroids'][seed - 1].cpu().numpy(), 1)

        im2cluster = [int(n[0]) for n in I]
        # sample-to-centroid distances for each cluster
        ## centroid in lower level to higher level
        Dcluster = [[] for c in range(k)]
        for im, i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])

        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

        if seed > 0:  # the im2cluster of higher hierarchy is the index of previous hierachy
            im2cluster = np.array(im2cluster)  # enable batch indexing
            results['cluster2cluster'].append(torch.LongTensor(im2cluster).cuda())
            im2cluster = im2cluster[results['im2cluster'][seed - 1].cpu().numpy()]
            im2cluster = list(im2cluster)

        if len(set(im2cluster)) == 1:
            print("Warning! All samples are assigned to one cluster")

        # concentration estimation (phi)
        density = np.zeros(k)
        for i, dist in enumerate(Dcluster):
            if len(dist) > 1:
                d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
                density[i] = d

        # if cluster only has one point, use the max to estimate its concentration
        dmax = density.max()
        for i, dist in enumerate(Dcluster):
            if len(dist) <= 1:
                density[i] = dmax

        density = density.clip(np.percentile(density, 10), np.percentile(density, 90))
        density = T * density / density.mean()

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        # centroids = nn.functional.normalize(centroids, p=2, dim=1)
        if seed > 0:  # maintain a logits from lower prototypes to higher
            proto_logits = torch.mm(results['centroids'][-1], centroids.t())
            results['logits'].append(proto_logits.cuda())

        density = torch.Tensor(density).cuda()
        im2cluster = torch.LongTensor(im2cluster).cuda()
        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)

    return results

def run_kmeans(x, num_cluster, temperature=0.2):
    """
    Args:
        x: data to be clustered
    """

    print('performing kmeans clustering')
    results = {'im2cluster': [], 'centroids': [], 'density': []}

    for seed, num_cluster in enumerate(num_cluster):
        # intialize faiss clustering parameters
        d = x.shape[1]
        k = int(num_cluster)
        clus = faiss.Clustering(d, k)
        clus.verbose = True
        clus.niter = 20
        clus.nredo = 5
        clus.seed = seed
        clus.max_points_per_centroid = 1000
        clus.min_points_per_centroid = 10

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = 0
        index = faiss.GpuIndexFlatL2(res, d, cfg)

        clus.train(x, index)

        D, I = index.search(x, 1)  # for each sample, find cluster distance and assignments
        im2cluster = [int(n[0]) for n in I]

        # get cluster centroids
        centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

        # sample-to-centroid distances for each cluster
        Dcluster = [[] for c in range(k)]
        for im, i in enumerate(im2cluster):
            Dcluster[i].append(D[im][0])

        # concentration estimation (phi)
        density = np.zeros(k)
        for i, dist in enumerate(Dcluster):
            if len(dist) > 1:
                d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
                density[i] = d

                # if cluster only has one point, use the max to estimate its concentration
        dmax = density.max()
        for i, dist in enumerate(Dcluster):
            if len(dist) <= 1:
                density[i] = dmax

        density = density.clip(np.percentile(density, 10),
                               np.percentile(density, 90))  # clamp extreme values for stability
        density = temperature * density / density.mean()  # scale the mean to temperature

        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).cuda()
        centroids = nn.functional.normalize(centroids, p=2, dim=1)

        im2cluster = torch.LongTensor(im2cluster).cuda()
        density = torch.Tensor(density).cuda()

        results['centroids'].append(centroids)
        results['density'].append(density)
        results['im2cluster'].append(im2cluster)

    return results
