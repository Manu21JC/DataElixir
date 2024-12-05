import torch
import torch.nn as nn
import torchvision.transforms as transforms
import logging
import numpy as np

from tqdm import tqdm
from copy import deepcopy
from collections import OrderedDict, Counter
from torch.utils.data import DataLoader
from torch.nn.functional import adaptive_avg_pool2d
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy as kl

from pytorch_diffusion import Diffusion

totensor = transforms.ToTensor()
topil = transforms.ToPILImage()
transform_test = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])])

def normalize_to_neg_one_to_one(x):
    return x * 2 - 1

def unnormalize_to_zero_to_one(x):
    return (x + 1) * 0.5

class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out

def distance(img_a, img_b, feature_a, feature_b):
    pixel_similarity = (ssim(np.array(img_a), np.array(img_b), multichannel=True, win_size=3) + 1) * 0.5
    feature_similarity = (torch.cosine_similarity(feature_a, feature_b, dim=0).item() + 1) * 0.5
    return 0.5 * (1 - pixel_similarity) + 0.5 * (1 - feature_similarity)

def tensor2img(tensorslist):
    imgslist = [[] for _ in range(len(tensorslist))]
    for i, tensors in enumerate(tensorslist):
        for tensor in tensors:
            tensor = torch.clamp(tensor, -1, 1)
            tensor = unnormalize_to_zero_to_one(tensor)
            tensor = torch.clamp(tensor, 0, 1)
            img = topil(tensor)
            imgslist[i].append(img)
    return imgslist

def anomalous_samples_identification(adv_model, intermediary_images, tau=5, device="cuda"):
    adv_model.to(device)
    candidate_data = [[] for _ in range(len(intermediary_images))]
    candidate_targets = [[] for _ in range(len(intermediary_images))]
    softmax = nn.Softmax(dim=-1)
    for idx, imgs in enumerate(intermediary_images):
        tensors = torch.stack([transform_test(img) for img in imgs]).to(device)
        with torch.no_grad():
            logits = adv_model(tensors)
            _, predicts = torch.max(softmax(logits), -1)
        predicts = predicts.detach().cpu().numpy()
        candidate_predicts = predicts[1:]
        counter = Counter(candidate_predicts)
        if len(counter) == 1:
            candidate_targets[idx].extend(predicts.tolist())
            candidate_data[idx].extend(imgs)
        else:
            if counter.most_common(2)[1][1] >= tau:
                candidate_targets[idx].extend(predicts.tolist())
                candidate_data[idx].extend(imgs)
            else:
                candidate_targets[idx].append(predicts[0])
                candidate_data[idx].append(imgs[0])
                most_common_label = counter.most_common(1)[0][0]
                for j, predict in enumerate(candidate_predicts):
                    if predict == most_common_label:
                        candidate_data[idx].append(imgs[j+1])
                        candidate_targets[idx].append(most_common_label)
    return candidate_data, candidate_targets

def target_label_detection(candidate_set_targets, raw_dataset, metric_type="kl"):
    target_label = []
    num_labels = len(set(raw_dataset.targets))
    label_intermediaries = [[] for _ in range(num_labels)]
    for i, targets in enumerate(candidate_set_targets):
        if len(set(targets)) != 1:
            label_intermediaries[raw_dataset.targets[i]].extend(targets)
    metrics = []
    if metric_type == "count":
        metrics = [len(intermediaries) for intermediaries in label_intermediaries]
    elif metric_type == "kl":
        for i, intermediaries in enumerate(label_intermediaries):
            p = np.zeros(num_labels)
            p[i] = 1
            q = np.array([intermediaries.count(j) / len(intermediaries) for j in range(num_labels)])
            metrics.append(kl(p, q))
    logging.info(f"metrics: {metrics}")
    metrics = np.array(metrics)
    median = np.median(metrics)
    mad = np.median(np.abs(metrics - median))
    scores = np.abs(metrics - median)/1.4826/(mad + 1e-6)
    logging.info(f"scores: {scores}")
    target_label = np.where(scores > 2)[0]
    logging.info(f"target label: {target_label}")
    return target_label

def get_predict(benign_model, suspicious_dataset, device='cuda'):
    benign_model.to(device)
    predicts = []
    softmax = nn.Softmax(dim=-1)
    for i in range(len(suspicious_dataset.data)):
        imgs = suspicious_dataset.data[i][0]
        tensors = torch.stack([transform_test(img) for img in imgs]).to(device)
        with torch.no_grad():
            logit = benign_model(tensors)
            _, predict = torch.max(softmax(logit), -1)
        predict = predict.detach().cpu().numpy()
        predicts.append(predict)
    return np.array(predicts)

def feature_extraction(candidate_set_data, raw_dataset, target_label, adv_model, layer_name="layer4", device="cuda"):
    candidate_set_features = []
    adv_model.to(device)
    adv_model = IntermediateLayerGetter(adv_model, {layer_name:'feature'})
    for i in range(len(candidate_set_data)):
        if raw_dataset.targets[i] in target_label:
            tensors = torch.stack([transform_test(img) for img in candidate_set_data[i]]).to(device)
            with torch.no_grad():
                out = adv_model(tensors)
                feature = out["feature"]
                feature = adaptive_avg_pool2d(feature, output_size=(1, 1))
                feature = feature.squeeze(3).squeeze(2).cpu()
            candidate_set_features.append(feature)
        else:
            candidate_set_features.append(torch.zeros([1,1]))
    return candidate_set_features

def image_selection(imgs, features):
    dists = [0]
    for i in range(1, len(imgs)):
        dist = distance(imgs[0], imgs[i], features[0], features[i])
        dists.append(dist)
    sorted_index = np.argsort(np.array(dists))
    return sorted_index

def cur_purified_dataset_generation(candidate_set_data, candidate_set_targets, raw_dataset, adv_model):
    cur_purified_data = []
    cur_purified_targets = []
    target_label = target_label_detection(candidate_set_targets, raw_dataset)
    candidate_set_features = feature_extraction(candidate_set_data, raw_dataset, target_label, adv_model, layer_name="layer4")
    for i in tqdm(range(len(candidate_set_data))):
        if len(set(candidate_set_targets[i])) == 1:
            cur_purified_data.append(candidate_set_data[i][0])
            cur_purified_targets.append(raw_dataset.targets[i])
        else:
            if len(set(candidate_set_targets[i][1:])) == 1:
                if raw_dataset.targets[i] in target_label:
                    cur_purified_data.append(candidate_set_data[i][image_selection(candidate_set_data[i],candidate_set_features[i])[int(len(candidate_set_data[i])*0.8)]])
                    cur_purified_targets.append(candidate_set_targets[i][1])
                else:
                    cur_purified_data.append(candidate_set_data[i][0])
                    cur_purified_targets.append(raw_dataset.targets[i])
            else:
                if raw_dataset.targets[i] in target_label:
                    cur_purified_data.append([candidate_set_data[i], candidate_set_features[i]])
                    cur_purified_targets.append(Counter(candidate_set_targets[i][1:]).most_common(1)[0][0])
                else:
                    cur_purified_data.append(candidate_set_data[i][0])
                    cur_purified_targets.append(raw_dataset.targets[i])
    return cur_purified_data, np.array(cur_purified_targets)

def candidate_set_construction(raw_dataset, adv_model=None, batch_size=1024, T=150, n=5, m=10, tau=5):
    dataloader = DataLoader(dataset=raw_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    candidate_set_data = []
    candidate_set_targets = []
    diffusion = Diffusion.from_pretrained("cifar10")
    for batch_idx, (x, *additional_info) in tqdm(enumerate(dataloader)):
        x = normalize_to_neg_one_to_one(x)
        intermediary_tensors = [[x[i]] for i in range(len(x))]
        with torch.no_grad():
            for _ in range(n):
                x = diffusion.diffuse_t_steps(x, T)
                x, intermediaries = diffusion.denoise(x.shape[0], n_steps=T, x=x.to(diffusion.device), curr_step=T)
                for i in range(len(intermediary_tensors)):
                    intermediary_tensors[i].extend(intermediaries[i][-m:])
                x = torch.clamp(x, -1, 1)
        intermediary_images = tensor2img(intermediary_tensors)
        candidate_data, candidate_targets = anomalous_samples_identification(adv_model, intermediary_images, tau=tau)
        candidate_set_data.extend(candidate_data)
        candidate_set_targets.extend(candidate_targets)
    return candidate_set_data, candidate_set_targets

def purify(adv_dataset, adv_model, args=None):
    raw_dataset = deepcopy(adv_dataset)
    adv_model.eval()
    candidate_set_data, candidate_set_targets = candidate_set_construction(raw_dataset, adv_model=adv_model, batch_size=args.purify_batch_size, T=args.steps_T, n=args.rounds_n, m=args.last_steps_m, tau=args.threshold_tau)
    cur_purified_data, cur_purified_targets = cur_purified_dataset_generation(candidate_set_data, candidate_set_targets, raw_dataset, adv_model)
    index_cur_purified = [i for i, data in enumerate(cur_purified_data) if not isinstance(data, list)]
    index_suspicious = [i for i, data in enumerate(cur_purified_data) if isinstance(data, list)]
    cur_purified_dataset = deepcopy(raw_dataset)
    cur_purified_dataset.data = cur_purified_data
    cur_purified_dataset.targets = cur_purified_targets
    suspicious_dataset = deepcopy(cur_purified_dataset)
    cur_purified_dataset.subset(index_cur_purified)
    suspicious_dataset.subset(index_suspicious)
    return cur_purified_dataset, suspicious_dataset, index_cur_purified, index_suspicious

def prepare_finetune_dataset(suspicious_dataset, cur_purified_dataset, benign_model):
    benign_model.eval()
    predicts = get_predict(benign_model, suspicious_dataset)
    targets = np.array([predicts[i][0] for i in range(len(predicts))])
    index_consensus = [i for i in range(len(targets)) if targets[i] == suspicious_dataset.targets[i]]
    consensus_dataset = deepcopy(suspicious_dataset)
    consensus_dataset.subset(index_consensus)
    consensus_predicts = predicts[index_consensus]
    finetune_dataset = deepcopy(cur_purified_dataset)
    finetune_data = finetune_dataset.data
    finetune_targets = finetune_dataset.targets.tolist()
    finetune_original_targets = finetune_dataset.original_targets.tolist()
    finetune_poison_indicator = finetune_dataset.poison_indicator.tolist()
    finetune_original_index = finetune_dataset.original_index.tolist()

    for i in tqdm(range(len(consensus_dataset.data))):
        imgs = [consensus_dataset.data[i][0][0]]
        features = [consensus_dataset.data[i][1][0]]
        for j in range(1, len(consensus_predicts[i])):
            if consensus_predicts[i][j] == consensus_dataset.targets[i]:
                imgs.append(consensus_dataset.data[i][0][j])
                features.append(consensus_dataset.data[i][1][j])
        if len(imgs) == 1:
            finetune_data.append(imgs[0])
        else:
            finetune_data.append(imgs[image_selection(imgs,features)[int(len(imgs)*0.8)]])
        finetune_targets.append(consensus_dataset.targets[i])
        finetune_original_targets.append(consensus_dataset.original_targets[i])
        finetune_poison_indicator.append(consensus_dataset.poison_indicator[i])
        finetune_original_index.append(consensus_dataset.original_index[i])

    random_index = np.arange(0,len(finetune_targets),1)
    np.random.shuffle(random_index)
    finetune_dataset.data = np.array(finetune_data, dtype=object)[random_index].tolist()
    finetune_dataset.targets = np.array(finetune_targets)[random_index]
    finetune_dataset.original_targets = np.array(finetune_original_targets)[random_index]
    finetune_dataset.poison_indicator = np.array(finetune_poison_indicator)[random_index]
    finetune_dataset.original_index = np.array(finetune_original_index)[random_index]
    return finetune_dataset

def final_purified_dataset_generation(adv_dataset, cur_purified_dataset, suspicious_dataset, index_cur_purified, index_suspicious, benign_model):
    benign_model.eval()
    predicts = get_predict(benign_model, suspicious_dataset)
    suspicious_targets = np.array([predicts[i][0] for i in range(len(predicts))])
    suspicious_dataset.targets = suspicious_targets
    suspicious_data = []
    for i in tqdm(range(len(suspicious_dataset.data))):
        imgs = [suspicious_dataset.data[i][0][0]]
        features = [suspicious_dataset.data[i][1][0]]
        for j in range(1, len(predicts[i])):
            if predicts[i][j] == suspicious_targets[i]:
                imgs.append(suspicious_dataset.data[i][0][j])
                features.append(suspicious_dataset.data[i][1][j])
        if len(imgs) == 1:
            suspicious_data.append(imgs[0])
        else:
            suspicious_data.append(imgs[image_selection(imgs,features)[int(len(imgs)*0.8)]])
    suspicious_dataset.data = suspicious_data
    final_purified_dataset = deepcopy(adv_dataset)
    final_purified_dataset.targets[index_cur_purified] = cur_purified_dataset.targets
    final_purified_dataset.targets[index_suspicious] = suspicious_dataset.targets
    final_purified_data = np.array(final_purified_dataset.data, dtype=object)
    final_purified_data[index_cur_purified] = np.array(cur_purified_dataset.data, dtype=object)
    final_purified_data[index_suspicious] = np.array(suspicious_dataset.data, dtype=object)
    final_purified_dataset.data = final_purified_data.tolist()

    real_poison_index = np.where(final_purified_dataset.poison_indicator==1)
    pred_poison_index = np.where(final_purified_dataset.targets!=adv_dataset.targets)
    correct_pred_poison_index = np.intersect1d(real_poison_index, pred_poison_index)
    wrong_pred_poison_index = np.setdiff1d(pred_poison_index, correct_pred_poison_index)
    logging.info(f"TPR: {len(correct_pred_poison_index) / len(real_poison_index[0])}")
    logging.info(f"FPR: {len(wrong_pred_poison_index) / len(np.where(final_purified_dataset.poison_indicator==0)[0])}")