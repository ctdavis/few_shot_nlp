import torch
import torch.nn as nn
import torch.optim as optim
from fast_pytorch_kmeans import KMeans

import learn2learn as l2l

import spacy
import pandas as pd
import numpy as np
import random

def wn(x):
    if hasattr(x, 'weight') and not hasattr(x, 'weight_g'):
        return torch.nn.utils.weight_norm(x)
    return x

class Classifier(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Classifier, self).__init__()
        self.l0 = wn(nn.Linear(96, 96))
        self.l1 = wn(nn.Linear(96, 96))
        self.l2 = wn(nn.Linear(96, 3))
        self.act = nn.SELU()
    def forward(self, x):
        l0 = self.act(self.l0(x))
        l1 = self.act(self.l1(l0))
        return self.l2(l1 + x)
        
class Protonet(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Protonet, self).__init__()
        self.l0 = wn(nn.Linear(96, 96))
        self.l1 = wn(nn.Linear(96, 96))
        self.l2 = wn(nn.Linear(96, 32))
        self.act = nn.SELU()
    def forward(self, supports, queries):
        output_matrix = torch.zeros((len(supports), len(queries)))
        supports = self.l2(self.act(self.l1(self.act(self.l0(supports))) + supports))
        queries = self.l2(self.act(self.l1(self.act(self.l0(queries))) + queries))
        for ix,query in enumerate(queries.chunk(len(supports))):
            output_matrix[ix] = -(supports - query).pow(2).sum(-1)
        return output_matrix
        
def to_fewshot_dataset(encodings, targets, n_shots, protonet=False):
    n = n_shots * 2
    grouped_encodings = {i:[] for i in set(targets)}
    for encoding,i in zip(encodings, targets):
        grouped_encodings[i] += [encoding]
    grouped_encodings = {
        i:random.sample(es, n)
        for i,es in grouped_encodings.items()
        if len(es) >= n
    }
    ordered_encodings = [(e, ix) for ix,es in enumerate(grouped_encodings.values()) for e in es]
    adaptation_ixs = list(map(lambda x: x*2, list(range(len(ordered_encodings)//2))))
    A = [io for ix,io in enumerate(ordered_encodings) if ix in adaptation_ixs]
    E = [io for ix,io in enumerate(ordered_encodings) if ix not in adaptation_ixs]
    if protonet:
        Ax, Ay = zip(*A)
        grouped_encodings = {i:[] for i in set(Ay)}
        for encoding,i in zip(Ax, Ay):
            grouped_encodings[i] += [encoding]
        A = [[torch.cat(encs[:-1]).mean(0, keepdim=True),encs[-1]] for trg,encs in grouped_encodings.items()]
        Ex, Ey = zip(*E)
        grouped_encodings = {i:[] for i in set(Ey)}
        for encoding,i in zip(Ex, Ey):
            grouped_encodings[i] += [encoding]
        E = [[torch.cat(encs[:-1]).mean(0, keepdim=True),encs[-1]] for trg,encs in grouped_encodings.items()]
        return A, E
    return A, E

nlp = spacy.load('en_core_web_sm')
df = pd.read_csv('data/Tweets.csv')

label_dict = {l:ix for ix,l in enumerate(df.airline_sentiment.unique().tolist())}

protonet = True
if protonet:
    classifier = Protonet()
else:
    classifier = Classifier()
maml = l2l.algorithms.MAML(classifier, lr=0.01)
opt = optim.Adam(maml.parameters())
ce = nn.CrossEntropyLoss()
if protonet:
    min_classes, max_classes = (2, 7)
else:
    min_classes, max_classes = (3, 4)
_n_ways = [i for i in range(min_classes, max_classes)]
kmeans = {i:KMeans(n_clusters=i, mode='euclidean') for i in range(min_classes, max_classes)}

freq = 100
n_iterations = 1000
sample_size = 500
_n_shots = [i for i in range(10, 31)]
n_adaptation_steps = 3

for ix in range(n_iterations):

    n_shots = random.choice(_n_shots)
    n_ways = random.choice(_n_ways)
    
    learner = maml.clone()
    
    samples = df.sample(sample_size)
    encodings = [
        torch.from_numpy(nlp(sample).vector).view(1, -1)
        for sample in samples.text.tolist()
    ]
    targets = kmeans[n_ways].fit_predict(torch.cat(encodings).detach()).tolist()
    
    if protonet:
        A, E = to_fewshot_dataset(encodings, targets, n_shots, protonet=True)
        Ax, Ay = zip(*A)
        Ex, Ey = zip(*E)
        Ax = torch.cat(Ax)
        Ay = torch.cat(Ay)
        Ex = torch.cat(Ex)
        Ey = torch.cat(Ey)
        trgs = torch.LongTensor(list(range(len(Ey))))
        for _ in range(n_adaptation_steps):
            preds = learner(Ax, Ay)
            learner.adapt(ce(preds, trgs))
        preds = learner(Ex, Ey)
        iteration_loss = ce(preds, trgs)
        iteration_acc = (preds.softmax(-1).argmax(-1) == trgs).long().float().mean()

    else:
        A, E = to_fewshot_dataset(encodings, targets, n_shots)
        Ax, Ay = zip(*A)
        Ex, Ey = zip(*E)
        Ax = torch.cat(Ax)
        Ay = torch.LongTensor(Ay)
        Ex = torch.cat(Ex)
        Ey = torch.LongTensor(Ey)
    
        for _ in range(n_adaptation_steps):
            learner.adapt(ce(learner(Ax), Ay))
        preds = learner(Ex)
        iteration_loss = ce(preds, Ey)
        iteration_acc = (preds.softmax(-1).argmax(-1) == Ey).long().float().mean()
        
    iteration_loss.backward()
    
    opt.step()
    opt.zero_grad()

    print(f'{ix}, Loss: {round(iteration_loss.item(), 3)}, Acc: {round(iteration_acc.item(), 2) * 100}')

learner = maml.clone()

samples = df.sample(sample_size)
encodings = [torch.from_numpy(nlp(sample_text).vector).view(1, -1) for sample_text in samples.text.tolist()]
targets = [label_dict[t] for t in samples.airline_sentiment.tolist()]


if protonet:
    A, E = to_fewshot_dataset(encodings, targets, max(_n_shots), protonet=True)
    Ax, Ay = zip(*A)
    Ex, Ey = zip(*E)
    Ax = torch.cat(Ax)
    Ay = torch.cat(Ay)
    Ex = torch.cat(Ex)
    Ey = torch.cat(Ey)
    trgs = torch.LongTensor(list(range(len(Ey))))
    for _ in range(n_adaptation_steps):
        preds = learner(Ax, Ay)
        learner.adapt(ce(preds, trgs))
    preds = learner(Ex, Ey)
    final_loss = ce(preds, trgs)
    final_acc = (preds.softmax(-1).argmax(-1) == trgs).long().float().mean()
else:
    A, E = to_fewshot_dataset(encodings, targets, max(_n_shots))
    Ax, Ay = zip(*A)
    Ex, Ey = zip(*E)
    Ax = torch.cat(Ax)
    Ay = torch.LongTensor(Ay)
    Ex = torch.cat(Ex)
    Ey = torch.LongTensor(Ey)
    for _ in range(n_adaptation_steps):
        learner.adapt(ce(learner(Ax), Ay))
    preds = learner(Ex)
    final_loss = ce(preds, Ey)
    final_acc = (preds.softmax(-1).argmax(-1) == Ey).long().float().mean()

print(f'Final, Loss: {round(final_loss.item(), 3)}, Acc: {round(final_acc.item(), 2) * 100}')

    

