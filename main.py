import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ================= DATA =================
def generate_dataset():
    entities = []
    relations = ["Rel_Direct", "Rel_Path1", "Rel_Path2", "Rel_Target"]

    train, test = [], []

    target = "T0"
    entities.append(target)

    # Easy cases
    for i in range(200):
        s = f"E{i}"
        entities.append(s)
        if i < 160:
            train.append((s, "Rel_Direct", target))
        else:
            test.append((s, "Rel_Direct", target))

    # 2-hop reasoning
    for i in range(60):
        s, m, o = f"S{i}", f"M{i}", f"O{i}"
        entities += [s, m, o]

        train.append((s, "Rel_Path1", m))
        train.append((m, "Rel_Path2", o))

        if i >= 40:
            test.append((s, "Rel_Target", o))

    entities = list(set(entities))
    ent2id = {e: i for i, e in enumerate(entities)}
    rel2id = {r: i for i, r in enumerate(relations)}

    n = len(entities)
    adj = torch.zeros(n, n)

    for s, r, o in train:
        adj[ent2id[s], ent2id[o]] = 1.0

    return train, test, ent2id, rel2id, adj


# ================= MODELS =================
class TransE(nn.Module):
    def __init__(self, n_ent, n_rel, dim):
        super().__init__()
        self.e = nn.Embedding(n_ent, dim)
        self.r = nn.Embedding(n_rel, dim)

        nn.init.xavier_uniform_(self.e.weight)
        nn.init.xavier_uniform_(self.r.weight)

    def forward(self, h, r):
        return torch.matmul(self.e(h) + self.r(r), self.e.weight.t())


class RotatE(nn.Module):
    def __init__(self, n_ent, n_rel, dim):
        super().__init__()
        self.e_re = nn.Embedding(n_ent, dim)
        self.e_im = nn.Embedding(n_ent, dim)
        self.r_phase = nn.Embedding(n_rel, dim)

        nn.init.xavier_uniform_(self.e_re.weight)
        nn.init.xavier_uniform_(self.e_im.weight)
        nn.init.uniform_(self.r_phase.weight, 0, 3.14)

    def forward(self, h, r):
        phase = self.r_phase(r)
        r_re = torch.cos(phase)
        r_im = torch.sin(phase)

        h_re = F.normalize(self.e_re(h), dim=-1)
        h_im = F.normalize(self.e_im(h), dim=-1)

        re = h_re * r_re - h_im * r_im
        im = h_re * r_im + h_im * r_re

        return torch.matmul(re, self.e_re.weight.t()) + \
               torch.matmul(im, self.e_im.weight.t())


class ComplEx(nn.Module):
    def __init__(self, n_ent, n_rel, dim):
        super().__init__()
        self.e_re = nn.Embedding(n_ent, dim)
        self.e_im = nn.Embedding(n_ent, dim)
        self.r_re = nn.Embedding(n_rel, dim)
        self.r_im = nn.Embedding(n_rel, dim)

        for emb in [self.e_re, self.e_im, self.r_re, self.r_im]:
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, h, r):
        h_re = F.normalize(self.e_re(h), dim=-1)
        h_im = F.normalize(self.e_im(h), dim=-1)
        r_re = self.r_re(r)
        r_im = self.r_im(r)

        return torch.matmul(h_re * r_re - h_im * r_im, self.e_re.weight.t()) + \
               torch.matmul(h_re * r_im + h_im * r_re, self.e_im.weight.t())


# ================= SYNERGIA (FINAL) =================
class SYNERGIA(nn.Module):
    def __init__(self, n_ent, n_rel, dim, adj):
        super().__init__()
        self.e = nn.Embedding(n_ent, dim)
        self.r = nn.Embedding(n_rel, dim)

        nn.init.xavier_uniform_(self.e.weight)
        nn.init.xavier_uniform_(self.r.weight)

        self.W = nn.Linear(dim, dim)
        self.adj = adj.float()

    def forward(self, h, r):
        h_emb = self.e(h)
        r_emb = self.r(r)

        # 1-hop score
        score1 = torch.matmul(h_emb + r_emb, self.e.weight.t())

        # Normalize adjacency
        deg = self.adj.sum(1, keepdim=True) + 1e-6
        norm_adj = self.adj / deg

        # Path embeddings
        path_emb = torch.matmul(norm_adj, self.e.weight)
        path_emb = F.normalize(self.W(path_emb), dim=-1)

        score2 = torch.matmul(h_emb + r_emb, path_emb.t())

        # 🔥 Adaptive gating (key innovation)
        gate = torch.sigmoid(2 * (h_emb * r_emb).sum(dim=-1, keepdim=True))

        final_score = score1 + 1.0 * gate * score2

        return final_score


# ================= TRAIN =================
def train_eval(model, train, test, ent2id, rel2id, epochs=80):
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total = 0
        model.train()

        for s, r, o in train:
            h = torch.tensor([ent2id[s]])
            r_id = torch.tensor([rel2id[r]])
            t = torch.tensor([ent2id[o]])

            out = model(h, r_id)
            loss = loss_fn(out, t)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss {total:.2f}")

    # Evaluation
    model.eval()
    correct = 0

    with torch.no_grad():
        for s, r, o in test:
            h = torch.tensor([ent2id[s]])
            r_id = torch.tensor([rel2id[r]])

            out = model(h, r_id)
            pred = torch.argmax(out)

            if pred.item() == ent2id[o]:
                correct += 1

    return correct / len(test)


# ================= MAIN =================
train, test, ent2id, rel2id, adj = generate_dataset()

n_ent = len(ent2id)
n_rel = len(rel2id)
dim = 64

print("Training TransE")
tre = train_eval(TransE(n_ent, n_rel, dim), train, test, ent2id, rel2id)

print("Training RotatE")
rot = train_eval(RotatE(n_ent, n_rel, dim), train, test, ent2id, rel2id)

print("Training ComplEx")
com = train_eval(ComplEx(n_ent, n_rel, dim), train, test, ent2id, rel2id)

print("Training SYNERGIA")
syn = train_eval(SYNERGIA(n_ent, n_rel, dim, adj), train, test, ent2id, rel2id)

print("\nFINAL RESULTS")
print("------------------")
print(f"TransE   : {tre*100:.2f}%")
print(f"RotatE   : {rot*100:.2f}%")
print(f"ComplEx  : {com*100:.2f}%")
print(f"SYNERGIA : {syn*100:.2f}%")
