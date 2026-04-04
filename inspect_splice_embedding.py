import torch, splice

w=torch.load('embeddings/splice/n01443537/n01443537_2.pth').flatten()
voc=splice.get_vocabulary('laion',10000)

# collect non-zero concepts
nonzeros = [(i, v.item()) for i, v in enumerate(w) if v != 0]

# sort by value descending
nonzeros.sort(key=lambda x: x[1], reverse=True)

# print sorted
for idx, val in nonzeros:
    print(f"{voc[idx]} {val:.4f}")