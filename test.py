from SMT5 import SMT5QADataLoader, SMT5CLModel
dataloader = SMT5QADataLoader('smt5_ckpt', 150, 50)
[train_dataloader] = dataloader.get_dataloader(batch_size=2, types=['train'])
limit = 3
cnt = 0
for batch in train_dataloader:
    print(batch)
    cnt += 1
    if cnt > limit:
        break

model = SMT5CLModel('smt5_ckpt')
l = model(batch)
print(l)