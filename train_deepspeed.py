import os
import tqdm
import json
import torch
import loralib as lora
import lora_utils.insert_lora
import dataset.GLM as GLM_Data
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from accelerate import Accelerator, DeepSpeedPlugin
from transformers import get_linear_schedule_with_warmup

checkpoint = "THUDM/chatglm-6b"
mixed_precision = 'bf16'
lora_config = {
    'r': 32,
    'lora_alpha': 32,
    'lora_dropout': 0.1,
    'enable_lora': [True, True, True],
}
max_length = 256
LR = 2e-5
NUM_EPOCHS = 2
batch = 1
accumulate_step = 8
warm_up_ratio = 0.1

tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True, revision='main')
model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True, revision='main')
model = lora_utils.insert_lora.get_lora_model(model, lora_config)

deepspeed_plugin = DeepSpeedPlugin(zero_stage=2, gradient_accumulation_steps=accumulate_step)
accelerator = Accelerator(mixed_precision=mixed_precision, gradient_accumulation_steps=accumulate_step,
                          deepspeed_plugin=deepspeed_plugin)
device = accelerator.device
GLM_Data.device = device

import dataset.Alpaca as Alpaca_Data

pairs = Alpaca_Data.load('./data/alpaca_data.json')
pairs_encoded = GLM_Data.encode_pairs(pairs, tokenizer)
pairs_encoded = list(filter(lambda pair: len(pair['prompt']) + len(pair['completion']) <= max_length, pairs_encoded))
train_dataset = GLM_Data.GLMDataset(pairs_encoded)
train_dataloader = DataLoader(dataset=train_dataset, collate_fn=GLM_Data.collate_fn, shuffle=True, batch_size=batch)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=int(len(train_dataloader) / accumulate_step * warm_up_ratio),
    num_training_steps=(int(len(train_dataloader) / accumulate_step) * NUM_EPOCHS),
)

model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
model.to(device).train()

total_step = 0
effective_step = 0
for epoch in range(NUM_EPOCHS):  # 遍历训练集指定的epoch次数
    epoch_loss_local = 0  # 初始化每个epoch的本地损失
    for step, batch in enumerate(t := tqdm.tqdm(train_dataloader)):  # 遍历训练数据集中的每个batch
        outputs = model(**batch)  # 模型前向传播
        loss_d = outputs.loss.detach()  # 计算当前batch的损失值，并将其从计算图中分离
        epoch_loss_local += loss_d  # 累加每个batch的损失值
        t.set_description(f"loss: {epoch_loss_local.cpu().float() / step}")  # 输出当前训练进度
        loss = outputs.loss / accumulate_step  # 对梯度进行累积后，计算平均损失
        accelerator.backward(loss)  # 反向传播计算梯度
        if (step + 1) % accumulate_step == 0:  # 根据设定的梯度累积步长，进行梯度更新
            optimizer.step()  # 更新优化器参数
            lr_scheduler.step()  # 更新学习率
            optimizer.zero_grad()  # 清空梯度

    accelerator.wait_for_everyone()  # 等待所有进程完成当前epoch的训练,每次epoch跑完之前要等待一下
    all_epoch_loss, all_step = accelerator.gather((epoch_loss_local, torch.tensor(step, device=device)))  # 收集所有进程的损失值和步数信息

    if accelerator.is_main_process:  # 如果当前进程是主进程
        model_id = f"finetune_{epoch}"  # 设置模型名称
        accelerator.save(lora.lora_state_dict(accelerator.unwrap_model(model)), '/saved/' + model_id + '.pt')  # 保存模型
        epoch_loss = all_epoch_loss.float().sum() / (all_step + 1).sum()  # 计算当前epoch的平均损失
        total_step += (all_step + 1).sum()  # 累加总步数
        effective_step += ((all_step + 1) // accumulate_step).sum()  # 累加有效步数
        print(f'epoch: {epoch}, step {effective_step.cpu().numpy()}, training_loss: {epoch_loss.cpu().numpy()}')  # 输出当前epoch的训练结果

    accelerator.wait_for_everyone()  # 等待所有进程完成当前epoch的训练并进行同步

# for epoch in range(NUM_EPOCHS):
#     epoch_loss_local = 0
#     for step, batch in enumerate(t := tqdm.tqdm(train_dataloader)):
#         outputs = model(**batch)
#         loss_d = outputs.loss.detach()#它计算了一个损失值（loss）并将其从计算图中分离（detach），这通常用于防止梯度计算
#         epoch_loss_local += loss_d
#         t.set_description(f"loss: {epoch_loss_local.cpu().float() / step}")
#         loss = outputs.loss / accumulate_step # 除/梯度累积
#         accelerator.backward(loss) # 前进
#         if (step + 1) % accumulate_step == 0:# 和chatglm 的写法类似
#             optimizer.step()
#             lr_scheduler.step()
#             optimizer.zero_grad()

#     accelerator.wait_for_everyone()
#     all_epoch_loss, all_step = accelerator.gather((epoch_loss_local, torch.tensor(step, device=device)))

#     if accelerator.is_main_process:
#         model_id = f"finetune_{epoch}"
#         accelerator.save(lora.lora_state_dict(accelerator.unwrap_model(model)), '/saved/' + model_id + '.pt')
#         epoch_loss = all_epoch_loss.float().sum() / (all_step + 1).sum()
#         total_step += (all_step + 1).sum()
#         effective_step += ((all_step + 1) // accumulate_step).sum()
#         print(f'epoch: {epoch}, step {effective_step.cpu().numpy()}, training_loss: {epoch_loss.cpu().numpy()}')

#     accelerator.wait_for_everyone()
