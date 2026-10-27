import json
import math
import random
import numpy as np
from bert4torch.layers import LayerNorm
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from bert4torch.snippets import sequence_padding, ListDataset
from bert4torch.callbacks import Callback
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

# 参数定义
maxlen = 512
batch_size = 8
config_path = 'chinese-roberta-wwm-ext/config.json'
checkpoint_path = 'chinese-roberta-wwm-ext/pytorch_model.bin'
dict_path = 'chinese-roberta-wwm-ext/vocab.txt'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 实体类型字典（作物病虫害统一）
entity_type2id = {
    "CROP": 0, "PEST": 1, "PATHOGEN": 2, "SITE": 3, "SYMPTOM": 4,
    "STAGE": 5, "ENVIRON": 6, "OTHERFACT": 7, "AGRCONTROL": 8,
    "CHEMCTRL": 9, "PHYSICTRL": 10, "BIOCONTROL": 11, "OVERWINTER": 12,
    "CULTIVF": 13, "TRANSMODE": 14, "INFSOURCE": 15,
}
id2entity_type = {v: k for k, v in entity_type2id.items()}

# 关系字典（作物病虫害统一）
predicate2id = {
    "SUFFER_IS": 0,
    "SYM_HAS": 1,
    "CONTROL_AGR": 2,
    "CONTROL_MED": 3,
    "CONTROL_BIO": 4,
    "STAGE_IN": 5,
    "EFFECT_TO": 6,
    "DIR_CAUSE": 7,
    "TRS_MTH": 8,
}
id2predicate = {v: k for k, v in predicate2id.items()}

# 分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

def get_spoes(text, entities, spo_list):
    """解析样本，生成三元组（SPO），并将实体类型转换为数值ID"""
    def search(pattern, sequence):
        """从sequence中寻找子串pattern"""
        n = len(pattern)
        for i in range(len(sequence)):
            if sequence[i:i + n] == pattern:
                return i
        return -1

    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
    entity_type_dict = {d['text']: entity_type2id[d['type']] for d in entities}
    spoes = {}
    for each in spo_list:
        s, p, o = each.values()
        s_type = entity_type_dict[s]
        o_type = entity_type_dict[o]
        s = tokenizer.encode(s)[0][1:-1]
        p = predicate2id[p]
        o = tokenizer.encode(o)[0][1:-1]
        s_idx = search(s, token_ids)
        o_idx = search(o, token_ids)
        if s_idx != -1 and o_idx != -1:
            s = (s_idx, s_idx + len(s) - 1, s_type)
            o = (o_idx, o_idx + len(o) - 1, p, o_type)
            if s not in spoes:
                spoes[s] = []
            spoes[s].append(o)
    return token_ids, segment_ids, spoes

class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        """加载数据"""
        D = []
        with open(filename, encoding='utf-8') as f:
            data = json.load(f)
            print(f"Loaded {len(data)} records from {filename}")
            for l in data:
                entities = l['entities']
                spo_list = l['triple_list']
                token_ids, segment_ids, spoes = get_spoes(l['text'], entities, spo_list)
                if spoes:
                    D.append({
                        'text': l['text'],
                        'entities': entities,
                        'spo_list': spo_list,
                        'token_ids': token_ids,
                        'segment_ids': segment_ids,
                        'spoes': spoes
                    })
                else:
                    print(f"No valid SPOs found for text: {l['text']}")
        print(f"Processed {len(D)} records into dataset")
        return D

def collate_fn(batch):
    batch_token_ids, batch_segment_ids = [], []
    batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []
    batch_entity_types, batch_start_pos, batch_end_pos = [], [], []
    
    for d in batch:
        token_ids, segment_ids, spoes = d['token_ids'], d['segment_ids'], d['spoes']
        entity_types = np.zeros(len(token_ids), dtype=np.int32)
        start_pos = np.zeros(len(token_ids), dtype=np.int32)
        end_pos = np.zeros(len(token_ids), dtype=np.int32)
        
        if spoes:
            subject_labels = np.zeros((len(token_ids), 2))
            for s in spoes:
                s_start, s_end, s_type = s
                entity_types[s_start:s_end+1] = s_type
                start_pos[s_start] = s_type + 1
                end_pos[s_end] = s_type + 1
                subject_labels[s_start, 0] = 1
                subject_labels[s_end, 1] = 1

            start, end, _ = random.choice(list(spoes.keys()))
            subject_ids = (start, end)

            object_labels = np.zeros((len(token_ids), len(predicate2id), 2))
            for o in spoes.get((start, end, entity_types[start]), []):
                object_labels[o[0], o[2], 0] = 1
                object_labels[o[1], o[2], 1] = 1

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_subject_labels.append(subject_labels)
            batch_subject_ids.append(subject_ids)
            batch_object_labels.append(object_labels)
            batch_entity_types.append(entity_types)
            batch_start_pos.append(start_pos)
            batch_end_pos.append(end_pos)
        else:
            print("No SPOs found in this sample!")
            
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_segment_ids = torch.tensor(sequence_padding(batch_segment_ids), dtype=torch.long, device=device)
    batch_subject_labels = torch.tensor(sequence_padding(batch_subject_labels), dtype=torch.float, device=device)
    batch_subject_ids = torch.tensor(batch_subject_ids, dtype=torch.long, device=device)
    batch_object_labels = torch.tensor(sequence_padding(batch_object_labels), dtype=torch.float, device=device)
    batch_entity_types = torch.tensor(sequence_padding(batch_entity_types), dtype=torch.long, device=device)
    batch_start_pos = torch.tensor(sequence_padding(batch_start_pos), dtype=torch.long, device=device)
    batch_end_pos = torch.tensor(sequence_padding(batch_end_pos), dtype=torch.long, device=device)
    batch_attention_mask = (batch_token_ids != tokenizer._token_pad_id)
    
    return [
        batch_token_ids, 
        batch_segment_ids, 
        batch_subject_ids, 
        batch_entity_types,
        batch_start_pos,
        batch_end_pos
    ], [batch_subject_labels, batch_object_labels, batch_attention_mask]

class CoAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, output_attentions, attention_probs_dropout_prob):
        super(CoAttention, self).__init__()
        self.output_attentions = output_attentions
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, context_states, query_states, attention_mask=None):
        mixed_query_layer = self.query(query_states)
        mixed_key_layer = self.key(context_states)
        mixed_value_layer = self.value(context_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class Model(BaseModel):
    def __init__(self, co_layers: int = 1) -> None:
        super().__init__()
        self.bert = build_transformer_model(config_path, checkpoint_path, model='bert')
        # 1) 实体类型嵌入 + 主体起止位置（实体类型标记）特征
        self.entity_type_embedding = nn.Embedding(len(entity_type2id)+1, 768)
        self.start_pos_embedding = nn.Embedding(len(entity_type2id)+1, 768)
        self.end_pos_embedding = nn.Embedding(len(entity_type2id)+1, 768)
        # 2) 主体/客体双流投影
        self.head = nn.Linear(768, 768)
        self.tail = nn.Linear(768, 768)
        # 3) 双向交叉注意力（head→tail / tail→head）
        self.co_layers = co_layers
        self.crossatt1 = CoAttention(768, 8, False, 0.)
        self.crossatt2 = CoAttention(768, 8, False, 0.)
        # 4) CasRel 主体解码 + Conditional LayerNorm + 客体解码
        self.linear1 = nn.Linear(768, 2)
        self.condLayerNorm = LayerNorm(768, conditional_size=768*2)
        self.linear2 = nn.Linear(768, len(predicate2id)*2)

    @staticmethod
    def extract_subject(inputs):
        output, subject_ids = inputs
        start = torch.gather(output, 1, subject_ids[:, :1].unsqueeze(2).expand(-1, -1, output.shape[-1]))
        end = torch.gather(output, 1, subject_ids[:, 1:].unsqueeze(2).expand(-1, -1, output.shape[-1]))
        subject = torch.cat([start, end], 2)
        return subject[:, 0]

    def forward(self, *inputs):
        token_ids, segment_ids, subject_ids, entity_types, start_pos, end_pos = inputs
        seq_output = self.bert([token_ids, segment_ids])
        entity_embed = self.entity_type_embedding(entity_types)
        start_embed = self.start_pos_embedding(start_pos)
        end_embed = self.end_pos_embedding(end_pos)
        combined_output = seq_output + entity_embed + start_embed + end_embed
        mask_token_ids = token_ids.gt(0).long()

        seq_output_head = self.head(combined_output)
        seq_output_tail = self.tail(combined_output)
        tmp_head, tmp_tail = seq_output_head, seq_output_tail

        for _ in range(self.co_layers):
            cq_biatt_output = self.crossatt1(seq_output_head, seq_output_tail, mask_token_ids)
            qc_biatt_output = self.crossatt2(seq_output_tail, seq_output_head, mask_token_ids)
            seq_output_tail = cq_biatt_output
            seq_output_head = qc_biatt_output

        seq_output_head += tmp_head
        seq_output_tail += tmp_tail

        subject_preds = (torch.sigmoid(self.linear1(seq_output_head))) ** 2
        subject = self.extract_subject([seq_output_tail, subject_ids])
        output = self.condLayerNorm(seq_output_tail, subject)
        output = (torch.sigmoid(self.linear2(output))) ** 4
        object_preds = output.view(*output.shape[:2], len(predicate2id), 2)
        return [subject_preds, object_preds]

    def predict_subject(self, inputs):
        self.eval()
        with torch.no_grad():
            token_ids, segment_ids, entity_types, start_pos, end_pos = inputs
            seq_output = self.bert([token_ids, segment_ids])
            entity_embed = self.entity_type_embedding(entity_types)
            start_embed = self.start_pos_embedding(start_pos)
            end_embed = self.end_pos_embedding(end_pos)
            combined_output = seq_output + entity_embed + start_embed + end_embed
            mask_token_ids = token_ids.gt(0).long()
            seq_output_head = self.head(combined_output)
            seq_output_tail = self.tail(combined_output)
            tmp_head, tmp_tail = seq_output_head, seq_output_tail

            for _ in range(self.co_layers):
                cq_biatt_output = self.crossatt1(seq_output_head, seq_output_tail, mask_token_ids)
                qc_biatt_output = self.crossatt2(seq_output_tail, seq_output_head, mask_token_ids)
                seq_output_tail = cq_biatt_output
                seq_output_head = qc_biatt_output

            seq_output_head += tmp_head
            seq_output_tail += tmp_tail
            subject_preds = (torch.sigmoid(self.linear1(seq_output_head))) ** 2
        return [seq_output_tail, subject_preds]

    def predict_object(self, inputs):
        self.eval()
        with torch.no_grad():
            seq_output_tail, subject_ids = inputs
            subject = self.extract_subject([seq_output_tail, subject_ids])
            output = self.condLayerNorm(seq_output_tail, subject)
            output = (torch.sigmoid(self.linear2(output))) ** 4
            object_preds = output.view(*output.shape[:2], len(predicate2id), 2)
        return object_preds

class BCELoss(nn.BCELoss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, inputs, targets):
        subject_preds, object_preds = inputs
        subject_labels, object_labels, mask = targets

        subject_loss = super().forward(subject_preds, subject_labels)
        subject_loss = subject_loss.mean(dim=-1)
        subject_loss = (subject_loss * mask).sum() / mask.sum()

        object_loss = super().forward(object_preds, object_labels)
        object_loss = object_loss.mean(dim=-1).sum(dim=-1)
        object_loss = (object_loss * mask).sum() / mask.sum()

        return subject_loss + object_loss

# 加载数据集（由 resplit_datasets.py 生成的合并数据）
train_dataset = MyDataset('../合并数据811/train.json')
val_dataset = MyDataset('../合并数据811/val.json')

# 初始化 DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

class Evaluator(Callback):
    """评估与保存"""
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, steps, epoch, logs=None):
        f1, precision, recall = evaluate(val_dataset.data)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            train_model.save_weights('checkpoint_merged811/best_model.pt')
        print(f'f1: {f1:.5f}, precision: {precision:.5f}, recall: {recall:.5f}, best f1: {self.best_val_f1:.5f}\n')

def extract_spoes(input_data):
    text = input_data['text']
    outputs, _ = collate_fn([input_data])
    batch_token_ids, batch_segment_ids, batch_subject_ids, batch_entity_types, batch_start_pos, batch_end_pos = outputs
    tokens = tokenizer.tokenize(text, maxlen=maxlen)
    mapping = tokenizer.rematch(text, tokens)

    seq_output, subject_preds = train_model.predict_subject([
        batch_token_ids.to(device), 
        batch_segment_ids.to(device), 
        batch_entity_types.to(device),
        batch_start_pos.to(device),
        batch_end_pos.to(device)
    ])

    subject_preds[:, [0, -1]] *= 0
    start = torch.where(subject_preds[0, :, 0] > 0.6)[0]
    end = torch.where(subject_preds[0, :, 1] > 0.5)[0]
    subjects = []
    for i in start:
        j = end[end >= i]
        if len(j) > 0:
            j = j[0]
            subjects.append((i.item(), j.item()))

    if subjects:
        spoes = []
        seq_output = seq_output.repeat([len(subjects)] + [1]*(len(seq_output.shape)-1))
        subjects = torch.tensor(subjects, dtype=torch.long, device=device)
        object_preds = train_model.predict_object([seq_output, subjects])

        object_preds[:, [0, -1]] *= 0
        for subject, object_pred in zip(subjects, object_preds):
            start = torch.where(object_pred[:, :, 0] > 0.6)
            end = torch.where(object_pred[:, :, 1] > 0.5)
            for _start, predicate1 in zip(*start):
                for _end, predicate2 in zip(*end):
                    if _start <= _end and predicate1 == predicate2:
                        spoes.append((
                            (mapping[subject[0]][0], mapping[subject[1]][-1]),
                            predicate1.item(),
                            (mapping[_start][0], mapping[_end][-1])
                        ))
                        break
        return [(text[s[0]:s[1]+1], id2predicate[p], text[o[0]:o[1]+1]) for s, p, o in spoes]
    else:
        return []

def evaluate(data):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    with open('checkpoint_merged811/bert_casrel_pred_0.7.json', 'w', encoding='utf-8') as f:
        for d in data:
            if 'spo_list' not in d:
                continue
            R = set([SPO((spo[0], spo[1], spo[2])) for spo in extract_spoes(d)])
            T = set([SPO((t['subject'], t['predicate'], t['object'])) for t in d['spo_list']])
            X += len(R & T)
            Y += len(R)
            Z += len(T)
            result = json.dumps({
                'text': d['text'],
                'spo_list': list(T),
                'spo_list_pred': list(R),
                'new': list(R - T),
                'lack': list(T - R),
            }, ensure_ascii=False, indent=4)
            f.write(result + '\n')
    precision = X / Y if Y else 0
    recall = X / Z if Z else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    return f1, precision, recall

class SPO(tuple):
    def __init__(self, spo):
        self.spox = (
            tuple(tokenizer.tokenize(spo[0])),
            spo[1],
            tuple(tokenizer.tokenize(spo[2])),
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox

if __name__ == '__main__':
    train_model = Model(co_layers=1).to(device)
    optimizer = optim.AdamW(train_model.parameters(), 1e-5)
    train_model.compile(loss=BCELoss(reduction='none'), optimizer=optimizer)
    evaluator = Evaluator()
    train_model.fit(
        train_dataloader,
        steps_per_epoch=None,
        epochs=50,
        callbacks=[evaluator]
    )