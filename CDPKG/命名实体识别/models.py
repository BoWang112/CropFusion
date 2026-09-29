import torch
import torch.nn as nn
import torch.functional as F
from torchcrf import CRF
from transformers import BertPreTrainedModel, BertModel, RobertaModel

class BERT_BiLSTM_CRF(BertPreTrainedModel):  # 改为继承 BertPreTrainedModel
    def __init__(self, config, model_type='roberta', need_birnn=False, rnn_dim=128):
        super(BERT_BiLSTM_CRF, self).__init__(config)  # 调用 BertPreTrainedModel 的构造函数
        self.num_tags = config.num_labels
        self.model_type = model_type.lower()
        self.need_birnn = need_birnn

        # 新版 transformers 默认自动启用 SDPA 注意力，旧版 RobertaModel 不支持，
        # 在 config 上强制 eager，保证直接实例化也能正常创建
        config.attn_implementation = 'eager'
        if hasattr(config, '_attn_implementation'):
            config._attn_implementation = 'eager'

        # 根据传入的模型类型，选择 BERT 或 RoBERTa 作为底层模型
        if self.model_type == 'roberta':
            self.bert = RobertaModel(config, add_pooling_layer=False)  # 使用 RoBERTa 作为编码器
        else:
            self.bert = BertModel(config, add_pooling_layer=False)     # 使用 BERT 作为编码器（默认）

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        out_dim = config.hidden_size

        if need_birnn:
            self.birnn = nn.LSTM(config.hidden_size, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
            out_dim = rnn_dim * 2

        self.hidden2tag = nn.Linear(out_dim, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # 新版 transformers 的 from_pretrained 会把 checkpoint 中缺失的
        # 新增层（BiLSTM/CRF）权重错误清零，导致训练无法收敛。
        # 因此这里自行加载预训练权重，仅加载 bert 主干部分。
        import os
        pretrained_path = args[0] if args else kwargs.get('pretrained_model_name_or_path')
        config = kwargs.pop('config', None)
        need_birnn = kwargs.pop('need_birnn', False)
        rnn_dim = kwargs.pop('rnn_dim', 128)
        model_type = kwargs.pop('model_type', 'roberta')
        if config is None:
            from transformers import BertConfig
            num_labels = kwargs.pop('num_labels', None)
            if num_labels is not None:
                config = BertConfig.from_pretrained(pretrained_path, num_labels=num_labels)
            else:
                config = BertConfig.from_pretrained(pretrained_path)
        model = cls(config, model_type=model_type, need_birnn=need_birnn, rnn_dim=rnn_dim)

        # 手动加载预训练权重，避免 transformers 将缺失的新增层权重清零
        bin_path = os.path.join(pretrained_path, 'pytorch_model.bin')
        if os.path.exists(bin_path):
            import torch
            sd = torch.load(bin_path, map_location='cpu')
            # checkpoint 权重带 'bert.' 前缀，与模型顶层 state_dict 键一致，
            # 直接用 model.load_state_dict 加载（strict=False 允许缺失/多余键）
            model.load_state_dict(sd, strict=False)
        return model

    def save_pretrained(self, *args, **kwargs):
        # BiLSTM 等模块产生的权重可能为非连续张量，safetensors 保存会报错；
        # 回退到传统的 .bin 保存（与论文原始环境一致）
        kwargs.setdefault('safe_serialization', False)
        # 确保保存的 config 带正确的标签数，避免重载时 num_labels=None
        if self.config.num_labels is None:
            self.config.num_labels = self.num_tags
        return super().save_pretrained(*args, **kwargs)

    def forward(self, input_ids, tags, token_type_ids=None, input_mask=None):
        emissions = self.tag_outputs(input_ids, token_type_ids, input_mask)
        loss = -1 * self.crf(emissions, tags, mask=input_mask.byte())
        return loss

    def tag_outputs(self, input_ids, token_type_ids=None, input_mask=None):
        if self.model_type == 'roberta':
            outputs = self.bert(input_ids, attention_mask=input_mask)
        else:
            outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=input_mask)

        sequence_output = outputs[0]

        if self.need_birnn:
            sequence_output, _ = self.birnn(sequence_output)

        sequence_output = self.dropout(sequence_output)
        emissions = self.hidden2tag(sequence_output)

        return emissions

    def predict(self, input_ids, token_type_ids=None, input_mask=None):
        emissions = self.tag_outputs(input_ids, token_type_ids, input_mask)
        return self.crf.decode(emissions, input_mask.byte())