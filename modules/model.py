import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import re

from collections import OrderedDict

class Residual(torch.nn.Module):
    def __init__(self, inp_size):
        super(Residual, self).__init__()
        self.act_1 = torch.nn.PReLU()
        self.act_2 = torch.nn.PReLU()

        self.dense_1 = torch.nn.Linear(inp_size, inp_size)
        self.dense_2 = torch.nn.Linear(inp_size, inp_size)

        torch.nn.init.xavier_normal_(self.dense_1.weight)
        torch.nn.init.constant_(self.dense_1.bias, 0.01)
        torch.nn.init.xavier_normal_(self.dense_2.weight)
        torch.nn.init.constant_(self.dense_2.bias, 0.01)

    def forward(self, x):
        res = self.act_1(self.dense_1(x))
        res = self.act_2(self.dense_2(res))
        res += x
        return res

class FakeDetector(torch.nn.Module):
    def __init__(self, num_labels, dropout_prob):
        super(FakeDetector, self).__init__()

        self.model = AutoModelForTokenClassification.from_pretrained('xlm-roberta-base')

        self.model = self.model.roberta

        self.config = self.model.config

        self.proj = torch.nn.Linear(self.config.hidden_size, 2 * self.config.hidden_size)
        self.residual = Residual(2 * self.config.hidden_size)
        self.prelu = torch.nn.PReLU()
        self.classifier = torch.nn.Linear(2 * self.config.hidden_size, num_labels)

        self.dropout = torch.nn.Dropout(dropout_prob)

        torch.nn.init.xavier_normal_(self.proj.weight)
        torch.nn.init.constant_(self.proj.bias, 0.01)
        torch.nn.init.xavier_normal_(self.classifier.weight)
        torch.nn.init.constant_(self.classifier.bias, 0.01)

    def forward(self, input_ids, attention_mask):
        bert_out = self.model.forward(input_ids, attention_mask)[0][:, 0, :]

        proj_out = self.prelu(self.proj.forward(self.dropout(bert_out)))
        logits = self.residual.forward(self.dropout(proj_out))
        logits = self.classifier.forward(logits)

        return logits # LOSS --> LogLoss


class NNDefaker:

    def __init__(self, weights_path):

        self.model = FakeDetector(1, 0.1)

        sd = torch.load(weights_path, map_location='cpu')
        state_dict = self.model.state_dict()
        state_dict.update(sd)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to('cuda:0')

        self.tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

    def infer_text(self, text_batch):
        ll = []
        for i, t in enumerate(text_batch):
            inp = self.tokenizer(t, add_special_tokens=True, return_tensors='pt', truncation=True)
            if torch.cuda.is_available():
                inp.to('cuda:0')
            with torch.no_grad():
                logits = self.model.forward(inp.input_ids, inp.attention_mask).cpu()
            ll.append(logits)
            probas = logits.squeeze(0).sigmoid()
        ll = torch.cat(ll, 0)
        fake_score = (ll.softmax(0) * ll).sigmoid()
        return fake_score
