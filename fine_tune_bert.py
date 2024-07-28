import pandas as pd
file='all.csv'
df=pd.read_csv(file)
import pandas as pd

# Assuming df is your DataFrame
result = pd.concat([df[df['label'] == '交通噪音'], df[df['label'] == '工业噪音（包括工地噪音等）'],df[df['label']=='生活噪音'],df[df['label']=='不属于噪音投诉']])

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
# 加载你的数据集
# df = pd.read_csv('path_to_your_dataset.csv')
# 示例数据
df = result

# 划分训练集和测试集
train_df, test_df = train_test_split(df, test_size=0.2,shuffle=True, random_state=42)

class TextDataset(Dataset):
    def __init__(self, tokenizer, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# 检查是否有可用的 CUDA 设备
if torch.cuda.is_available():
    # 指定使用 cuda:1
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# 初始化分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 创建类别到索引的映射
label_to_index = {label: idx for idx, label in enumerate(df['label'].unique())}

# 使用映射来更新标签
train_labels = [label_to_index[label] for label in train_df['label']]
test_labels = [label_to_index[label] for label in test_df['label']]

# 然后在创建数据集时使用这些更新后的标签
train_dataset = TextDataset(tokenizer, train_df['text'].tolist(), train_labels)
test_dataset = TextDataset(tokenizer, test_df['text'].tolist(), test_labels)


# 创建数据集
# train_dataset = TextDataset(tokenizer, train_df['text'].tolist(), train_df['label'].tolist())
# test_dataset = TextDataset(tokenizer, test_df['text'].tolist(), test_df['label'].tolist())

model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=len(df['label'].unique()))
model = model.to(device)

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.evaluate()

# 保存模型到本地
output_dir = "./saved_model"
model.save_pretrained(output_dir)
# pr
