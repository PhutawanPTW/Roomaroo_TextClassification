## BERT Deep Classification on TripAdvisor/Dorm Reviews (Colab Guide)

This guide reproduces a deep BERT pipeline like your earlier notebook, adapted to your dataset. It includes cleaning, tokenization, train/val/test splits, training, and evaluation with classification report and confusion matrix. It does NOT save the model and does NOT use FastAPI.

Use it directly in Google Colab by running cells in order.

### 1) Runtime & GPU
```python
import torch
print("GPU Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```

### 2) Install dependencies
```python
!pip -q install transformers==4.44.2 accelerate==0.34.2
!pip -q install datasets==2.21.0 nltk==3.9.1 scikit-learn==1.5.2 seaborn==0.13.2
```

### 3) Imports
```python
import re
import nltk
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
```

### 4) Load data
- Option A: Mount Drive and read CSV from your project folder (recommended if file is in Drive)
```python
from google.colab import drive
drive.mount('/content/drive')

# Update the path if your CSV is in another location
csv_path = '/content/drive/MyDrive/Roomaroo_TextClassification/Data/tripadvisor_hotel_reviews.csv'
# Alternative: dorm CSV
# csv_path = '/content/drive/MyDrive/Roomaroo_TextClassification/Data/dorm_reviews.csv'

df = pd.read_csv(csv_path)
df.head()
```

- Option B: Upload the CSV manually
```python
from google.colab import files
uploaded = files.upload()  # choose your CSV file
import io
fn = next(iter(uploaded))
df = pd.read_csv(io.BytesIO(uploaded[fn]))
df.head()
```

Expected columns:
- `Review`: text content
- `Rating`: review score in 1..5 (will be shifted to 0..4 labels for training)

### 5) Text cleaning (Thai + English)
This cell reuses your Thai cleaning style (PyThaiNLP tokenization, emoji removal, Thai stopwords minus important words, punctuation handling, and normalization). It also falls back to simple English cleaning for non-Thai text.
```python
# For English stopwords (optional fallback)
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words_en = set(stopwords.words('english'))

# Thai cleaning utilities
!pip -q install pythainlp==5.0.4 emoji==2.12.1
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_stopwords, thai_negations
import emoji as emoji_mod
import string

def clean_text_th(text: str) -> str:
    if not isinstance(text, str):
        return ""
    thai_stop_words = list(thai_stopwords())
    important_words = [
        "ไม่", "ไม่มี", "ไม่ได้", "ไม่ค่อย", "ไม่เคย", "ห้าม", "ยังไม่", "ไม่ยอม",
        "ดี", "ดีมาก", "สะอาด", "เย็น", "กว้าง", "ใหม่", "สวย", "น่าอยู่", "สบาย", "ชอบ",
        "ประทับใจ", "สะดวก", "ปลอดภัย", "คุ้ม", "คุ้มค่า", "เงียบ", "ครบ", "พอใจ", "เร็ว",
        "โอเค", "โอเคเลย", "เยี่ยม", "ถูกใจ", "ทำเลดี", "ใกล้", "ครบครัน",
        "แย่", "ไม่ดี", "เหม็น", "ร้อน", "แคบ", "เก่า", "สกปรก", "พัง", "เสียงดัง", "แพง",
        "ไม่ชอบ", "แออัด", "รก", "อันตราย", "ช้า", "ผิดหวัง", "ห่วย", "เฟล", "กาก",
        "ไม่คุ้ม", "ไกล", "รั่ว", "ทรุด", "ทรุดโทรม", "เสื่อม",
        "มาก", "สุดๆ", "เยอะ", "น้อย", "ที่สุด", "หลาย", "ทุก", "เกิน", "จัด", "โคตร",
        "มากๆ", "สุดยอด", "ธรรมดา", "พอใช้", "บ่อย", "ตลอด", "เวอร์",
        "แอร์", "น้ำ", "ไฟ", "ห้องน้ำ", "เตียง", "ฝักบัว", "เน็ต", "ไวไฟ", "ไฟฟ้า", "ประปา",
        "เฟอร์", "ลิฟต์", "ที่จอด", "จอดรถ", "ซักผ้า", "ตู้เย็น", "ทีวี", "จาน", "ไมโครเวฟ",
        "เตา", "น้ำอุ่น", "ผ้าปู", "โต๊ะ", "เก้าอี้", "ตู้", "ชั้นวาง", "ปลั๊ก", "สัญญาณ",
        "เสียง", "มด", "แมลง", "แมลงสาบ", "หนู", "ยุง", "ฝุ่น", "กลิ่น", "เพื่อนบ้าน",
        "ข้างห้อง", "ข้างนอก", "ถนน", "ทางเดิน", "ลานจอด", "ชั้นบน", "บันได", "กำแพง",
        "ดูแล", "บริการ", "ซ่อม", "แก้ไข", "จัดการ", "พนักงาน", "แม่บ้าน", "รปภ", "เจ้าของ",
        "นิติ", "กฎ", "ระเบียบ", "ค่าเช่า", "ค่าไฟ", "ค่าน้ำ", "ค่าส่วนกลาง", "มัดจำ", "ประกัน",
        "สัญญา", "ฝากของ", "รับพัสดุ", "คีย์การ์ด", "ล็อค", "รอนาน", "ไม่มาดู", "ไม่ซ่อม",
        "แต่", "แต่ว่า", "ถึงแม้", "อย่างไรก็ตาม", "เพราะ", "เพราะว่า", "เนื่องจาก", "คือ", "ก็คือ",
        "ส่วน", "นอกจากนี้", "ที่จริง", "จริงๆ", "ก็", "แม้", "ที่", "ตอนแรก", "พอดี", "แล้วก็"
    ]
    custom_stop_words = [w for w in thai_stop_words if w not in important_words]
    punct = string.punctuation.replace('!', '').replace('?', '').replace('.', '')
    text = emoji_mod.replace_emoji(text, replace="")
    text = ''.join(ch if ch not in punct else ' ' for ch in text)
    text = re.sub(r'([ก-๙a-zA-Z])\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    words = word_tokenize(text, engine='newmm')
    words = [w for w in words if w not in custom_stop_words]
    return ' '.join(words)

def looks_thai(text: str) -> bool:
    return bool(re.search(r'[ก-๙]', text or ''))

def clean_text(text: str) -> str:
    # Route Thai to Thai cleaner, else basic English cleaning
    if looks_thai(text):
        return clean_text_th(text)
    if not isinstance(text, str) or text.strip() == "":
        return ""
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    words = [w for w in text.split() if w not in stop_words_en]
    return ' '.join(words)

# Ensure required columns exist
assert 'Review' in df.columns, "Column 'Review' not found in CSV"
assert 'Rating' in df.columns, "Column 'Rating' not found in CSV"

# Map ratings 1..5 -> 0..4 for model labels
df = df.copy()
df['Rating'] = df['Rating'].astype(int)
df['Label'] = df['Rating'] - 1
df['Cleaned_Review'] = df['Review'].astype(str).apply(clean_text)

df[['Review', 'Cleaned_Review', 'Rating', 'Label']].head()
```

### 6) Tokenizer and encoding
```python
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

MAX_LEN = 256

def encode_text(text: str):
    return tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=MAX_LEN,
        return_tensors='pt'
    )

# Pre-encode all rows into a column (optional; you can also encode on-the-fly in Dataset)
df['Encoded'] = df['Cleaned_Review'].apply(encode_text)
```

### 7) Dataset & DataLoaders
```python
class ReviewDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx: int):
        encoded = self.frame.iloc[idx]['Encoded']
        label = int(self.frame.iloc[idx]['Label'])
        item = {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
        }
        return item

full_dataset = ReviewDataset(df)

train_size = int(0.6 * len(full_dataset))
val_size = int(0.2 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])

BATCH_SIZE = 8
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"Total: {len(full_dataset)} | Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
print(f"Batches -> Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")
```

### 8) Model & Optimizer
```python
num_labels = 5  # ratings 0..4
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
```

### 9) Training loop
```python
def train(model, train_loader, val_loader, optimizer, epochs=3):
    train_losses = []
    train_accuracies = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        seen = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            seen += labels.size(0)

        avg_loss = epoch_loss / max(1, len(train_loader))
        accuracy = correct / max(1, seen)
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, acc={accuracy:.4f}")

    return train_losses, train_accuracies

train_losses, train_accuracies = train(model, train_loader, val_loader, optimizer, epochs=5)
```

### 10) Evaluation
```python
def evaluate(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    print("Classification Report (labels 0..4):\n")
    print(classification_report(all_labels, all_preds, digits=4))
    return all_labels, all_preds

true_labels, pred_labels = evaluate(model, test_loader)
```

### 11) Confusion Matrix
```python
classes = ['0', '1', '2', '3', '4']
cm = confusion_matrix(true_labels, pred_labels)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
```

### 12) Plot training curves (optional)
```python
import numpy as np
epochs_axis = np.arange(1, len(train_losses)+1)
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(epochs_axis, train_losses, marker='o', label='Loss')
ax.plot(epochs_axis, train_accuracies, marker='o', label='Accuracy')
ax.set_xlabel('Epoch')
ax.set_title('Training Loss/Accuracy')
ax.set_ylim(0, 1)
ax.legend()
plt.show()
```

### 13) Predict single review (no saving)
```python
def predict_review(text: str):
    model.eval()
    encoded = tokenizer(text, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt')
    with torch.no_grad():
        outputs = model(encoded['input_ids'].to(device), attention_mask=encoded['attention_mask'].to(device))
        logits = outputs.logits
        pred = int(torch.argmax(logits, dim=1).item())  # 0..4
        conf = torch.softmax(logits, dim=1).cpu().numpy()[0].tolist()
    return pred + 1, conf  # convert back to 1..5 for readability

example_pred, confs = predict_review("Great location and friendly staff, room was clean and cozy!")
print("Predicted rating:", example_pred)
print("Confidences:", [f"{c:.4f}" for c in confs])
```

Notes:
- Labels use 0..4 internally and are shown as 1..5 in `predict_review` output.
- No model saving is performed.


