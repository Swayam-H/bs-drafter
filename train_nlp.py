import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import time

print("--- BRAWL STARS NLP TRANSFORMER (v13 - Advanced BERT Style) ---")

# 1. Load Data (High ELO only)
print("Loading matches.csv...")
rows = []
with open('matches.csv', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(',')
        if len(parts) == 10:
            t, map_name, mode, a1, a2, a3, e1, e2, e3, res = parts
            peak_rank = 4926.0
        elif len(parts) == 11:
            t, pr, map_name, mode, a1, a2, a3, e1, e2, e3, res = parts
            try:
                peak_rank = float(pr)
            except (ValueError, TypeError):
                peak_rank = 4926.0
        else:
            continue
        rows.append([peak_rank, map_name, mode, a1, a2, a3, e1, e2, e3, res])

df = pd.DataFrame(rows, columns=['peak_rank', 'map', 'mode', 'ally_1', 'ally_2', 'ally_3', 'enemy_1', 'enemy_2', 'enemy_3', 'result'])
# Fix drop_duplicates bug to include enemy team in subset
df = df[df['result'] != 'draw'].drop_duplicates(subset=['ally_1', 'ally_2', 'ally_3', 'enemy_1', 'enemy_2', 'enemy_3']).reset_index(drop=True)

# Filter draft logic
def is_valid_draft(row):
    allies = {row.ally_1, row.ally_2, row.ally_3}
    enemies = {row.enemy_1, row.enemy_2, row.enemy_3}
    return len(allies) == 3 and len(enemies) == 3 and not (allies & enemies)

df = df[df.apply(is_valid_draft, axis=1)]

# Only >6000 ELO
df = df[df['peak_rank'] > 6000].reset_index(drop=True)

df['target'] = df['result'].apply(lambda x: 1 if x == 'victory' else 0)
print(f"Dataset ready: {len(df)} matches.")

# Calculate class imbalance for Loss Function
num_pos = df['target'].sum()
num_neg = len(df) - num_pos
pos_weight = num_neg / num_pos
print(f"Pos Weight for BCE: {pos_weight:.4f}")

# 2. Tokenization Vocabulary
all_tokens = set(df['map'].unique()).union(df['mode'].unique())
for col in ['ally_1', 'ally_2', 'ally_3', 'enemy_1', 'enemy_2', 'enemy_3']:
    all_tokens = all_tokens.union(df[col].unique())

token_list = ['<PAD>', '<CLS>'] + sorted(list(all_tokens))
token_to_id = {t: i for i, t in enumerate(token_list)}
VOCAB_SIZE = len(token_list)
print(f"Vocab size: {VOCAB_SIZE}")

# 3. Dataset
class DraftDataset(Dataset):
    def __init__(self, df):
        self.X = []
        self.y = df['target'].values
        
        for row in df.itertuples():
            # [CLS, MAP, MODE, A1, A2, A3, E1, E2, E3]
            seq = [
                token_to_id['<CLS>'],
                token_to_id[row.map],
                token_to_id[row.mode],
                token_to_id[row.ally_1],
                token_to_id[row.ally_2],
                token_to_id[row.ally_3],
                token_to_id[row.enemy_1],
                token_to_id[row.enemy_2],
                token_to_id[row.enemy_3]
            ]
            self.X.append(seq)
        
        self.X = torch.tensor(self.X, dtype=torch.long)
        self.y = torch.tensor(self.y, dtype=torch.float32)
        
        # Segment IDs: 0 for CLS, 1 for context (map/mode), 2 for allies, 3 for enemies
        self.segments = torch.tensor([0, 1, 1, 2, 2, 2, 3, 3, 3], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.segments, self.y[idx]

# Split 80/20
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
train_dataset = DraftDataset(df_train)
test_dataset = DraftDataset(df_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# 4. Model Definition
class DraftTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.segment_emb = nn.Embedding(4, d_model) # 4 segments: CLS, context, ally, enemy
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4,
            batch_first=True,
            dropout=0.15
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc1 = nn.Linear(d_model, 64)
        self.dropout = nn.Dropout(0.15)
        self.fc2 = nn.Linear(64, 1)
        
    def forward(self, x, seg):
        # x: [batch, 9]
        # seg: [9] -> broadcast to [batch, 9]
        seg = seg.unsqueeze(0).expand(x.size(0), -1)
        
        emb = self.token_emb(x) + self.segment_emb(seg) # [batch, 9, d_model]
        
        out = self.transformer(emb) # [batch, 9, d_model]
        
        # Use [CLS] token (index 0) for classification instead of mean pooling
        cls_token_output = out[:, 0, :] # [batch, d_model]
        
        x = torch.relu(self.fc1(cls_token_output))
        x = self.dropout(x)
        logits = self.fc2(x).squeeze(1) # [batch]
        return logits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = DraftTransformer(VOCAB_SIZE).to(device)

# 5. Training Loop
pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

epochs = 20
print("Training started...")
best_auc = 0.0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_b, seg_b, y_b in train_loader:
        X_b, seg_b, y_b = X_b.to(device), seg_b.to(device), y_b.to(device)
        
        optimizer.zero_grad()
        logits = model(X_b, seg_b[0]) 
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    # Evaluate
    model.eval()
    val_loss = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for X_b, seg_b, y_b in test_loader:
            X_b, seg_b, y_b = X_b.to(device), seg_b.to(device), y_b.to(device)
            logits = model(X_b, seg_b[0])
            loss = criterion(logits, y_b)
            val_loss += loss.item()
            
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(y_b.cpu().numpy())
            
    val_loss /= len(test_loader)
    preds_binary = (np.array(all_preds) >= 0.5).astype(int)
    acc = accuracy_score(all_targets, preds_binary)
    auc = roc_auc_score(all_targets, all_preds)
    
    # Step the scheduler based on AUC
    scheduler.step(auc)
    
    print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f} | Val AUC: {auc:.4f}")

print("Training finished!")
print(classification_report(all_targets, preds_binary, target_names=["Defeat", "Victory"]))
