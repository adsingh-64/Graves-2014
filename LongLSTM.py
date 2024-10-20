"""
Single hidden layer 1000 unit LSTM following Graves 2014
Peephole connections omitted
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(64)

# Load Tiny Shakespeare Data
# ------------------------------------------------------------------------------------
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
# -------------------------------------------------------------------------------------

# Model Variables
# ---------------------------------------------------------------------------------
vocab_size = len(itos)
block_size = 64
d_model = 24  # embedding dimension
n_hidden = 1000
batch_size = 32
num_layers = 1
# ------------------------------------------------------------------------------------


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    if split == "train":
        data = train_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
    else:
        data = val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


class InputLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.C = nn.Embedding(vocab_size, d_model)

        # Combine weights for all gates
        self.U = nn.Linear(d_model, 4 * n_hidden, bias=True)  # For x_t
        self.W = nn.Linear(n_hidden, 4 * n_hidden, bias=False)  # For h_t

        self.h_0 = nn.Parameter(0.1 * torch.ones(1, n_hidden, device=device))
        self.c_0 = nn.Parameter(0.1 * torch.ones(1, n_hidden, device=device))

    def forward(self, xb):
        B, T = xb.shape
        emb = self.C(xb)  # [B, T, d_model]
        h_t = self.h_0.repeat(B, 1)  # [B, n_hidden]
        c_t = self.c_0.repeat(B, 1)  # [B, n_hidden]
        h_all = torch.zeros(B, T, n_hidden, device=device)  # Preallocate memory

        for t in range(T):
            x_t = emb[:, t, :]  # [B, d_model]

            # Compute gates
            gates = self.U(x_t) + self.W(h_t)  # [B, 4 * n_hidden]
            i_t, f_t, g_t, o_t = gates.chunk(
                4, dim=1
            )  # Split into 4 gates - g_t is the candidate cell memory

            # Apply activations
            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)
            g_t = torch.tanh(g_t)
            o_t = torch.sigmoid(o_t)

            # Update cell state and hidden state
            c_t = f_t * c_t + i_t * g_t  # [B, n_hidden]
            h_t = o_t * torch.tanh(c_t)  # [B, n_hidden]

            h_all[:, t, :] = h_t  # Store hidden state
        return emb, h_all


class HiddenLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.U = nn.Linear(n_hidden, 4 * n_hidden, bias=True)  # For x_t
        self.W = nn.Linear(n_hidden, 4 * n_hidden, bias=False)  # For h_t
        self.V = nn.Linear(d_model, n_hidden, bias=True)

        self.h_0 = nn.Parameter(0.1 * torch.ones(1, n_hidden, device=device))
        self.c_0 = nn.Parameter(0.1 * torch.ones(1, n_hidden, device=device))

    def forward(self, emb, h_past):
        B, T, _ = h_past.shape
        h_t = self.h_0.repeat(B, 1)  # [B, n_hidden]
        c_t = self.c_0.repeat(B, 1)
        h_all = torch.zeros(B, T, n_hidden, device=device)
        for t in range(T):
            e_t = emb[:, t, :]  # [B, d_model]
            x_t = h_past[:, t, :]  # [B, n_hidden]

            gates = self.U(x_t) + self.W(h_t)  # [B, 4 * n_hidden]
            i_t, f_t, g_t, o_t = gates.chunk(
                4, dim=1
            )  # Split into 4 gates - g_t is the candidate cell memory

            # Apply activations
            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)
            g_t = torch.tanh(g_t)
            o_t = torch.sigmoid(o_t)

            # Update cell state and hidden state
            c_t = f_t * c_t + i_t * g_t  # [B, n_hidden]
            h_t = o_t * torch.tanh(c_t)  # [B, n_hidden]

            # Skip connection from embedding
            h_t = h_t + torch.tanh(self.V(e_t))

            h_all[:, t, :] = h_t  # Store hidden state
        return h_all


class OutputLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # Output weight matrices for each layer
        self.W_hy = nn.ModuleList(
            [nn.Linear(n_hidden, vocab_size, bias=False) for _ in range(num_layers)]
        )
        # Shared bias term
        self.b_y = nn.Parameter(torch.zeros(vocab_size, device=device))

    def forward(self, h_all, targets=None):
        """
        h_all: Tensor of shape [B, T, n_hidden, num_layers]
        """
        # Initialize logits with the bias term
        logits = self.b_y  # Shape: [vocab_size]
        # Sum over contributions from each layer -- perhaps can tensorize
        for n in range(num_layers):
            h_n = h_all[:, :, :, n]  # Shape: [B, T, n_hidden]
            logits_n = self.W_hy[n](h_n)  # Shape: [B, T, vocab_size]
            logits = logits_n + logits  # Broadcasting over [B, T]
        return logits


# Model Definition
# -----------------------------------------------------------------------------------
class CharRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = InputLayer()
        self.hidden_layers = nn.ModuleList(
            [HiddenLayer() for _ in range(num_layers - 1)]
        )
        self.output_layer = OutputLayer()

    def forward(self, xb, targets=None):
        emb, h_input = self.input_layer(xb)
        h_all_layers = [torch.unsqueeze(h_input, dim=-1)]
        h_past = h_input
        for hidden_layer in self.hidden_layers:
            h_current = hidden_layer(emb, h_past)
            h_all_layers.append(torch.unsqueeze(h_current, -1))
            h_past = h_current
        h_all = torch.cat(h_all_layers, dim=-1)
        logits = self.output_layer(h_all)
        loss = None
        if targets is not None:
            logits = logits.view(-1, logits.shape[-1])
            targets = targets.view(-1)
            loss = F.cross_entropy(logits, targets)
        return logits, loss


# Load Model
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
model = CharRNN().to(device)

# Optimization
import math

max_lr = 1e-3  # Maximum learning rate
min_lr = max_lr * 0.1  # Minimum learning rate (10% of max_lr)
warmup_steps = 1000  # Number of warmup steps
max_steps = 10000  # Total number of training steps


def get_lr(it):
    if it < warmup_steps:
        # Linear warmup
        return max_lr * (it + 1) / warmup_steps
    elif it > max_steps:
        # After max_steps, keep learning rate constant at min_lr
        return min_lr
    else:
        # Cosine decay
        decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)


optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)
for step in range(max_steps):
    optimizer.zero_grad()
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    if step % 1000 == 0:
        print(f"Step {step} | Training loss: {loss.item()}")


# Inference
def evaluate(model):
    model.eval()
    total_loss = 0.0
    num_batches = 1000  # GPU memory -- average loss over 1000 batches from val data

    with torch.no_grad():  # Disable gradient computation
        for _ in range(num_batches):
            xb, yb = get_batch("val")
            logits, loss = model(xb, yb)
            total_loss += loss.item()

    average_loss = total_loss / num_batches
    print(f"Validation Loss: {average_loss}")
    model.train()


evaluate(model)


# Generation
def generate(model, idx, max_new_tokens):
    """
    Very slow, processes entire sequence every time
    """
    model.eval()
    idx = idx.to(next(model.parameters()).device)
    h_t = None
    for _ in range(max_new_tokens):
        logits, _ = model(idx)
        logits = logits[:, -1, :]  # Get logits for the last time step
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


# Sampling Time!
# -------------------------------------------------------------------------------
context = torch.tensor([[stoi[c] for c in "Once upon a time"]], dtype=torch.long)
generated_idx = generate(model, context, max_new_tokens=300)
generated_text = decode(generated_idx[0].tolist())
print(generated_text)

"""
ubuntu@129-213-23-213:~$ python LongLSTM.py
Step 0 | Training loss: 4.186212062835693
Step 1000 | Training loss: 1.5537413358688354
Step 2000 | Training loss: 1.4098496437072754
Step 3000 | Training loss: 1.2567373514175415
Step 4000 | Training loss: 1.0556023120880127
Step 5000 | Training loss: 1.061169147491455
Step 6000 | Training loss: 0.9470760822296143
Step 7000 | Training loss: 0.8427449464797974
Step 8000 | Training loss: 0.7760028839111328
Step 9000 | Training loss: 0.6951309442520142
Validation Loss: 1.7877245633602141

Once upon a time shape in this place:
The bagempty beauty can haste;
For I can no more me wrong, and by and bow
The causer of the glasses of the people,
Repoar'd a moiety. And you a body
To fight against that two more?

COMINIUS:
He would not shun me from the single.

ANTIGONUS:
If it prevented
Desire the wrong wer
"""
