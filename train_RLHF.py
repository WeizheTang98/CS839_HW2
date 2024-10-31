"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import seaborn as sns
from RLHF import LanguageModelRLHF


from model import GPTConfig, GPT
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from train_Reward_Function import RewardRegressionModel,TextCNN,Seq2SeqRewardModel
from torch.nn.functional import kl_div


# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out-shakespeare-char'
eval_interval = 250
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'shakespeare_char'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 64
# model
n_layer = 16
n_head = 16
n_embd = 128
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line

def init_model(model_args):
    # resume training from a checkpoint.
    ckpt_path = os.path.join('out-shakespeare-char', 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    reward_path = os.path.join(os.path.join('out','model_parameter'),'txtcnn.pth')
    reward_model_state_dict = torch.load(reward_path,map_location=device)
    # reward_model_args = reward_path['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model_thinker = GPT(gptconf)
    model_origin = GPT(gptconf)
    reward_model = TextCNN(vocab_size = 86, embedding_dim = 768, kernel_sizes = [3,5,7,9,11], num_channels = [512,512,512,512,512])
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model_thinker.load_state_dict(state_dict)
    model_origin.load_state_dict(state_dict)
    reward_model.load_state_dict(reward_model_state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    model_origin.eval()
    model_thinker.train()
    reward_model.eval()
    return model_thinker,model_origin,reward_model,iter_num,best_val_loss,checkpoint

model_thinker,model_origin,reward_model,iter_num,best_val_loss,checkpoint = init_model(model_args)







# crop down the model block size if desired, using model surgery
if block_size < model_thinker.config.block_size:
    model_thinker.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value

model_thinker.to(device)
reward_model.to(device)
model_origin.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model_thinker.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = torch.load('out-shakespeare-char/ckpt.pt', map_location=device) # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model_thinker
    model_thinker = torch.compile(model_thinker) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model_thinker = DDP(model_thinker, device_ids=[ddp_local_rank])


model = LanguageModelRLHF(model_thinker,model_origin,reward_model,get_batch)
# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# helps estimate an arbitrarily accurate loss over either split using many batches
# helps estimate an arbitrarily accurate loss over either split using many batches
from collections import Counter
@torch.no_grad()
def estimate_loss_and_perplexity(model):
    out = {}
    perplexity = {}
    model.eval()
    true_labels = []
    predicted_labels = []
    pred_char_count = Counter()  # To count predicted characters
    true_char_count = Counter()  # To count true characters
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            # X.to(device)
            # Y.to(device)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
            if split == "train":
                predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                true = Y.cpu().numpy()
                true_labels.extend(true.flatten())
                predicted_labels.extend(predictions.flatten())
                pred_char_count.update(predictions.flatten())  # Count predicted characters
                true_char_count.update(true.flatten())         # Count true characters

        mean_loss = losses.mean()
        out[split] = mean_loss
        # Calculate perplexity as the exponential of the cross-entropy loss
        perplexity[split] = math.exp(mean_loss)
    model.train()
    return out, perplexity,predicted_labels,true_labels,pred_char_count, true_char_count





@torch.no_grad()
def transfer_dic2list(dictionary):
    dic = dict(dictionary)
    list_dic = [0 for i in range(86)]
    for i in range(86):
        try:
            list_dic[i] = dic[i]
        except KeyError:
            list_dic[i] = 1

    return list_dic


@torch.no_grad()
def kl_divergence(P, Q):
    """
    Calculate the Kullback-Leibler divergence between two probability distributions P and Q.

    Parameters:
    P (np.array): True distribution (e.g., labels).
    Q (np.array): Predicted distribution (e.g., model outputs).

    Returns:
    float: The KL-divergence between P and Q.
    """
    # Ensure P and Q are numpy arrays
    P = np.asarray(P, dtype=np.float32)
    Q = np.asarray(Q, dtype=np.float32)
    P = P / np.sum(P)
    Q = Q / np.sum(Q)
    # Calculate the KL divergence
    return np.sum(P * np.log(P / Q))

class ArgmaxSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits):
        # Perform argmax to get deterministic actions
        return torch.argmax(logits, dim=-1)

    @staticmethod
    def backward(ctx, grad_output):
        # Approximate gradient by passing it through unchanged
        return grad_output.clone(), None





# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, Y = get_batch('train') # fetch the very first batch


t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.model_thinker.module if ddp else model.model_thinker # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process and iter_num - checkpoint['iter_num'] != 0:
        losses, perplexities, predicted_labels, true_labels, pred_char_count, true_char_count = estimate_loss_and_perplexity(model=model.model_thinker)

        pred_list = np.array(transfer_dic2list(pred_char_count))
        true_list = np.array(transfer_dic2list(true_char_count))

        kl_score = kl_divergence(true_list,pred_list)

        # losses = estimate_loss()

        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print(f"step {iter_num}: train perplexity {perplexities['train']:.4f}, val perplexity {perplexities['val']:.4f}")
        print(f"step {iter_num}: pred_train_KL_score {kl_score:.4f}")


        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'after_RLHF.pt'))

        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(12, 10))  # Increase figure size
        sns.heatmap(conf_matrix_normalized, cmap='Blues', cbar=True, xticklabels=True, yticklabels=True,
                    annot=False)  # Remove annotations
        plt.title('Confusion Matrix of Token Predictions after using RLHF')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        save_path = os.path.join('out', 'graph/confusion_matrix'+ '_after_using_RLHF'+'.jpg')
        # save_path = os.path.join('out', 'graph_1_6/confusion_matrix' + '_10000' +'.jpg')
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")


    if iter_num - checkpoint['iter_num'] == 0:
        losses, perplexities, predicted_labels, true_labels, pred_char_count, true_char_count = estimate_loss_and_perplexity(model=model.model_thinker)

        pred_list = np.array(transfer_dic2list(pred_char_count))
        true_list = np.array(transfer_dic2list(true_char_count))

        kl_score = kl_divergence(true_list,pred_list)

        # losses = estimate_loss()
        print("Before RLHF")
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        print(f"step {iter_num}: train perplexity {perplexities['train']:.4f}, val perplexity {perplexities['val']:.4f}")
        print(f"step {iter_num}: pred_train_KL_score {kl_score:.4f}")


        conf_matrix = confusion_matrix(true_labels, predicted_labels)
        conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(12, 10))  # Increase figure size
        sns.heatmap(conf_matrix_normalized, cmap='Blues', cbar=True, xticklabels=True, yticklabels=True,
                    annot=False)  # Remove annotations
        plt.title('Confusion Matrix of Token Predictions Before Using RLHF')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        save_path = os.path.join('out', 'graph/confusion_matrix'+ 'Before_using_RLHF'+'.jpg')
        # save_path = os.path.join('out', 'graph_1_6/confusion_matrix' + '_10000' +'.jpg')
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")



    if eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            # Forward pass through the thinker model
            logits_thinkers, _ = model.model_thinker(X, Y)

            # print(torch.argmax(logits_thinkers,dim=-1))
            # (batch_size, block_size, vocab_size)
            # Convert logits to probabilities
            # probs = torch.softmax(logits_thinkers, dim=-1)
            # # Create a categorical distribution
            # m = torch.distributions.Categorical(probs)
            # # Sample actions (tokens)
            # actions = m.sample()  # (batch_size, block_size)
            # # Ensure actions are of type LongTensor
            # actions = actions.long()
            #
            # # Compute log probabilities of the sampled actions
            # log_probs = m.log_prob(actions)  # (batch_size, block_size)
            # # Sum over sequence length
            # log_probs = log_probs.sum(dim=1)  # (batch_size,)

            actions = ArgmaxSTE.apply(logits_thinkers)
            actions = actions.long()

            # Compute rewards using the reward model
            rewards = model.reward_model(X, actions)  # (batch_size, 1)
            rewards = rewards.squeeze(-1)  # (batch_size,)

            # Compute the policy loss
            # policy_loss = - (log_probs * rewards.detach()).mean()
            policy_loss = - rewards.detach().mean()

            # Compute KL divergence loss
            logits_origins, _ = model.model_origin(X, Y)
            log_probs_thinkers = torch.nn.functional.log_softmax(logits_thinkers, dim=-1)
            probs_origins = torch.nn.functional.softmax(logits_origins, dim=-1)

            # Apply epsilon to avoid zeros
            epsilon = 1e-8
            probs_origins = probs_origins + epsilon
            probs_origins = probs_origins / probs_origins.sum(dim=-1, keepdim=True)

            kl_loss = torch.nn.functional.kl_div(
                log_probs_thinkers,
                probs_origins,
                reduction='batchmean',
                log_target=False
            )

            # Total loss
            loss = policy_loss + kl_loss
            loss = loss / gradient_accumulation_steps

        # Prepare for the next batch
        X, Y = get_batch('train')
        # Backward pass
        scaler.scale(loss).backward()

    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.model_thinker.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
