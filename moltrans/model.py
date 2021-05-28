import math
from rdkit import Chem
from pathlib import Path
from functools import partial

import torch
import torchvision
import torch.nn as nn
import pytorch_lightning as pl
from pl_bolts.metrics import precision_at_k, mean
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR


# ************************************************************************************************
# ***************************************** Util Classes *****************************************
# ************************************************************************************************


class FuncLR(LambdaLR):
    def get_lr(self):
        return [lmbda(self.last_epoch) for lmbda in self.lr_lambdas]


# Use Pytorch implementation but with 'pre-norm' style layer normalisation
class PreNormEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self attention block
        att = self.norm1(src)
        att = self.self_attn(att, att, att, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        att = src + self.dropout1(att)

        # Feedforward block
        out = self.norm2(att)
        out = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = att + self.dropout2(out)
        return out


# Use Pytorch implementation but with 'pre-norm' style layer normalisation
class PreNormDecoderLayer(nn.TransformerDecoderLayer):
    def forward(
        self,
        tgt,
        memory,
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None
    ):
        # Self attention block 
        query = self.norm1(tgt)
        query = self.self_attn(query, query, query, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        query = tgt + self.dropout1(query)

        # Context attention block
        att = self.norm2(query)
        att = self.multihead_attn(att, memory, memory, attn_mask=memory_mask, 
                key_padding_mask=memory_key_padding_mask)[0]
        att = query + self.dropout2(att)

        # Feedforward block
        out = self.norm3(att)
        out = self.linear2(self.dropout(self.activation(self.linear1(out))))
        out = att + self.dropout3(out)
        return out


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.out_dim = 2048

        resnet = torchvision.models.resnext50_32x4d()
        resnet.avgpool = None
        resnet.fc = None
        self.resnet = resnet

    def forward(self, x):
        out = self.resnet.conv1(x)
        out = self.resnet.bn1(out)
        out = self.resnet.relu(out)
        out = self.resnet.maxpool(out)
        out = self.resnet.layer1(out)
        out = self.resnet.layer2(out)
        out = self.resnet.layer3(out)
        out = self.resnet.layer4(out)
        return out


class _AbsTransformerModel(pl.LightningModule):
    def __init__(
        self,
        d_model,
        lr,
        max_seq_len,
        schedule,
        num_steps,
        weight_decay,
        vocab_size="None",
        warm_up_steps=0,
        pad_token_idx=0,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()

        self.d_model = d_model
        self.lr = lr
        self.max_seq_len = max_seq_len
        self.schedule = schedule
        self.num_steps = num_steps
        self.weight_decay = weight_decay
        self.vocab_size = vocab_size
        self.warm_up_steps = warm_up_steps
        self.pad_token_idx = pad_token_idx
        self.dropout = dropout

        if self.schedule == "transformer":
            assert warm_up_steps is not None, "A value for warm_up_steps is required for transformer LR schedule"

        # Additional args passed in to **kwargs in init will also be saved
        self.save_hyperparameters()

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_emb", self._positional_embs())

    def forward(self, x):
        raise NotImplementedError()

    def _calc_loss(self, batch_input, model_output):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        self.train()

        model_output = self.forward(batch)
        loss = self._calc_loss(batch, model_output)

        self.log("train_loss", loss, on_step=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        self.eval()

        model_output = self.forward(batch)
        loss = self._calc_loss(batch, model_output)

        val_outputs = {"val_loss": loss}
        return val_outputs

    def validation_epoch_end(self, outputs):
        avg_outputs = self._avg_dicts(outputs)
        self._log_dict(avg_outputs)

    def configure_optimizers(self):
        params = self.parameters()
        optim = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.999))

        if self.schedule == "const":
            print("Using constant LR schedule.")
            const_sch = FuncLR(optim, lr_lambda=self._const_lr)
            sch = {"scheduler": const_sch, "interval": "step"}

        elif self.schedule == "cycle":
            print("Using cyclical LR schedule.")
            cycle_sch = OneCycleLR(optim, self.lr, total_steps=self.num_steps)
            sch = {"scheduler": cycle_sch, "interval": "step"}

        elif self.schedule == "transformer":
            print("Using original transformer schedule.")
            trans_sch = FuncLR(optim, lr_lambda=self._transformer_lr)
            sch = {"scheduler": trans_sch, "interval": "step"}

        else:
            raise ValueError(f"Unknown schedule {self.schedule}")

        return [optim], [sch]

    def _transformer_lr(self, step):
        mult = self.d_model ** -0.5
        step = 1 if step == 0 else step  # Stop div by zero errors
        lr = min(step ** -0.5, step * (self.warm_up_steps ** -1.5))
        return self.lr * mult * lr

    def _const_lr(self, step):
        if self.warm_up_steps is not None and step < self.warm_up_steps:
            return (self.lr / self.warm_up_steps) * step

        return self.lr

    def _add_pos_embs(self, seq):
        seq_len, _, _ = tuple(seq.size())
        positional_embs = self.pos_emb[:seq_len, :].unsqueeze(0).transpose(0, 1)
        embs = seq + positional_embs
        embs = self.dropout(embs)
        return embs

    def _positional_embs(self):
        encs = torch.tensor([dim / self.d_model for dim in range(0, self.d_model, 2)])
        encs = 10000 ** encs
        encs = [(torch.sin(pos / encs), torch.cos(pos / encs)) for pos in range(self.max_seq_len)]
        encs = [torch.stack(enc, dim=1).flatten()[:self.d_model] for enc in encs]
        encs = torch.stack(encs)
        return encs

    def _generate_square_subsequent_mask(self, sz, device="cpu"):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _avg_dicts(self, colls):
        complete_dict = {key: [] for key, val in colls[0].items()}
        for coll in colls:
            [complete_dict[key].append(coll[key]) for key in complete_dict.keys()]

        avg_dict = {key: sum(l) / len(l) for key, l in complete_dict.items()}
        return avg_dict

    def _log_dict(self, coll):
        for key, val in coll.items():
            self.log(key, val, sync_dist=True)


# ************************************************************************************************
# ******************************************* Encoders *******************************************
# ************************************************************************************************


class BMSEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        d_feedforward,
        num_layers,
        num_heads,
        max_seq_len,
        dropout=0.1,
        activation="gelu"
    ):
        super().__init__()

        self.d_model = d_model
        self.d_feedforward = d_feedforward
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.resnet = ResNet()
        self.fc = nn.Linear(self.resnet.out_dim, d_model)

        enc_norm = nn.LayerNorm(d_model)
        enc_layer = PreNormEncoderLayer(d_model, num_heads, d_feedforward, dropout, activation)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers, norm=enc_norm)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_emb", self._positional_embs())

    def forward(self, imgs):
        batch_size, _, _, _ = tuple(imgs.shape)
        features = self.resnet(imgs)
        features = features.reshape(batch_size, self.resnet.out_dim, -1).permute(2, 0, 1)
        features = self.fc(features)
        features = self._construct_enc_input(features)
        out = self.encoder(features)
        return out

    def _construct_enc_input(self, features):
        src_len, _, _ = tuple(features.shape)
        pos_embs = self.pos_emb[:src_len, :].unsqueeze(0).transpose(0, 1)
        feat = features + pos_embs
        feat = self.dropout(feat)
        return feat

    def _positional_embs(self):
        encs = torch.tensor([dim / self.d_model for dim in range(0, self.d_model, 2)])
        encs = 10000 ** encs
        encs = [(torch.sin(pos / encs), torch.cos(pos / encs)) for pos in range(self.max_seq_len)]
        encs = [torch.stack(enc, dim=1).flatten()[:self.d_model] for enc in encs]
        encs = torch.stack(encs)
        return encs


class MoCoEncoder(_AbsTransformerModel):
    def __init__(
        self,
        d_model,
        d_feedforward,
        num_layers,
        num_heads,
        lr,
        max_seq_len,
        schedule,
        num_steps,
        weight_decay,
        warm_up_steps=None,
        pad_token_idx=0,
        dropout=0.1,
        activation="gelu",
        feat_dim=128,
        dict_size=4096,
        momentum=0.999,
        temp=0.07,
        **kwargs
    ):
        super().__init__(
            d_model,
            lr,
            max_seq_len,
            schedule,
            num_steps,
            weight_decay,
            warm_up_steps=warm_up_steps,
            pad_token_idx=pad_token_idx,
            dropout=dropout,
            d_feedforward=d_feedforward,
            num_layers=num_layers,
            num_heads=num_heads,
            activation=activation,
            feat_dim=feat_dim,
            dict_size=dict_size,
            momentum=momentum,
            temp=temp,
            **kwargs
        )

        self.feat_dim = feat_dim
        self.dict_size = dict_size
        self.momentum = momentum
        self.temp = temp

        self.encoder_q = BMSEncoder(d_model, d_feedforward, num_layers, num_heads, max_seq_len, dropout, activation)
        self.encoder_k = BMSEncoder(d_model, d_feedforward, num_layers, num_heads, max_seq_len, dropout, activation)

        self.enc_q_fc = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, feat_dim))
        self.enc_k_fc = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, feat_dim))

        # Set same params and no grad for key encoder
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Create the queue
        self.register_buffer("queue", torch.randn(feat_dim, dict_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.loss_fn = nn.CrossEntropyLoss()

    def encode(self, imgs):
        return self.encoder_q(imgs)

    def training_step(self, batch, batch_idx):
        output, target = self.forward(*batch)
        loss = self.loss_fn(output.float(), target.long())

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        log = {"train_loss": loss, "train_acc1": acc1, "train_acc5": acc5}
        self.log_dict(log)
        return loss

    def validation_step(self, batch, batch_idx):
        output, target = self.forward(*batch)
        loss = self.loss_fn(output.float(), target.long())

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        results = {"val_loss": loss, "val_acc1": acc1, "val_acc5": acc5}
        return results

    def validation_epoch_end(self, outputs):
        val_loss = mean(outputs, "val_loss")
        val_acc1 = mean(outputs, "val_acc1")
        val_acc5 = mean(outputs, "val_acc5")

        log = {"val_loss": val_loss, "val_acc1": val_acc1, "val_acc5": val_acc5}
        self.log_dict(log)

    def forward(self, img_q, img_k):
        """ Assumes we are training on single GPU
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # Compute query features
        q = self.encoder_q(img_q).mean(dim=0)  # queries: NxC
        q = self.enc_q_fc(q)
        q = nn.functional.normalize(q, dim=1)

        # Compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(img_k).mean(dim=0)
            k = self.enc_k_fc(k)
            k = nn.functional.normalize(k, dim=1)

        # Compute logits
        # positive logits: Nx1
        # negative logits: NxK
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # Apply temperature
        logits /= self.temp

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)

        self._dequeue_and_enqueue(k)

        return logits, labels

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        em = self.momentum
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * em + param_q.data * (1. - em)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.dict_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.dict_size  # move pointer

        self.queue_ptr[0] = ptr


# ************************************************************************************************
# ******************************************* Decoders *******************************************
# ************************************************************************************************


class BMSDecoder(nn.Module):
    def __init__(self, d_model, d_feedforward, num_layers, num_heads, dropout=0.1, activation="gelu"):
        super().__init__()
        dec_norm = nn.LayerNorm(d_model)
        dec_layer = PreNormDecoderLayer(d_model, num_heads, d_feedforward, dropout, activation)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers, norm=dec_norm)

    def forward(self, dec_input, dec_pad_mask, memory):
        seq_len, _, _ = tuple(dec_input.size())
        tgt_mask = self._generate_square_subsequent_mask(seq_len, device=dec_input.device)
        output = self.decoder(dec_input, memory, tgt_key_padding_mask=dec_pad_mask, tgt_mask=tgt_mask)
        return output

    def _generate_square_subsequent_mask(self, sz, device="cpu"):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class BARTModel(_AbsTransformerModel):
    def __init__(
        self,
        d_model,
        d_feedforward,
        num_layers,
        num_heads,
        lr,
        max_seq_len,
        schedule,
        num_steps,
        weight_decay,
        vocab_size,
        warm_up_steps=None,
        pad_token_idx=0,
        dropout=0.1,
        activation="gelu",
        **kwargs
    ):
        super().__init__(
            d_model,
            lr,
            max_seq_len,
            schedule,
            num_steps,
            weight_decay,
            vocab_size=vocab_size,
            warm_up_steps=warm_up_steps,
            pad_token_idx=pad_token_idx,
            dropout=dropout,
            d_feedforward=d_feedforward,
            num_layers=num_layers,
            num_heads=num_heads,
            **kwargs
        )

        enc_norm = nn.LayerNorm(d_model)
        enc_layer = PreNormEncoderLayer(d_model, num_heads, d_feedforward, dropout, activation)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers, norm=enc_norm)

        self.decoder = BMSDecoder(d_model, d_feedforward, num_layers, num_heads, dropout, activation)

        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_idx)
        self.token_fc = nn.Linear(d_model, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_idx)
        self.log_softmax = nn.LogSoftmax(dim=2)

        self._init_params()

    def forward(self, x):
        encoder_input = x["encoder_input"]
        decoder_input = x["decoder_input"]
        encoder_pad_mask = x["encoder_pad_mask"].transpose(0, 1)
        decoder_pad_mask = x["decoder_pad_mask"].transpose(0, 1)

        encoder_embs = self._construct_input(encoder_input)
        memory = self.encoder(encoder_embs, src_key_padding_mask=encoder_pad_mask)
        model_output = self.decode(decoder_input, decoder_pad_mask, memory)
        return model_output

    def decode(self, dec_input, dec_pad_mask, memory):
        decoder_embs = self._construct_input(dec_input)
        model_output = self.decoder(decoder_embs, dec_pad_mask, memory)
        token_output = self.token_fc(model_output)
        return token_output

    def _calc_loss(self, batch_input, model_output):
        target = batch_input["target"]
        target_mask = batch_input["target_mask"]
        seq_len, batch_size = tuple(target.size())

        token_pred = model_output.reshape((seq_len * batch_size, -1)).float()
        loss = self.loss_fn(token_pred, target.reshape(-1)).reshape((seq_len, batch_size))

        inv_target_mask = ~(target_mask > 0)
        num_tokens = inv_target_mask.sum()
        loss = loss.sum() / num_tokens
        return loss

    def _construct_input(self, token_ids):
        token_embs = self.emb(token_ids) * math.sqrt(self.d_model)
        embs = self._add_pos_embs(token_embs)
        return embs


# ***********************************************************************************************
# ****************************************** BMS Model ******************************************
# ***********************************************************************************************


class BMSModel(_AbsTransformerModel):
    def __init__(
        self,
        encoder,
        decoder,
        d_model,
        sampler,
        lr,
        vocab_size,
        max_seq_len,
        schedule,
        num_steps,
        weight_decay,
        warm_up_steps=None,
        pad_token_idx=0,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(
            d_model,
            lr,
            max_seq_len,
            schedule,
            num_steps,
            weight_decay,
            vocab_size=vocab_size,
            warm_up_steps=warm_up_steps,
            pad_token_idx=pad_token_idx,
            dropout=dropout,
            **kwargs
        )

        self.encoder = encoder
        self.decoder = decoder
        self.sampler = sampler

        self.val_sampling_alg = "greedy"
        self.num_beams = 5

        self.connector_fc = nn.Linear(d_model, d_model)
        self.loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=pad_token_idx)
        self.log_softmax = nn.LogSoftmax(dim=2)

        self._init_params()

    def forward(self, x):
        imgs = x["images"]
        decoder_input = x["decoder_input"]
        decoder_pad_mask = x["decoder_pad_mask"].transpose(0 ,1)

        memory = self.encoder.encode(imgs)
        memory = self.connector_fc(memory)
        token_output = self.decoder.decode(decoder_input, decoder_pad_mask, memory)
        return token_output

    def _calc_loss(self, batch_input, model_output):
        """ Calculate the loss for the model

        Args:
            batch_input (dict): Input given to model,
            model_output (torch.Tensor): Output from model

        Returns:
            loss (singleton tensor),
        """

        tokens = batch_input["target"]
        pad_mask = batch_input["target_mask"]
        token_mask_loss = self._calc_mask_loss(model_output, tokens, pad_mask)
        return token_mask_loss

    def _calc_mask_loss(self, token_output, target, target_mask):
        """ Calculate the loss for the token prediction task

        Args:
            token_output (Tensor of shape (seq_len, batch_size, vocab_size)): token output from transformer
            target (Tensor of shape (seq_len, batch_size)): Original (unmasked) SMILES token ids from the tokeniser
            target_mask (Tensor of shape (seq_len, batch_size)): Pad mask for target tokens

        Output:
            loss (singleton Tensor): Loss computed using cross-entropy,
        """

        seq_len, batch_size = tuple(target.size())

        token_pred = token_output.reshape((seq_len * batch_size, -1)).float()
        loss = self.loss_fn(token_pred, target.reshape(-1)).reshape((seq_len, batch_size))

        inv_target_mask = ~(target_mask > 0)
        num_tokens = inv_target_mask.sum()
        loss = loss.sum() / num_tokens

        return loss

    def validation_step(self, batch, batch_idx):
        model_output = self.forward(batch)
        target_inchis = batch["target_string"]

        loss = self._calc_loss(batch, model_output)
        token_acc = self._calc_token_acc(batch, model_output)
        mol_strs, log_lhs = self.sample_molecules(batch, sampling_alg=self.val_sampling_alg)
        metrics = self.sampler.calc_sampling_metrics(mol_strs, target_inchis)

        mol_acc = torch.tensor(metrics["accuracy"], device=loss.device)
        invalid = torch.tensor(metrics["invalid"], device=loss.device)
        lev_dist = torch.tensor(metrics["lev_dist"], device=loss.device)

        # Log for prog bar only
        self.log("lev_dist", lev_dist, prog_bar=True, logger=False, sync_dist=True)

        val_outputs = {
            "val_loss": loss,
            "val_token_acc": token_acc,
            "val_mol_acc": mol_acc,
            "val_invalid": invalid,
            "val_lev_dist": lev_dist
        }
        return val_outputs

    def sample_molecules(self, batch_input, sampling_alg="greedy"):
        self.freeze()

        imgs = batch_input["images"]
        memory = self.encoder.encode(imgs)
        memory = self.connector_fc(memory)

        _, batch_size, _ = tuple(memory.size())
        decode_fn = partial(self._decode_fn, memory=memory)

        if sampling_alg == "greedy":
            mol_strs, log_lhs = self.sampler.greedy_decode(decode_fn, batch_size, memory.device)
        elif sampling_alg == "beam":
            mol_strs, log_lhs = self.sampler.beam_decode(decode_fn, batch_size, memory.device, k=self.num_beams)
        else:
            raise ValueError(f"Unknown sampling algorithm {sampling_alg}")

        self.unfreeze()
        return mol_strs, log_lhs

    def _decode_fn(self, dec_input, dec_pad_mask, memory):
        token_output = self.decoder.decode(dec_input, dec_pad_mask.transpose(0, 1), memory)
        token_probs = self.log_softmax(token_output)
        return token_probs

    def _calc_token_acc(self, batch_input, model_output):
        token_ids = batch_input["target"]
        target_mask = batch_input["target_mask"]

        target_mask = ~(target_mask > 0)
        _, pred_ids = torch.max(model_output.float(), dim=2)
        correct_ids = torch.eq(token_ids, pred_ids)
        correct_ids = correct_ids * target_mask

        num_correct = correct_ids.sum().float()
        total = target_mask.sum().float()

        accuracy = num_correct / total
        return accuracy
