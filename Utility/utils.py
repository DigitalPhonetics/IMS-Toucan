"""
Taken from ESPNet, modified by Florian Lux
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing
from matplotlib.lines import Line2D

import Architectures.GeneralLayers.ConditionalLayerNorm
from Preprocessing.TextFrontend import ArticulatoryCombinedTextFrontend
from Preprocessing.TextFrontend import get_language_id


def integrate_with_utt_embed(hs, utt_embeddings, projection, embedding_training):
    if not embedding_training:
        # concat hidden states with spk embeds and then apply projection
        embeddings_expanded = torch.nn.functional.normalize(utt_embeddings).unsqueeze(1).expand(-1, hs.size(1), -1)
        hs = projection(torch.cat([hs, embeddings_expanded], dim=-1))
    else:
        # in this case we don't want to normalize the embeddings to not impair the gradient flow
        hs = projection(hs, utt_embeddings)
    return hs


def float2pcm(sig, dtype='int16'):
    """
    https://gist.github.com/HudsonHuang/fbdf8e9af7993fe2a91620d3fb86a182
    """
    sig = np.asarray(sig)
    if sig.dtype.kind != 'f':
        raise TypeError("'sig' must be a float array")
    dtype = np.dtype(dtype)
    if dtype.kind not in 'iu':
        raise TypeError("'dtype' must be an integer type")
    i = np.iinfo(dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig * abs_max + offset).clip(i.min, i.max).astype(dtype)


def make_estimated_durations_usable_for_inference(xs, offset=1.0):
    return torch.clamp(torch.round(xs.exp() - offset), min=0).long()


def cut_to_multiple_of_n(x, n=4, return_diff=False, seq_dim=1):
    max_frames = x.shape[seq_dim] // n * n
    if return_diff:
        return x[:, :max_frames], x.shape[seq_dim] - max_frames
    return x[:, :max_frames]


def pad_to_multiple_of_n(x, n=4, seq_dim=1, pad_value=0):
    max_frames = ((x.shape[seq_dim] // n) + 1) * n
    diff = max_frames - x.shape[seq_dim]
    return torch.nn.functional.pad(x, [0, 0, 0, diff, 0, 0], mode="constant", value=pad_value)


@torch.inference_mode()
def plot_progress_spec_toucantts(net,
                                 device,
                                 save_dir,
                                 step,
                                 lang,
                                 default_emb,
                                 run_stochastic):
    tf = ArticulatoryCombinedTextFrontend(language=lang)
    sentence = tf.get_example_sentence(lang=lang)
    if sentence is None:
        return None
    phoneme_vector = tf.string_to_tensor(sentence).squeeze(0).to(device)
    mel, durations, pitch, energy = net.inference(text=phoneme_vector,
                                                  return_duration_pitch_energy=True,
                                                  utterance_embedding=default_emb,
                                                  lang_id=get_language_id(lang).to(device),
                                                  run_stochastic=run_stochastic)

    plot_code_spec(pitch, energy, sentence, durations, mel, os.path.join(save_dir, "visualization"), tf, step)
    return os.path.join(os.path.join(save_dir, "visualization"), f"{step}.png")


def plot_code_spec(pitch, energy, sentence, durations, mel, save_path, tf, step):
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(9, 8))

    expanded_pitch = list()
    expanded_energy = list()
    for p, e, d in zip(pitch.cpu().squeeze().numpy(), energy.cpu().squeeze().numpy(), durations.cpu().numpy()):
        for _ in range(d):
            expanded_energy.append(e)
            expanded_pitch.append(p)
    pitch = expanded_pitch
    energy = expanded_energy

    spec_plot_axis = ax[1]
    pitch_and_energy_axis = ax[0]

    spec_plot_axis.imshow(mel.cpu().numpy(), origin="lower", cmap='GnBu')
    pitch_and_energy_axis.yaxis.set_visible(False)
    pitch_and_energy_axis.xaxis.set_visible(False)
    spec_plot_axis.yaxis.set_visible(False)
    duration_splits, label_positions = cumsum_durations(durations.cpu().numpy())
    spec_plot_axis.xaxis.grid(True, which='minor')
    spec_plot_axis.set_xticks(label_positions, minor=False)
    phones = tf.get_phone_string(sentence, for_plot_labels=True)
    spec_plot_axis.set_xticklabels(phones)
    word_boundaries = list()
    for label_index, phone in enumerate(phones):
        if phone == "|":
            word_boundaries.append(label_positions[label_index])
    try:
        prev_word_boundary = 0
        word_label_positions = list()
        for word_boundary in word_boundaries:
            word_label_positions.append((word_boundary + prev_word_boundary) / 2)
            prev_word_boundary = word_boundary
        word_label_positions.append((duration_splits[-1] + prev_word_boundary) / 2)

        secondary_ax = spec_plot_axis.secondary_xaxis('bottom')
        secondary_ax.tick_params(axis="x", direction="out", pad=24)
        secondary_ax.set_xticks(word_label_positions, minor=False)
        secondary_ax.set_xticklabels(sentence.split())
        secondary_ax.tick_params(axis='x', colors='orange')
        secondary_ax.xaxis.label.set_color('orange')
    except ValueError:
        spec_plot_axis.set_title(sentence)
    except IndexError:
        spec_plot_axis.set_title(sentence)

    spec_plot_axis.vlines(x=duration_splits, colors="green", linestyles="solid", ymin=0, ymax=15, linewidth=1.0)
    spec_plot_axis.vlines(x=word_boundaries, colors="orange", linestyles="solid", ymin=0, ymax=15, linewidth=2.0)

    pitch_and_energy_axis.plot(pitch, color="blue")
    pitch_and_energy_axis.plot(energy, color="green")

    spec_plot_axis.set_aspect("auto")
    pitch_and_energy_axis.set_aspect("auto")

    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=.95, wspace=0.0, hspace=0.0)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{step}.png"), dpi=100)
    plt.clf()
    plt.close()


def plot_spec_tensor(spec, save_path, name, title=None):
    fig, spec_plot_axis = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
    spec_plot_axis.imshow(spec.detach().cpu().numpy(), origin="lower", cmap='GnBu')
    spec_plot_axis.yaxis.set_visible(False)
    spec_plot_axis.set_aspect("auto")
    if title is not None:
        spec_plot_axis.set_title(title)
    plt.subplots_adjust(left=0.05, bottom=0.1, right=0.95, top=.95 if title is None else .85, wspace=0.0, hspace=0.0)
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"{name}.png"), dpi=100)
    plt.clf()
    plt.close()


def cumsum_durations(durations):
    out = [0]
    for duration in durations:
        out.append(duration + out[-1])
    centers = list()
    for index, _ in enumerate(out):
        if index + 1 < len(out):
            centers.append((out[index] + out[index + 1]) // 2)
    return out, centers


def delete_old_checkpoints(checkpoint_dir, keep=5):
    checkpoint_list = list()
    for el in os.listdir(checkpoint_dir):
        if el.endswith(".pt"):
            try:
                checkpoint_list.append(int(el.replace("checkpoint_", "").replace(".pt", "")))
            except ValueError:
                pass
    if len(checkpoint_list) <= keep:
        return
    else:
        checkpoint_list.sort(reverse=False)
        checkpoints_to_delete = [os.path.join(checkpoint_dir, "checkpoint_{}.pt".format(step)) for step in
                                 checkpoint_list[:-keep]]
        for old_checkpoint in checkpoints_to_delete:
            os.remove(os.path.join(old_checkpoint))


def plot_grad_flow(named_parameters):
    """
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function after loss.backwards() and unscaling as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
    """
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Gradient")
    plt.title("Gradient Flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.show()


def get_most_recent_checkpoint(checkpoint_dir, verbose=True):
    checkpoint_list = list()
    for el in os.listdir(checkpoint_dir):
        if el.endswith(".pt") and el != "best.pt" and el != "embedding_function.pt":
            try:
                checkpoint_list.append(int(el.split(".")[0].split("_")[1]))
            except ValueError:
                pass
    if len(checkpoint_list) == 0:
        print("No previous checkpoints found, cannot reload.")
        return None
    checkpoint_list.sort(reverse=True)
    if verbose:
        print("Reloading checkpoint_{}.pt".format(checkpoint_list[0]))
    return os.path.join(checkpoint_dir, "checkpoint_{}.pt".format(checkpoint_list[0]))


def make_pad_mask(lengths, xs=None, length_dim=-1, device=None):
    """
    Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if xs is None:
        maxlen = int(max(lengths))
    else:
        maxlen = xs.size(length_dim)

    if device is not None:
        seq_range = torch.arange(0, maxlen, dtype=torch.int64, device=device)
    else:
        seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(slice(None) if i in (0, length_dim) else None for i in range(xs.dim()))
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


def make_non_pad_mask(lengths, xs=None, length_dim=-1, device=None):
    """
    Make mask tensor containing indices of non-padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        ByteTensor: mask tensor containing indices of padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    """
    return ~make_pad_mask(lengths, xs, length_dim, device=device)


def initialize(model, init):
    """
    Initialize weights of a neural network module.

    Parameters are initialized using the given method or distribution.

    Args:
        model: Target.
        init: Method of initialization.
    """

    # weight init
    for p in model.parameters():
        if p.dim() > 1:
            if init == "xavier_uniform":
                torch.nn.init.xavier_uniform_(p.data)
            elif init == "xavier_normal":
                torch.nn.init.xavier_normal_(p.data)
            elif init == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
            elif init == "kaiming_normal":
                torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
            else:
                raise ValueError("Unknown initialization: " + init)
    # bias init
    for p in model.parameters():
        if p.dim() == 1:
            p.data.zero_()

    # reset some modules with default init
    for m in model.modules():
        if isinstance(m, (torch.nn.Embedding,
                          torch.nn.LayerNorm,
                          Architectures.GeneralLayers.ConditionalLayerNorm.ConditionalLayerNorm,
                          Architectures.GeneralLayers.ConditionalLayerNorm.SequentialWrappableConditionalLayerNorm
                          )):
            m.reset_parameters()


def pad_list(xs, pad_value):
    """
    Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad


def curve_smoother(curve):
    if len(curve) < 3:
        return curve
    new_curve = list()
    for index in range(len(curve)):
        if curve[index] != 0:
            current_value = curve[index]
            if index != len(curve) - 1:
                if curve[index + 1] != 0:
                    next_value = curve[index + 1]
                else:
                    next_value = curve[index]
            if index != 0:
                if curve[index - 1] != 0:
                    prev_value = curve[index - 1]
                else:
                    prev_value = curve[index]
            else:
                prev_value = curve[index]
            smooth_value = (current_value * 3 + prev_value + next_value) / 5
            new_curve.append(smooth_value)
        else:
            new_curve.append(0)
    return new_curve


def remove_elements(tensor, indexes):
    # Create a boolean mask where True represents the elements to keep
    print("\n\n\n")
    print(tensor.shape)
    print(indexes)
    mask = torch.ones(tensor.size(0), dtype=torch.bool)
    mask[indexes] = False

    # Use the mask to select the elements to keep
    result = tensor[mask, :]
    print(result.shape)
    return result


def load_json_from_path(path):
    with open(path, "r", encoding="utf8") as f:
        obj = json.loads(f.read())

    return obj

if __name__ == '__main__':
    data = np.random.randn(50)
    plt.plot(data, color="b")
    smooth = curve_smoother(data)
    plt.plot(smooth, color="g")
    plt.show()
