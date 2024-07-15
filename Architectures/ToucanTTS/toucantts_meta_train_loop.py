import torch.multiprocessing
import wandb
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from Architectures.ToucanTTS.LanguageEmbeddingSpaceStructureLoss import LanguageEmbeddingSpaceStructureLoss
from Preprocessing.AudioPreprocessor import AudioPreprocessor
from Preprocessing.EnCodecAudioPreprocessor import CodecAudioPreprocessor
from Utility.WarmupScheduler import ToucanWarmupScheduler as WarmupScheduler
from Utility.path_to_transcript_dicts import *
from Utility.utils import delete_old_checkpoints
from Utility.utils import get_most_recent_checkpoint
from Utility.utils import plot_progress_spec_toucantts
from run_weight_averaging import average_checkpoints
from run_weight_averaging import get_n_recent_checkpoints_paths
from run_weight_averaging import load_net_toucan
from run_weight_averaging import save_model_for_use


def collate_and_pad(batch):
    # text, text_len, speech, speech_len, durations, energy, pitch, utterance condition, language_id
    return (pad_sequence([datapoint[0] for datapoint in batch], batch_first=True).float(),
            torch.stack([datapoint[1] for datapoint in batch]).squeeze(1),
            [datapoint[2] for datapoint in batch],
            torch.stack([datapoint[3] for datapoint in batch]).squeeze(1),
            pad_sequence([datapoint[4].squeeze() for datapoint in batch], batch_first=True),
            pad_sequence([datapoint[5].squeeze() for datapoint in batch], batch_first=True),
            pad_sequence([datapoint[6].squeeze() for datapoint in batch], batch_first=True),
            None,
            torch.stack([datapoint[8] for datapoint in batch]),
            torch.stack([datapoint[9] for datapoint in batch]))


def train_loop(net,
               datasets,
               device,
               save_directory,
               batch_size,
               steps,
               steps_per_checkpoint,
               lr,
               path_to_checkpoint,
               lang,
               resume,
               fine_tune,
               warmup_steps,
               use_wandb,
               train_samplers,
               gpu_count,
               use_less_loss,
               freeze_lang_embs
               ):
    """
    see train loop arbiter for explanations of the arguments
    """
    net = net.to(device)
    if steps_per_checkpoint is None:
        steps_per_checkpoint = 1000
    if steps % steps_per_checkpoint == 0:
        steps = steps + 1
    else:
        steps = steps + ((steps_per_checkpoint + 1) - (steps % steps_per_checkpoint))  # making sure to stop at the closest point that makes sense to the specified stopping point
    if steps < warmup_steps * 5:
        print(f"too much warmup given the amount of steps, reducing warmup to {warmup_steps} steps")
        warmup_steps = steps // 5

    if use_less_loss:
        less_loss = LanguageEmbeddingSpaceStructureLoss()
        pretrained_language_codes = [
            "eng", "deu", "fra", "spa", "cmn", "por", "pol", "ita", "nld", "ell", "fin", "vie", "jpn", "rus", "hun", "asm", "ben", "brx", "dgo", "guj", "hin", "kan", "kas", "knn", "mai", "mal", "mni", "mar", "nep", "ory", "pan", "san", "sat", "snd", "tam", "tel", "urd", "bem", "swh", "amh", "wol", "chv", "iba", "jav", "fon", "hau", "lbb",
            "kik", "lin", "lug", "luo", "sxb", "yor", "nya", "loz", "toi", "afr", "arb", "ast", "azj", "bel", "bul", "bos", "cat", "ceb", "sdh", "ces", "cym", "dan", "ekk", "pes", "fil", "gle", "glg", "heb", "hrv", "hye", "ind", "ibo", "isl", "kat", "kam", "kea", "kaz", "khm", "kor", "ltz", "lao", "lit", "lvs", "mri", "mkd", "xng", "zsm",
            "mlt", "oci", "pst", "ron", "slk", "slv", "sna", "som", "srp", "swe", "tgk", "tur", "ukr", "umb", "uzn", "bhd", "kfs", "gbk", "bgc", "xnr", "kfx", "mjl", "bfz", "acf", "bss", "inb", "nca", "quh", "wap", "acr", "bus", "dgr", "maz", "nch", "qul", "tav", "wmw", "acu", "byr", "dik", "iou", "mbb", "ncj", "qvc", "tbc", "xed", "agd",
            "bzh", "djk", "ipi", "mbc", "ncl", "qve", "tbg", "xon", "agg", "bzj", "dop", "jac", "mbh", "ncu", "qvh", "tbl", "xtd", "agn", "caa", "jic", "mbj", "ndj", "qvm", "tbz", "xtm", "agr", "cab", "emp", "jiv", "mbt", "nfa", "qvn", "tca", "yaa", "agu", "cap", "jvn", "mca", "ngp", "qvs", "tcs", "yad", "aia", "car", "ese", "mcb", "ngu",
            "qvw", "yal", "cax", "kaq", "mcd", "nhe", "qvz", "tee", "ycn", "ake", "cbc", "far", "mco", "qwh", "yka", "alp", "cbi", "kdc", "mcp", "nhu", "qxh", "ame", "cbr", "gai", "kde", "mcq", "nhw", "qxn", "tew", "yre", "amf", "cbs", "gam", "kdl", "mdy", "nhy", "qxo", "tfr", "yva", "amk", "cbt", "geb", "kek", "med", "nin", "rai", "zaa",
            "apb", "cbu", "glk", "ken", "mee", "nko", "rgu", "zab", "apr", "cbv", "meq", "tgo", "zac", "arl", "cco", "gng", "kje", "met", "nlg", "rop", "tgp", "zad", "grc", "klv", "mgh", "nnq", "rro", "zai", "ata", "cek", "gub", "kmu", "mib", "noa", "ruf", "tna", "zam", "atb", "cgc", "guh", "kne", "mie", "not", "rug", "tnk", "zao", "atg",
            "chf", "knf", "mih", "npl", "tnn", "zar", "awb", "chz", "gum", "knj", "mil", "sab", "tnp", "zas", "cjo", "guo", "ksr", "mio", "obo", "seh", "toc", "zav", "azg", "cle", "gux", "kue", "mit", "omw", "sey", "tos", "zaw", "azz", "cme", "gvc", "kvn", "miz", "ood", "sgb", "tpi", "zca", "bao", "cni", "gwi", "kwd", "mkl", "shp", "tpt",
            "zga", "bba", "cnl", "gym", "kwf", "mkn", "ote", "sja", "trc", "ziw", "bbb", "cnt", "gyr", "kwi", "mop", "otq", "snn", "ttc", "zlm", "cof", "hat", "kyc", "mox", "pab", "snp", "tte", "zos", "bgt", "con", "kyf", "mpm", "pad", "tue", "zpc", "bjr", "cot", "kyg", "mpp", "soy", "tuf", "zpl", "bjv", "cpa", "kyq", "mpx", "pao", "tuo",
            "zpm", "bjz", "cpb", "hlt", "kyz", "mqb", "pib", "spp", "zpo", "bkd", "cpu", "hns", "lac", "mqj", "pir", "spy", "txq", "zpu", "blz", "crn", "hto", "lat", "msy", "pjt", "sri", "txu", "zpz", "bmr", "cso", "hub", "lex", "mto", "pls", "srm", "udu", "ztq", "bmu", "ctu", "lgl", "muy", "poi", "srn", "zty", "bnp", "cuc", "lid", "mxb",
            "stp", "upv", "zyp", "boa", "cui", "huu", "mxq", "sus", "ura", "boj", "cuk", "huv", "llg", "mxt", "poy", "suz", "urb", "box", "cwe", "hvn", "prf", "urt", "bpr", "cya", "ign", "lww", "myk", "ptu", "usp", "bps", "daa", "ikk", "maj", "myy", "vid", "bqc", "dah", "nab", "qub", "tac", "bqp", "ded", "imo", "maq", "nas", "quf", "taj",
            "vmy"
        ]
        pretrained_language_ids = list()  # an alternative to the valid_language_ids
        for language_code in pretrained_language_codes:
            pretrained_language_ids.append(less_loss.iso_codes_to_ids[language_code])

        # there are 7233 language IDs, but there are a few illegal ones: "ajp", "ajt", "en-sc", "en-us", "fr-be", "fr-sw", "lak", "lno", "nul", "pii", "plj", "pt-br", "slq", "smd", "snb", "spa-lat", "tpw", "vi-ctr", "vi-so", "wya", "zua"
        valid_language_ids = list(less_loss.ids_to_iso_codes.keys())
        for illegal_lang in ["ajp", "ajt", "en-sc", "en-us", "fr-be", "fr-sw", "lak", "lno", "nul", "pii", "plj", "pt-br", "slq", "smd", "snb", "spa-lat", "tpw", "vi-ctr", "vi-so", "wya", "zua"]:
            remove_id = less_loss.iso_codes_to_ids[illegal_lang]
            valid_language_ids.remove(remove_id)

    if isinstance(net, torch.nn.parallel.DistributedDataParallel):
        model = net.module
    else:
        model = net

    torch.multiprocessing.set_sharing_strategy('file_system')
    train_loaders = list()
    train_iters = list()
    ap = CodecAudioPreprocessor(input_sr=-1, device=device)
    spec_extractor = AudioPreprocessor(input_sr=16000, output_sr=16000, device=device)

    for dataset, sampler in zip(datasets, train_samplers):
        batch_sampler_train = torch.utils.data.BatchSampler(sampler, 1, drop_last=True)
        train_loaders.append(DataLoader(dataset=dataset,
                                        batch_sampler=batch_sampler_train,
                                        num_workers=0,  # has to be 0, otherwise copies of the dataset are created, which is not feasible for large scale trainings. This is not optimal for small trainings, but necessary for scalability.
                                        pin_memory=True,
                                        prefetch_factor=None,
                                        collate_fn=lambda x: x[0]))
        train_iters.append(iter(train_loaders[-1]))

    # embedding training is not supported here
    optimizer = torch.optim.RAdam(model.parameters(), lr=lr)

    scheduler = WarmupScheduler(optimizer, peak_lr=lr, warmup_steps=warmup_steps, max_steps=steps)

    steps_run_previously = 0
    regression_losses_total = list()
    stochastic_losses_total = list()
    duration_losses_total = list()
    pitch_losses_total = list()
    energy_losses_total = list()

    if resume:
        path_to_checkpoint = get_most_recent_checkpoint(checkpoint_dir=save_directory)
    if path_to_checkpoint is not None:
        check_dict = torch.load(path_to_checkpoint, map_location=device)
        if freeze_lang_embs:
            filtered_state_dict = {}
            for name, param in check_dict["model"].items():
                if name in model.state_dict():
                    if param.size() == model.state_dict()[name].size():
                        filtered_state_dict[name] = param
                        print(f"Loading parameter {name}")
            model.load_state_dict(filtered_state_dict, strict=False)
            model.encoder.language_embedding.weight.requires_grad = False  # and we never reset that.
        else:
            model.load_state_dict(check_dict["model"])
        if not fine_tune and not freeze_lang_embs:
            optimizer.load_state_dict(check_dict["optimizer"])
            scheduler.load_state_dict(check_dict["scheduler"])
            steps_run_previously = check_dict["step_counter"]
        if steps_run_previously > steps:
            print("Desired steps already reached in loaded checkpoint.")
            return

    net.train()
    # =============================
    # Actual train loop starts here
    # =============================

    if not fine_tune and not resume and use_less_loss and not freeze_lang_embs:
        print("Priming the language embedding space...")
        original_lr = optimizer.param_groups[0]['lr']
        pretraining_lr = 0.001
        for param_group in optimizer.param_groups:
            param_group['lr'] = pretraining_lr
        less_values = list()
        for i in tqdm(range(warmup_steps * 8)):
            language_ids = random.sample(valid_language_ids, batch_size)
            language_embeddings = model.encoder.language_embedding(torch.LongTensor(language_ids).to(device))
            less_value_unsupervised = less_loss(language_ids, language_embeddings)
            less_values.append(less_value_unsupervised.item())
            optimizer.zero_grad()
            less_value_unsupervised.backward()
            optimizer.step()
            if i % warmup_steps // 2 == 0:
                print(sum(less_values) / len(less_values))
                less_values = list()
        for param_group in optimizer.param_groups:
            param_group['lr'] = original_lr

    for step_counter in tqdm(range(steps_run_previously, steps)):
        run_stochastic = step_counter > warmup_steps

        batches = []
        while len(batches) < batch_size:
            for index in random.sample(list(range(len(datasets))), len(datasets)):
                if len(batches) < batch_size:
                    # we get one batch for each task (i.e. language in this case) in a randomized order
                    try:
                        batch = next(train_iters[index])
                        batches.append(batch)
                    except StopIteration:
                        train_iters[index] = iter(train_loaders[index])
                        batch = next(train_iters[index])
                        batches.append(batch)
        batch = collate_and_pad(batches)

        text_tensors = batch[0].to(device)
        text_lengths = batch[1].squeeze().to(device)
        speech_indexes = batch[2]
        speech_lengths = batch[3].squeeze().to(device)
        gold_durations = batch[4].to(device)
        gold_pitch = batch[6].unsqueeze(-1).to(device)  # mind the switched order
        gold_energy = batch[5].unsqueeze(-1).to(device)  # mind the switched order
        lang_ids = batch[8].squeeze(1).to(device)

        speech_batch = list()  # I wish this could be done in the collate function or in the getitem, but using DL models in multiprocessing on very large datasets causes just way too many issues.
        for index, speech_sample in enumerate(speech_indexes):
            with torch.inference_mode():
                wave = ap.indexes_to_audio(speech_sample.int().to(device)).detach()
                mel = spec_extractor.audio_to_mel_spec_tensor(wave, explicit_sampling_rate=16000).transpose(0, 1).detach().cpu()
            gold_speech_sample = mel.clone()
            speech_batch.append(gold_speech_sample)
        gold_speech = pad_sequence(speech_batch, batch_first=True).to(device)

        train_loss = 0.0
        # we sum the loss for each task, as we would do for the
        # second order regular MAML, but we do it only over one
        # step (i.e. iterations of inner loop = 1)

        utterance_embedding = batch[9].to(device)
        regression_loss, stochastic_loss, duration_loss, pitch_loss, energy_loss = net(
            text_tensors=text_tensors,
            text_lengths=text_lengths,
            gold_speech=gold_speech,
            speech_lengths=speech_lengths,
            gold_durations=gold_durations,
            gold_pitch=gold_pitch,
            gold_energy=gold_energy,
            utterance_embedding=utterance_embedding,
            lang_ids=lang_ids,
            return_feats=False,
            run_stochastic=run_stochastic
        )

        # then we directly update our meta-parameters without
        # the need for any task specific parameters

        if torch.isnan(regression_loss) or torch.isnan(duration_loss) or torch.isnan(pitch_loss) or torch.isnan(energy_loss):
            print("One of the losses turned to NaN! Skipping this batch ...")
            continue

        train_loss = train_loss + regression_loss
        train_loss = train_loss + duration_loss
        train_loss = train_loss + pitch_loss
        train_loss = train_loss + energy_loss

        if stochastic_loss is not None:
            if torch.isnan(stochastic_loss) or torch.isinf(stochastic_loss):
                print("Flow loss turned to NaN! Skipping this batch ...")
                continue
            train_loss = train_loss + stochastic_loss
            stochastic_losses_total.append(stochastic_loss.item())
        else:
            stochastic_losses_total.append(0)

        regression_losses_total.append(regression_loss.item())
        duration_losses_total.append(duration_loss.item())
        pitch_losses_total.append(pitch_loss.item())
        energy_losses_total.append(energy_loss.item())

        optimizer.zero_grad()
        if type(train_loss) is float:
            print("There is no loss for this step! Skipping ...")
            continue
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=False)
        optimizer.step()
        scheduler.step()

        if step_counter % steps_per_checkpoint == 0 and step_counter != 0:
            # ==============================
            # Enough steps for some insights
            # ==============================
            if gpu_count > 1:
                rank = int(os.environ["LOCAL_RANK"])
            else:
                rank = 0
            if rank == 0:
                net.eval()
                default_embedding = datasets[0][0][9].to(device)
                print("Reconstruction Loss:    {}".format(round(sum(regression_losses_total) / len(regression_losses_total), 3)))

                print("Steps:                  {}\n".format(step_counter))
                torch.save({
                    "model"       : model.state_dict(),
                    "optimizer"   : optimizer.state_dict(),
                    "scheduler"   : scheduler.state_dict(),
                    "step_counter": step_counter,
                    "default_emb" : default_embedding,
                    "config"      : model.config
                },
                    os.path.join(save_directory, "checkpoint_{}.pt".format(step_counter)))
                delete_old_checkpoints(save_directory, keep=5)

                if use_wandb:
                    wandb.log({
                        "regression_loss": round(sum(regression_losses_total) / len(regression_losses_total), 5),
                        "stochastic_loss": round(sum(stochastic_losses_total) / len(stochastic_losses_total), 5),
                        "duration_loss"  : round(sum(duration_losses_total) / len(duration_losses_total), 5),
                        "pitch_loss"     : round(sum(pitch_losses_total) / len(pitch_losses_total), 5),
                        "energy_loss"    : round(sum(energy_losses_total) / len(energy_losses_total), 5),
                        "learning_rate"  : optimizer.param_groups[0]['lr']
                    }, step=step_counter)

                try:
                    path_to_most_recent_plot = plot_progress_spec_toucantts(model,
                                                                            device,
                                                                            save_dir=save_directory,
                                                                            step=step_counter,
                                                                            lang=lang,
                                                                            default_emb=default_embedding,
                                                                            run_stochastic=run_stochastic)
                    if use_wandb:
                        wandb.log({
                            "progress_plot": wandb.Image(path_to_most_recent_plot)
                        }, step=step_counter)

                except IndexError:
                    print("generating progress plots failed.")

                checkpoint_paths = get_n_recent_checkpoints_paths(checkpoint_dir=save_directory, n=1)
                averaged_model, default_embed = average_checkpoints(checkpoint_paths, load_func=load_net_toucan)
                save_model_for_use(model=averaged_model, default_embed=default_embed, name=os.path.join(save_directory, "best.pt"))

                net.train()

            regression_losses_total = list()
            stochastic_losses_total = list()
            duration_losses_total = list()
            pitch_losses_total = list()
            energy_losses_total = list()
            if gpu_count > 1:
                # just to be extra sure tht all models are synchronous
                torch.distributed.barrier()
                checkpoint_paths = get_n_recent_checkpoints_paths(checkpoint_dir=save_directory, n=1)
                check_dict = torch.load(checkpoint_paths[0], map_location=device)
                model.load_state_dict(check_dict["model"])
                torch.distributed.barrier()
