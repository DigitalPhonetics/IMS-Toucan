import torch


class CodecRefinementTransformer(torch.nn.Module):

    def __init__(self,
                 attention_dimension=512,
                 utt_embed_dim=512,
                 use_conditional_layernorm_embedding_integration=False,
                 num_codebooks=4,
                 codebook_size=1024,
                 backtranslation_dim=8
                 ):
        super().__init__()

        self.attention_dimension = attention_dimension
        self.multispeaker_model = utt_embed_dim is not None
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size

        self.input_embeddings = torch.nn.ModuleList()
        self.backtranslation_heads = torch.nn.ModuleList()
        self.hierarchical_classifier = torch.nn.ModuleList()
        self.padding_id = self.codebook_size + 5
        for head in range(self.num_codebooks):
            self.input_embeddings.append(torch.nn.Embedding(num_embeddings=self.padding_id + 1, embedding_dim=backtranslation_dim, padding_idx=self.padding_id))
            self.backtranslation_heads.append(torch.nn.Embedding(num_embeddings=self.padding_id + 1, embedding_dim=backtranslation_dim, padding_idx=self.padding_id))
            self.hierarchical_classifier.append(torch.nn.Linear(num_codebooks * backtranslation_dim + head * backtranslation_dim, self.codebook_size))

        self.criterion = MaskedRefinementObjective()
        for backtranslation_head in self.backtranslation_heads:
            torch.nn.init.normal_(backtranslation_head.weight, mean=0, std=attention_dimension ** -0.5)
        for input_embedding in self.input_embeddings:
            torch.nn.init.normal_(input_embedding.weight, mean=0, std=attention_dimension ** -0.5)

    def forward(self, index_sequence, is_inference, speaker_embedding, padding_mask=None, gold_index_sequence=None):
        """
        index_sequence: [batch, codebook_index, time_steps] a sequence of indexes that come from an argmax of the previous prediction layer.
        is_inference: boolean flag that indicates whether to return the masked language modelling loss or the refined sequence
        speaker_embedding: [batch, speaker_embed_dim]
        padding_mask: [batch, time_steps] a mask that is True for all time steps that are padding and should not be considered and False everywhere else.

        return: loss if is_inference is false, otherwise [batch, codebook_index, time_steps] a sequence of indexes with the same shape and same interpretation, refined through iterative masked language modelling.
        """

        if not is_inference:
            index_sequence_padding_accounted = index_sequence.masked_fill(mask=padding_mask.unsqueeze(1), value=self.padding_id)
        else:
            index_sequence_padding_accounted = index_sequence  # in the case of inference, there is no padding

        sequence_of_continuous_tokens = self.indexes_per_codebook_to_stacked_embedding_vector(index_sequence_padding_accounted)  # return [batch, time_steps, num_codebooks x backtranslation_dim]

        masked_sequence = self.randomly_mask_sequence(sequence_of_continuous_tokens)
        reconstructed_sequence = self.reconstruct_masked_sequence(masked_sequence)

        # TODO 1. Teile von sequence_of_continuous_tokens maskieren
        #      2. Durch einen Transformer durch passen

        # TODO inspiration for the transformer can probably be found in https://github.com/suno-ai/bark/blob/main/bark/model_fine.py

        predicted_indexes_one_hot = list()
        backtranslated_indexes = list()
        for head_index, classifier_head in enumerate(self.hierarchical_classifier):
            # each codebook considers all previous codebooks.
            predicted_indexes_one_hot.append(classifier_head(torch.cat([reconstructed_sequence] + backtranslated_indexes, dim=2)))
            predicted_lookup_index = torch.argmax(predicted_indexes_one_hot[-1], dim=-1)
            backtranslation = self.backtranslation_heads[head_index](predicted_lookup_index)
            if len(backtranslation.size()) == 1:
                backtranslation = backtranslation.unsqueeze(0)
            backtranslated_indexes.append(backtranslation)
        indexes = torch.cat(predicted_indexes_one_hot, dim=2)
        # [Batch, Sequence, Hidden]
        indexes = indexes.view(reconstructed_sequence.size(0), reconstructed_sequence.size(1), self.num_codebooks, self.codebook_size)
        # [Batch, Sequence, Codebook, Classes]
        indexes = indexes.transpose(1, 2)
        # [Batch, Codebook, Sequence, Classes]
        indexes = indexes.transpose(2, 3)
        # [Batch, Codebook, Classes, Sequence]
        indexes = indexes.transpose(0, 1)
        # [Codebook, Batch, Classes, Sequence]

        refined_index_sequence_one_hot_encoded = indexes

        if is_inference:
            return refined_index_sequence_one_hot_encoded
        else:
            return self.criterion(predicted_features=refined_index_sequence_one_hot_encoded, gold_features=gold_index_sequence, non_pad_mask=~padding_mask)

    def randomly_mask_sequence(self):
        # TODO
        return None

    def reconstruct_masked_sequence(self):
        # TODO
        return None

    def indexes_per_codebook_to_stacked_embedding_vector(self, index_sequence):
        index_sequence = index_sequence.transpose(0, 1)
        continuous_frame_sequences = list()
        for codebook_id, codebook_sequence in enumerate(index_sequence):
            continuous_frame_sequences.append(self.backtranslation_heads[codebook_id](codebook_sequence))
        stacked_embedding_vector = torch.cat(continuous_frame_sequences, dim=-1)
        return stacked_embedding_vector


class MaskedRefinementObjective(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.classification_loss = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, predicted_features, gold_features, non_pad_mask):
        ce = list()
        for one_hot_pred, one_hot_target in zip(predicted_features, gold_features.transpose(0, 1).transpose(2, 3)):
            # we iterate over codebooks
            ce.append(self.classification_loss(one_hot_pred, one_hot_target))
        distance_loss = torch.stack(ce).sum(0)
        # make weighted mask and apply it
        out_masks = non_pad_mask.unsqueeze(-1).to(gold_features.device)
        out_masks = torch.nn.functional.pad(out_masks.transpose(1, 2), [0, gold_features.size(2) - out_masks.size(1), 0, 0, 0, 0], value=False).transpose(1, 2)
        out_weights = out_masks.float() / out_masks.sum(dim=1, keepdim=True).float()
        out_weights /= gold_features.size(0) * gold_features.size(-1)
        # apply weight
        distance_loss = distance_loss.mul(out_weights.squeeze()).masked_select(out_masks.squeeze()).sum()

        return distance_loss


def one_hot_sequence_to_token_sequence(batch_of_indexes_one_hot_per_codebook):
    return torch.argmax(batch_of_indexes_one_hot_per_codebook, dim=-2).transpose(0, 1)


if __name__ == '__main__':
    from TTSTrainingInterfaces.ToucanTTS.ToucanTTS import ToucanTTS
    from Utility.utils import make_pad_mask

    # prepare dummy inputs
    num_codebooks = 4
    dummy_text_batch = torch.randint(low=0, high=2, size=[3, 3, 62]).float()  # [Batch, Sequence Length, Features per Phone]
    dummy_text_lens = torch.LongTensor([2, 3, 3])
    dummy_speech_batch = torch.randn([3, num_codebooks, 30, 1024])  # [Batch, Sequence Length, Spectrogram Buckets]
    dummy_speech_lens = torch.LongTensor([10, 30, 20])
    dummy_durations = torch.LongTensor([[10, 0, 0], [10, 15, 5], [5, 5, 10]])
    dummy_pitch = torch.Tensor([[[1.0], [0.], [0.]], [[1.1], [1.2], [0.8]], [[1.1], [1.2], [0.8]]])
    dummy_energy = torch.Tensor([[[1.0], [1.3], [0.]], [[1.1], [1.4], [0.8]], [[1.1], [1.2], [0.8]]])
    dummy_utterance_embed = torch.randn([3, 512])  # [Batch, Dimensions of Speaker Embedding]
    dummy_language_id = torch.LongTensor([5, 3, 2]).unsqueeze(1)

    # run TTS on pseudo inputs
    batch_of_indexes_one_hot_per_codebook, _, _, _ = ToucanTTS(num_codebooks=num_codebooks)._forward(dummy_text_batch,
                                                                                                     dummy_text_lens,
                                                                                                     dummy_speech_batch,
                                                                                                     dummy_speech_lens,
                                                                                                     dummy_durations,
                                                                                                     dummy_pitch,
                                                                                                     dummy_energy,
                                                                                                     utterance_embedding=dummy_utterance_embed,
                                                                                                     lang_ids=dummy_language_id)

    # reformat outputs to be a token sequence
    batch_of_indexes = one_hot_sequence_to_token_sequence(batch_of_indexes_one_hot_per_codebook)

    # refine the output of the TTS with the Language Model
    refiner = CodecRefinementTransformer()

    loss = refiner(index_sequence=batch_of_indexes, padding_mask=make_pad_mask(dummy_speech_lens), is_inference=False, speaker_embedding=dummy_utterance_embed, gold_index_sequence=None)
    print(loss)

    refined_indexes = refiner(index_sequence=batch_of_indexes[1].unsqueeze(0), is_inference=True, speaker_embedding=dummy_utterance_embed, gold_index_sequence=None)
    print(refined_indexes.shape)
    refined_indexes = one_hot_sequence_to_token_sequence(refined_indexes)
    refined_indexes = refiner(index_sequence=refined_indexes, is_inference=True, speaker_embedding=dummy_utterance_embed, gold_index_sequence=None)
    print(refined_indexes.shape)
    refined_indexes = one_hot_sequence_to_token_sequence(refined_indexes)
    refined_indexes = refiner(index_sequence=refined_indexes, is_inference=True, speaker_embedding=dummy_utterance_embed, gold_index_sequence=None)
    print(refined_indexes.shape)
    refined_indexes = one_hot_sequence_to_token_sequence(refined_indexes)
    refined_indexes = refiner(index_sequence=refined_indexes, is_inference=True, speaker_embedding=dummy_utterance_embed, gold_index_sequence=None)
    print(refined_indexes.shape)
