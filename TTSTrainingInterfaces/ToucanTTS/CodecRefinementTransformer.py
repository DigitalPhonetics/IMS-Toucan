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

        self.backtranslation_heads = torch.nn.ModuleList()
        self.hierarchical_classifier = torch.nn.ModuleList()
        self.padding_id = self.codebook_size + 5
        for head in range(self.num_codebooks):
            self.backtranslation_heads.append(torch.nn.Embedding(num_embeddings=self.padding_id + 1, embedding_dim=backtranslation_dim, padding_idx=self.padding_id))
            self.hierarchical_classifier.append(torch.nn.Linear(num_codebooks * backtranslation_dim + head * backtranslation_dim, self.codebook_size))

        self.criterion = MaskedLanguageModellingObjective()
        for backtranslation_head in self.backtranslation_heads:
            torch.nn.init.normal_(backtranslation_head.weight, mean=0, std=attention_dimension ** -0.5)

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

        # TODO do some iterative language modeling on the tokens using the hierarchical classifier from toucan to map to one-hot-encoded indexes again
        # TODO inspiration can probably be found in https://github.com/suno-ai/bark/blob/main/bark/model_fine.py

        refined_index_sequence_one_hot_encoded = None

        if is_inference:
            return refined_index_sequence_one_hot_encoded
        else:
            return self.criterion(refined_index_sequence_one_hot_encoded, gold_index_sequence)

    def indexes_per_codebook_to_stacked_embedding_vector(self, index_sequence):
        index_sequence = index_sequence.transpose(0, 1)
        continuous_frame_sequences = list()
        for codebook_id, codebook_sequence in enumerate(index_sequence):
            continuous_frame_sequences.append(self.backtranslation_heads[codebook_id](codebook_sequence))
        stacked_embedding_vector = torch.cat(continuous_frame_sequences, dim=-1)
        return stacked_embedding_vector


class MaskedLanguageModellingObjective(torch.nn.Module):

    def __init__(self):
        super().__init__()
        pass

    def forward(self, index_sequence, gold_index_sequence):
        pass


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
    batch_of_indexes_per_codebook = list()
    for predicted_indexes_one_hot in batch_of_indexes_one_hot_per_codebook:
        predicted_lookup_index = torch.argmax(predicted_indexes_one_hot, dim=-2)
        batch_of_indexes_per_codebook.append(predicted_lookup_index)
    batch_of_indexes = torch.stack(batch_of_indexes_per_codebook, dim=-1).transpose(1, 2)

    # refine the output of the TTS with the Language Model
    refiner = CodecRefinementTransformer()
    loss = refiner(index_sequence=batch_of_indexes, padding_mask=make_pad_mask(dummy_speech_lens), is_inference=False, speaker_embedding=dummy_utterance_embed, gold_index_sequence=None)
    refined_indexes = refiner(index_sequence=batch_of_indexes[1].unsqueeze(0), is_inference=True, speaker_embedding=dummy_utterance_embed, gold_index_sequence=None)
