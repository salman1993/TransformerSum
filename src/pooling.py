import torch
from torch import nn


class Pooling(nn.Module):
    """Methods to obtains sentence embeddings from word vectors. Multiple methods
    can be specificed and their results will be concatenated together.

    Arguments:
        sent_rep_tokens (bool, optional): Use the sentence representation token
                as sentence embeddings. Default is True.
        mean_tokens (bool, optional): Take the mean of all the token vectors in
        each sentence. Default is False.
    """

    def __init__(self):
        super(Pooling, self).__init__()

    def forward(
        self, word_vectors=None, sent_rep_token_ids=None, sent_rep_mask=None,
    ):
        r"""Forward pass of the Pooling nn.Module.

        Args:
            word_vectors (torch.Tensor, optional): Vectors representing words created by
                a ``word_embedding_model``. Defaults to None.
            sent_rep_token_ids (torch.Tensor, optional): See :meth:`extractive.ExtractiveSummarizer.forward`.
                Defaults to None.
            sent_rep_mask (torch.Tensor, optional): See :meth:`extractive.ExtractiveSummarizer.forward`.
                Defaults to None

        Returns:
            tuple: (output_vector, output_mask) Contains the sentence scores and mask as
            ``torch.Tensor``\ s. The mask is either the ``sent_rep_mask`` or ``sent_lengths_mask``
            depending on the pooling mode used during model initialization.
        """
        output_vectors = []
        output_masks = []

        sents_vec = word_vectors[
            torch.arange(word_vectors.size(0)).unsqueeze(1), sent_rep_token_ids
        ]
        sents_vec = sents_vec * sent_rep_mask[:, :, None].float()
        output_vectors.append(sents_vec)
        output_masks.append(sent_rep_mask)

        output_vector = torch.cat(output_vectors, 1)
        output_mask = torch.cat(output_masks, 1)

        return output_vector, output_mask
