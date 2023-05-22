import torch


class DualSelfTeachingLoss:
    def __init__(self, beta=0.5, gamma=0.5, sigma=0.2, temperature=1.0, n_passages_per_query=8):
        """
        n_passages_per_query: number of passages per query (positive passage + hard negative passages), in our case is 8 since we have 7 hard negatives
        """
        self.beta = beta
        self.gamma = gamma
        self.sigma = sigma
        self.temperature = temperature
        self.n_passages_per_query = n_passages_per_query

        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")
        self.KL = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def __call__(self, q_reps: torch.Tensor, p_reps: torch.Tensor, typo_q_reps: torch.Tensor, effective_bsz: int):
        """
        q_reps: original queries representations with shape (queries_num, embedding_dim)
        p_reps: passages representations with shape (passages_num, embedding_dim)
        typo_q_reps: misspelled queries representations with shape (queries_num x typos_num, embedding_dim)
        effective_bsz: batch size
        """
        ################################################## Similarity Score Calculation ##################################################
        # Passage retrieval (query-to-passages)
        # Original queries
        qp_scores = torch.matmul(q_reps, p_reps.transpose(0, 1))                    # (queries_num, passages_num)
        qp_scores = qp_scores.view(effective_bsz, -1)                               # (queries_num, passages_num)
        queries_num, passages_num = qp_scores.shape
        # Misspelled queries
        typo_qp_scores = torch.matmul(typo_q_reps, p_reps.transpose(0, 1))          # (queries_num x typos_num, passages_num)
        typo_qp_scores = typo_qp_scores.view(queries_num, -1, passages_num)         # (queries_num, typos_num, passages_num)
        typos_num = typo_qp_scores.shape[1]

        # Query retrieval (passage-to-queries)
        # Original queries
        pq_scores = torch.matmul(p_reps, q_reps.transpose(0, 1))                    # (passages_num, queries_num)
        pq_scores = pq_scores.view(passages_num, queries_num)                       # (passages_num, queries_num)
        # Remove hard negative passages
        pos_pq_scores = pq_scores[torch.arange(0, pq_scores.shape[0], self.n_passages_per_query), :]  # (pos_passages_num, queries_num)
        # Misspelled queries
        typo_pq_scores = torch.matmul(p_reps, typo_q_reps.transpose(0, 1))          # (passages_num, queries_num x typos_num)
        typo_pq_scores = typo_pq_scores.view(passages_num, queries_num, typos_num)  # (passages_num, queries_num, typos_num)
        ######################################################## Loss Calculation ########################################################
        # Dual Cross-Entropy Loss
        # Passage Retrieval
        qp_target = torch.arange(
            qp_scores.size(0),
            device=qp_scores.device,
            dtype=torch.long
        )
        qp_target = qp_target * self.n_passages_per_query
        qp_ce_loss = self.cross_entropy(qp_scores, qp_target)
        # Query Retrieval
        pq_target = torch.arange(
            pos_pq_scores.size(0),
            device=pos_pq_scores.device,
            dtype=torch.long
        )
        pq_ce_loss = self.cross_entropy(pos_pq_scores, pq_target)
        ce_loss = (1 - self.gamma) * qp_ce_loss + self.gamma * pq_ce_loss

        # Dual KL-Divergence Loss
        # Passage Retrieval Consistency
        qp_kl_loss = 0.0
        for i in range(typos_num):
            qp_kl_loss += self.KL(
                self.log_softmax(typo_qp_scores[:, i, :]),
                self.log_softmax(qp_scores.detach() / self.temperature)
            ) / typos_num
        # Query Retrieval Consistency
        pq_kl_loss = 0.0
        for i in range(typos_num):
            pq_kl_loss += self.KL(
                self.log_softmax(typo_pq_scores[:, :, i]),
                self.log_softmax(pq_scores.detach() / self.temperature)
            ) / typos_num
        kl_loss = (1 - self.sigma) * qp_kl_loss + self.sigma * pq_kl_loss

        # Dual Self-Teaching Loss
        loss = (1 - self.beta) * ce_loss + self.beta * kl_loss
        return loss, qp_scores