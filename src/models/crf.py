from typing import List, Optional

import torch
import torch.nn as nn


class CRF(nn.Module):
    """Conditional random field.

    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.

    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.

    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.


    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.

    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        # 从起始位置到各标签的转移分数
        self.start_transitions = nn.Parameter(torch.empty(num_tags)) # shape: (num_tags,)
        # 从各标签到结束位置的转移分数
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        
        # 从各标签到各标签的转移分数
        '''
        shape: (num_tags, num_tags)
        '''
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags)) # shape: (num_tags, num_tags)
        
        # 参数初始化
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.

        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: Optional[torch.ByteTensor] = None,
            reduction: str = 'sum', # 指定对输出的处理方式
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
                
            reduction: Specifies  the reduction to apply to the output:
                【怎么处理每段路径上的概率分数？】
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.

        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
            
            给tags: 返回 log(P(y|x)),  如果要计算loss，还要取个负数
            
            不给tags: 返回 
        """
        self._validate(emissions, tags=tags, mask=mask) # 检查输入的维度和有效性
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        # 计算分子（路径得分）
        # print(" CRF compute score")
        numerator = self._compute_score(emissions, tags, mask)
        # print("CRF compute score end ~~~")
        
        # shape: (batch_size,)
        # 计算分母（归一化因子）, 用来对上面的路径分数进行归一化
        # print(" CRF compute normalizer start")
        denominator = self._compute_normalizer(emissions, mask)
        # print(" CRF compute normalizer end !!!")
        
        # shape: (batch_size,)
        # 计算对数似然
        # print(" CRF compute llh 对数似然")
        llh = numerator - denominator  # logA-logB = log(A/B)
        
        # 对结果进行归约
        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        return llh.sum() / mask.float().sum()

    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.

        Returns:
            List of list containing the best tag sequence for each batch.
        """
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)
    
    

    
    
    
    
    
    
    
    def _validate(
            self,
            emissions: torch.Tensor,
            tags: Optional[torch.LongTensor] = None,
            mask: Optional[torch.ByteTensor] = None) -> None:
        
        '''
         检查 emission， mask， tags 的维度
        '''
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            
            if not no_empty_seq and not no_empty_seq_bf:
                # 每个batch对应的第一个token mask 必须是1
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(
            self, emissions: torch.Tensor, tags: torch.LongTensor,
            mask: torch.ByteTensor) -> torch.Tensor:
        '''
         计算路径得分
         
         遍历每个时间步， 通过在每轮中累加 emission 和 transition分数 
         
         最终得到每个序列的路径分数 
         
         return score # score.shape = (batch_size, )
        '''
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all() # 所有batch上的第一个token的mask必须为True
        
        assert (tags >= 0).all() and (tags < self.num_tags).all(), \
            f"标签索引越界：有效范围[0,{self.num_tags-1}]，实际范围[{tags.min()}, {tags.max()}]"  

        seq_length, batch_size = tags.shape
        mask = mask.float()
        
        '''
           以下处理序列的第一个时间步
        '''

        # Start transition score and first emission
        # start_transitions.shape: (num_tags,)
        # tags[0]: (batch_size,) 取出每个序列 在第0个时间步 的tag下标
        # score.shape = (batch_size,)
        
        # 假设 tag[0] = [12, 30, 9, 2, 1]
        # start_transitions[[12, 30, 9, 2, 1]] = [0.2, 0.3, 0.1, 0.05, 0.0]
        # 就是通过一个tag下标列表，来获得一个tag转移概率列表，作为初始的转移分数
        score = self.start_transitions[tags[0]] # shape = (batch_size,)
        
        '''
         score 这么赋值的含义：
            初始化每个序列的得分为从起始状态转移到其第一个标签的分数。
        '''
        
        # 拿到 每个序列 在第0个时间步 的发射分数（发射到tag[0]中的标签)
            # emissions: (seq_length, batch_size, num_tags)
            # tags 的形状是 (seq_length, batch_size)
            # tags[0] 的形状是 (batch_size,)（取第一个时间步的所有标签）
            
            # 第一个索引 0：选择第一个时间步 → 结果形状变为 (batch_size, num_tags)
            # 第二个索引 torch.arange(batch_size)：选择所有batch样本 → 保持 (batch_size, num_tags)
            # 第三个索引 tags[0]：为每个样本选择对应标签 → 从 num_tags 维度中选择1个
                # 最终为每个样本 i 选择 emissions[0, i, tags[0][i]] 的值
        score += emissions[0, torch.arange(batch_size), tags[0]]   # shape = (batch_size, )
        '''
            为每个序列的初始得分添加其在第一个时间步的发射分数。
            完成了路径得分的第一步计算，即初始转移分数加上第一个时间步的发射分数。
            
            
            emissions[0]
                形状：(batch_size, num_tags)
                含义：第一个时间步的发射分数，针对每个序列和每个可能的标签。
                
            tags.shape = (seq_length, batch_size)  真实的标签
            
            tags[0].shape = (bsz, )
            
            emissions[0, torch.arange(batch_size), tags[0]]

                操作：对于每个序列 i，取出其在时间步 0、标签为 tags[0][i] 的发射分数。
                索引方式：emissions[时间步, 序列索引, 标签索引]
                形状：(batch_size,)
                含义：每个序列在第一个时间步实际标签的发射分数。
        '''

        for i in range(1, seq_length):
            '''
            转移得分：
                从前一个标签转移到当前标签的得分，乘以对应的 mask，确保只在有效位置计算。
            发射得分：
                当前标签的发射得分，乘以 mask。
                
            '''
            
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            # mask[i].shape: (batch_size,)
            # self.transitions.shape: (num_tags, num_tags)
            # self.transitions[tags[i - 1], tags[i]].shape = (batch_size,)
            
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]
            # (batch_size, batch_size) x (batch_size, ) = (batch_size, )
 
            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # score.shape: (batch_size,)
            # mask[i].shape: (batch_size,)
            # emissions[i, torch.arange(batch_size), tags[i]].shape = (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        
        '''
            mask.long().sum(dim=0)
                对 mask 在 dim=0（时间步维度）上求和，计算每个序列的有效长度。
                形状：(batch_size,)。
                含义：得到批次中每个序列的长度（即有效时间步的数量）。
                
            seq_ends = mask.long().sum(dim=0) - 1
                对每个序列的长度减去 1，得到最后一个有效时间步的索引。
                形状：(batch_size,)。
        '''
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1 # 得到每个序列的 最后一个有效时间步的索引。
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)] # 批次中每个序列的最后一个有效标签的索引。
        # 怎么理解：在tags矩阵中，横坐标取了 batch_size 个数， 纵坐标也取了batch_size个数， 一共只取出了batch个值
        '''
        示例：
        对于第 i 个序列：
        
        seq_ends[i]：序列 i 的最后一个有效时间步的索引。
        tags[seq_ends[i], i]：序列 i 在最后一个有效时间步的标签索引。
        
        '''
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]
        
        '''
        结束转移得分：
            从序列最后一个标签转移到结束状态的得分。
        '''

        return score

    def _compute_normalizer(
            self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        '''
            计算 Z(x)
        '''
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all() # 如果mask[0]中的所有值都为真，则断言成立：所有序列在第一个时间步的掩码都为1

        seq_length = emissions.size(0)

        # Start transition score（转移分数） and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        
        '''
        
        #开始转移分数和首次发射分数，大小为

            #（batch_size，num_tags），
            # 其中对于每个批次，第j列存储第一个时间步标记为j的分数
            emissions[0] 存储了每个样本在第一个时间步的 emission 分数 #(batch_size, num_tags)
        '''
        # shape: (batch_size, num_tags)
        # # emissions: (seq_length, batch_size, num_tags)
        # self.start_transitions.shape = (num_tags,)
        
        '''
        # 每个样本的开始转移分数都相同
        
        emission[0]: 每个序列在第一个时间步的发射分数 # (batch_size, num_tags)
        
        self.start_transitions + emissions[0] 导致不同形状的矩阵相加，就只能给start_transitions做广播

        相当于把一个 start_transition向量扩展到 长度为 batch_size的矩阵
        相当于，每个序列在开始节点，都用了相同的转移分数
        
        然后就可以  把每个序列在开始节点（第0个时间步）的转移分数+发射分数
        '''
        score = self.start_transitions + emissions[0]  #(batch_size, num_tags)


        for i in range(1, seq_length):
            # Broadcast score for "every" possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for "every" possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            '''
                self.transitions.shape = (num_tags, num_tags)
                self.broadcast_score.shape = (batch_size, num_tags, 1)   
                self.broadcast_emissions.shape = (batch_size, 1, num_tags)
                
                由于3者形状不匹配，要对他们进行广播
                
                最后他们的形状都变成了：(batch_size, num_tags, num_tags)
                
                self.transitions 广播后 = [
                                        [[0.5, 0.5, 0.5],
                                        [0.5, 0.5, 0.5],
                                        [0.5, 0.5, 0.5]],
                                        
                                        [[0.5, 0.5, 0.5],
                                        [0.5, 0.5, 0.5],
                                        [0.5, 0.5, 0.5]],
                                        ...      
                                    ]
                
                self.broadcast_score 广播前 = [
                    [[0.5], [0.2], [0.3]],
                    [[0.6], [0.1], [0.3]],
                    ...
                ]
                
                
                self.broadcast_emissions 广播前= [
                    [[0.5, 0.4, 0.1]],
                    
                    [[0.2, 0.3, 0.5]],
                    ...
                ]
                
                # broadcast score for every possible next tag
                self.broadcast_score 广播后 = [
                        [[0.5, 0.5, 0.5], 
                        [0.2, 0.2, 0.2],   表示当前序列中 当前tag 和下一个tag之间的转移分数
                        [0.3, 0.3, 0.3]],   这里相当于为next tag初始化转移分数，这里每个current tag
                                            转移到所有next tag的转移分数都初始化为相同的。 
                                            【关键】：列方向值相同
                        [[0.6, 0.6, 0.6], 
                        [0.1, 0.1, 0.1], 
                        [0.3, 0.3, 0.3]],
                    ...
                ]
                
                # Broadcast emission score for "every" possible current tag
                self.broadcast_emissions 广播后= [
                    [[0.5, 0.4, 0.1],
                     [0.5, 0.4, 0.1],
                     [0.5, 0.4, 0.1]],     表示当前序列中 当前tag 和下一个tag之间的转移分数
                                            这里相当于为所有的current tag初始化转移分数，这里每个current tag
                                            转移到所有next tag的转移分数列表都初始化为相同的。 
                                            【关键】：行方向值相同

                    [[0.2, 0.3, 0.5],       
                     [0.2, 0.3, 0.5],
                     [0.2, 0.3, 0.5]],
                    ...
                ]
                
                
                self.transitions 广播后 = [
                                        [[0.5, 0.5, 0.5],
                                        [0.5, 0.5, 0.5],
                                        [0.5, 0.5, 0.5]],
                                        
                                        [[0.5, 0.5, 0.5],
                                        [0.5, 0.5, 0.5],
                                        [0.5, 0.5, 0.5]],
                                        ...      
                                    ]
                

                
            '''
            
            # shape = (batch_size, num_tags, num_tags)
            '''
                总结：
                    broadcast_score: 每个序列各不相同，同一序列中，列方向相同，行方向不同【current_tag转移到所有next_tag的分数都相同】

                    self.transitions: 每个序列的转移分数都相同，列方向、行方向均不相同
                    
                    broadcast_emissions: 每个序列各不相同，同一序列中，列方向不相同，行方向相同【同一个current_tag转移到所有next_tag的分数不相同；
                                                不同current_tag之间的转移分数列表相同】
                    
                    next_score[0] = [[0.5, 0.5, 0.5],
                                    [0.5, 0.5, 0.5],
                                    [0.5, 0.5, 0.5]],

                                        +
                                    [[0.5, 0.5, 0.5],   broadcase_score
                                    [0.2, 0.2, 0.2],        特点：每个current_tag 转到所有 next_tag 的分数都一样
                                    [0.3, 0.3, 0.3]],   他是所有current_tag转移到第一个next_tag的分数的广播
                                    
                                        +
                                    [[0.5, 0.4, 0.1],    emission
                                    [0.5, 0.4, 0.1],         特点：同一个 next_tag 转到所有 current_tag 的分数都一样, 所义，对于上面score中的 score[i][j] (从i转到j的分数)，我们应当在emission矩阵中去寻找从j转到所有next_tag的分数， 正好对应了emission矩阵的第j列。
                                    [0.5, 0.4, 0.1]],    他是第一个current_tag转移到所有next_tag的分数的广播
                                    
                                        =
                                        
                                    [[1.5, 1.4, 1.1],
                                    [1.2, 1.1, 0.8],
                                    [1.3, 1.2, 0.9]]       可以看出 next_score[0][i][j]  = 之前所有可以到达[i][j]的路径的总分 +  从i到j的转移分数 + 从j转移到任意一个tag的分数     
                                    
                                                       下一步 logsumexp，我们直接从j出发， 已经 不需要i了，因此，我们等下需要在i这一维做一个加和归约。 加的结果变成了从j出发的发射分数, shape = (bsz, num_tags)
                                    
                                    
                                   经过 next_score = torch.logsumexp(next_score, dim=1)  
                                   
                                   [4.0, 3.7, 2.8]            
                    
            '''
            next_score = broadcast_score + self.transitions + broadcast_emissions  # (bsz, num_tags, num_tags)

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: 
            
            # for each sample, entry i (next_score[j][i] for sequence j) stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            '''
                对于每个样本，条目 i 存储到目前为止所有可能的以标签 i 结尾的序列的分数之和。
                
                为啥？因为 next_score 存储了每一个时间步的转移分数和发射分数。
            '''
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            # mask[i].shape = (batch_size,)  mask[i] 取了第i个时间步的mask
            '''
                更新每个序列在第i个时间步的累积的转移分数+发射分数
            '''
            score = torch.where(mask[i].unsqueeze(1)==1, next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)

        # \logexp(x1, x2,..., xn) = log(\sum_{i=1}^{n}(exp(xi)))
        # 这个函数常被用于处理数值稳定性的问题，特别是在处理概率的对数空间时，
        # 因为直接在对数空间计算(log(\sum(xi)))和会导致数值溢出(xi太小了)的风险。
        return torch.logsumexp(score, dim=1) # shape = (batch_size,)

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)  发射分数
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0] # 第一个序列位置上的发射分数 + 开始转移分数
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)   
            # indices.shape = (batch_size,)
            # next_score.shape = (batch_size, num_tags)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score) # 更新当前时间步的每个样本的最高分
            history.append(indices) # 把每个时间步选择的标签的索引记录下来

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,) # 每个序列的真实序列长度-1， 相当于
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]  # 获取前一个时间步的 best tag
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list
    
    
    def _beam_search_decode(self, emissions: torch.Tensor, 
                            beam_size = 5,
                            mask: Optional[torch.ByteTensor] = None)-> List[List[int]]:
        """Find the top-k most likely tag sequences using beam search algorithm.

        Args:
            emissions: Emission score tensor of size (seq_length, batch_size, num_tags)
            mask: Mask tensor of size (seq_length, batch_size)
            beam_size: Number of beams to keep at each step

        Returns:
            List of lists containing the top-k tag sequences for each batch

        这个实现的主要特点：
            使用束搜索算法，保持每一步的top-k候选序列
            考虑了mask，对于被mask的位置不进行扩展
            包含了起始转移分数、转移分数、发射分数和结束转移分数
            返回每个batch的top-k序列
        """
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if mask is None:
            mask = torch.ones_like(emissions[:, :, 0], dtype=torch.uint8)

        seq_length, batch_size = mask.shape
        num_tags = emissions.size(2)

        # Initialize beams: (batch_size, beam_size)
        # Each beam contains (score, [tag_sequence])
        beams = [[(self.start_transitions + emissions[0, i], [tag]) 
                 for tag in range(num_tags)] 
                 for i in range(batch_size)]
        
        # Sort and select top-k beams for each batch
        beams = [sorted(batch_beams, key=lambda x: x[0][tag], reverse=True)[:beam_size]
                for batch_beams, tag in zip(beams, range(batch_size))]

        for t in range(1, seq_length):
            new_beams = [[] for _ in range(batch_size)]
            
            for i in range(batch_size):
                if not mask[t, i]:
                    # If mask is 0, keep the same beams
                    new_beams[i] = beams[i]
                    continue
                    
                for score, tags in beams[i]:
                    # Expand each beam with all possible next tags
                    for tag in range(num_tags):
                        new_score = score + self.transitions[tags[-1], tag] + emissions[t, i, tag]
                        new_tags = tags + [tag]
                        new_beams[i].append((new_score, new_tags))
            
            # Select top-k beams for each batch
            beams = [sorted(batch_beams, key=lambda x: x[0], reverse=True)[:beam_size]
                    for batch_beams in new_beams]

        # Add end transition scores
        final_beams = []
        for i in range(batch_size):
            batch_final = []
            for score, tags in beams[i]:
                final_score = score + self.end_transitions[tags[-1]]
                batch_final.append((final_score, tags))
            final_beams.append(batch_final)

        # Get top-k sequences for each batch
        top_sequences = []
        for i in range(batch_size):
            batch_sequences = sorted(final_beams[i], key=lambda x: x[0], reverse=True)
            top_sequences.append([seq for _, seq in batch_sequences])

        return top_sequences





if __name__ == '__main__':
    # model = CRF(num_tags=5, batch_first=True)
    
      # 设置随机种子以获得可重复的结果  
    torch.manual_seed(1)  

    # 定义参数  
    BATCH_SIZE = 2  
    SEQ_LENGTH = 5  
    NUM_TAGS = 3  
    FEATURE_DIM = 4  # 发射分数的特征维度  

    # 创建随机的发射分数（通常来自于上一层，如 BiLSTM）  
    emissions = torch.randn(BATCH_SIZE, SEQ_LENGTH, NUM_TAGS)  
    # 随机生成标签序列  
    tags = torch.randint(NUM_TAGS, (BATCH_SIZE, SEQ_LENGTH), dtype=torch.long)  
    # 创建掩码，假设所有序列都是完整的（没有填充）  
    mask = torch.ones(BATCH_SIZE, SEQ_LENGTH, dtype=torch.uint8)  

    # 初始化 CRF 模型  
    model = CRF(num_tags=NUM_TAGS, batch_first=True)  

    # 计算对数似然损失，并取负数（因为我们通常最小化损失）  
    loss = -model.forward(emissions, tags, mask=mask)  
    print(f'Negative log-likelihood loss: {loss.item()}')  

    # 使用维特比算法解码最可能的标签序列  
    best_tag_sequences = model.decode(emissions, mask=mask)  
    print('Best tag sequences:')  
    for seq in best_tag_sequences:  
        print(seq)  
    






