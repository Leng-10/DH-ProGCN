import torch
import torch.nn as nn
class channel_cluster_layer(nn.Module):

    '''
    adopt fc layers for clusterring:
    input: 2048 * 14 * 14
    output: [cluster_result, attention_mask]

    cluster_result: part_nums * channel_num
    weighted_feature: part_nums * 14 * 14
    '''

    def __init__(self, part_num, channel_num):
        super(channel_cluster_layer, self).__init__()
        self.part_num = part_num
        self.channel_num = channel_num

        hidden_states = int(self.channel_num * self.part_num /2)

        self.fc1 = nn.Linear(self.channel_num, hidden_states)
        self.fc2 = nn.Linear(hidden_states, self.channel_num * self.part_num)

        #         self.fc1 = nn.Linear(self.channel_num,self.part_num)
        #         self.fc2 = nn.Linear(self.part_num,self.channel_num)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.avgpool2 = nn.AdaptiveAvgPool3d((891, 1, 1))


    def forward(self, x):

        # the input feature map is the output of conv22 in Net_v2 with the dimension of 2048*14*14
        # dimension of 2048*14*14 need to replaced by 512*9*11*9
        conv_matrix = torch.clone(x)
        conv_matrix = conv_matrix.reshape(conv_matrix.size(0), self.channel_num, 1, 891)

        '''
        get the weights for each channel
        input: feature maps of 2048 * 14 * 14, part_num n --> 512*9*11*9
        output: channel weights of part_num * 2048 --> 512
        '''
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        grouping_result = x.unfold(-1, self.channel_num, self.channel_num)
        #  calculate the loss here to supervise pre-training

        '''
        get the weighted featuremap of the regions
        '''
        conv_matrix = conv_matrix.unsqueeze(1)
        x = grouping_result.unsqueeze(-1).unsqueeze(-1)
        x = x * conv_matrix
        x = x.transpose(2, 4)
        x = self.avgpool2(x)  # avgpool over the channels

        x = x.reshape(x.size(0), self.part_num, 9, 11, 9)


        return grouping_result, x



