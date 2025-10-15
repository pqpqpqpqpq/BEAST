from model.model_ST import ST
from model.model_TS import TS
import torch.nn as nn
from model.net import Unit2D
from model.unit_agcn import unit_agcn


class ST_GCN_AltFormer(nn.Module):

    def __init__(self,
                 channel = 8,
                 #num_class,
                 backbone_in_c = 128,
                 num_frame = 6,
                 num_joints=23,
                 style = None,
                 graph=None,
                 graph_args=dict(),
                 mask_learning=False,
                 use_local_bn=False,
                 ):
        super(ST_GCN_AltFormer, self).__init__()
        '''
        if graph is None:
            raise ValueError()
        else:
            # print(graph)
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)
            self.A = torch.from_numpy(self.graph.A.astype(np.float32))
        '''

        self.channel = channel
        self.num_frame = num_frame
        self.num_joints = num_joints
        #self.num_class = num_class
        self.backbone_in_c = backbone_in_c
        self.style = style
        self.mask_learning = mask_learning
        self.use_local_bn = use_local_bn

        self.gcn0 = unit_agcn(
            channel,
            self.backbone_in_c,
            mask_learning=mask_learning,
            use_local_bn=use_local_bn)

        self.tcn0 = Unit2D(self.backbone_in_c, self.backbone_in_c, kernel_size=9)


        self.modelA = ST(  1, num_frame=self.num_frame, num_joints= self.num_joints, in_chans=128, embed_dim_ratio=256, depth=1,
                             num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1)

        self.modelB = TS( 1, num_frame=self.num_frame, num_joints= self.num_joints, in_chans=128, embed_dim_ratio=256,
                             depth=2,
                             num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1)

        self.conv = nn.Conv2d(in_channels=128, out_channels=512, kernel_size=1)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 1)
        )



    def forward(self, x,A_batched):

        '''
        gcn0 = unit_agcn(
            self.channel,
            self.backbone_in_c,
            A_batched,
            mask_learning=self.mask_learning,
            use_local_bn=self.use_local_bn).cuda(x.get_device())
        '''


        x = x.permute(0,3,1,2)

        N, C, T, V = x.size()  ##Batch  channel  frame  joint

        x = x.permute(0, 1, 2, 3).contiguous().view(N, C, T, V)

        x = self.gcn0(x, A_batched)

        x = self.tcn0(x)

        if self.style == 'ST':
            x_st, spatial_attns, temporal_attns = self.modelA(x)
            pred = x_st

            return pred,spatial_attns, temporal_attns


        elif self.style == 'TS':
            x_ts = self.modelB(x)
            pred = x_ts

        elif self.style == 'FC':
            x = self.conv(x)
            x = x.mean(dim=[2, 3])

            pred = self.mlp_head(x)

        else:
            x_st = self.modelA(x)
            x_ts = self.modelB(x)
            pred = x_ts + x_st

        return  pred
