import torch
import torch.nn as nn
from einops import rearrange, repeat
from monai.networks.blocks.unetr_block import UnetrUpBlock

class LFFI(nn.Module):
    def __init__(self, in_channels:int, output_text_len:int, input_text_len:int=24, embed_dim:int=768):
        super(LFFI, self).__init__()
        self.visual_norm = nn.LayerNorm(in_channels)
        self.text_norm = nn.LayerNorm(in_channels)
        self.cross_attn1 = nn.MultiheadAttention(embed_dim=in_channels,num_heads=4,batch_first=True)
        self.cross_attn2 = nn.MultiheadAttention(embed_dim=in_channels,num_heads=4,batch_first=True)
        self.text_project = nn.Sequential(
            nn.Conv1d(input_text_len,output_text_len,kernel_size=1,stride=1),
            nn.GELU(),
            nn.Linear(embed_dim,in_channels),
            nn.LayerNorm(in_channels),
        )
        self.filter_linear = nn.Linear(output_text_len, in_channels)
        self.output_norm = nn.LayerNorm(in_channels)

    def forward(self,x,txt):
        '''
        x:[B N C1]
        tExt:[B,L,C]
        '''
        txt = self.text_project(txt)
        vis = self.visual_norm(x)
        txt = self.text_norm(txt)
        vis_attn,_ = self.cross_attn1(query=vis,key=txt,value=txt)
        txt_attn,_ = self.cross_attn2(query=txt,key=vis,value=vis)
        filter_map = torch.matmul(vis_attn, txt_attn.transpose(1, 2))
        filter_weights = torch.sigmoid(self.filter_linear(filter_map))
        output = x + vis_attn * filter_weights
        return self.output_norm(output)

class Decoder(nn.Module):
    def __init__(self,in_channels, out_channels, spatial_size, text_len) -> None:
        super().__init__()
        self.lffi_layer = LFFI(in_channels,text_len)  
        self.spatial_size = spatial_size
        self.decoder = UnetrUpBlock(2,in_channels,out_channels,3,2,norm_name='BATCH')

    def forward(self, vis, skip_vis, txt):
        if txt is not None:
            vis =  self.lffi_layer(vis, txt)
        vis = rearrange(vis,'B (H W) C -> B C H W',H=self.spatial_size,W=self.spatial_size)
        skip_vis = rearrange(skip_vis,'B (H W) C -> B C H W',H=self.spatial_size*2,W=self.spatial_size*2)
        output = self.decoder(vis,skip_vis)
        output = rearrange(output,'B C H W -> B (H W) C')
        return output


