import os
import torch
import torch.nn as nn
from einops import rearrange, repeat
from net.decoder import Decoder
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.upsample import SubpixelUpsample

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_CACHE_DIR = os.path.join(PROJECT_ROOT, '.hf_cache')
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ.setdefault('HF_HOME', HF_CACHE_DIR)
os.environ.setdefault('HUGGINGFACE_HUB_CACHE', os.path.join(HF_CACHE_DIR, 'hub'))
os.environ.setdefault('TRANSFORMERS_CACHE', os.path.join(HF_CACHE_DIR, 'hub'))
os.environ.setdefault('HF_MODULES_CACHE', os.path.join(HF_CACHE_DIR, 'modules'))

import transformers.dynamic_module_utils as dynamic_module_utils
import transformers.utils.hub as transformers_hub

dynamic_module_utils.HF_MODULES_CACHE = os.path.join(HF_CACHE_DIR, 'modules')
transformers_hub.TRANSFORMERS_CACHE = os.path.join(HF_CACHE_DIR, 'hub')
os.makedirs(dynamic_module_utils.HF_MODULES_CACHE, exist_ok=True)
os.makedirs(transformers_hub.TRANSFORMERS_CACHE, exist_ok=True)

from transformers import AutoTokenizer, AutoModel



class BERTModel(nn.Module):
    def __init__(self, bert_type, project_dim):
        super(BERTModel, self).__init__()
        self.model = AutoModel.from_pretrained(bert_type,output_hidden_states=True,trust_remote_code=True)
        self.project_head = nn.Sequential(             
            nn.Linear(768, project_dim),
            nn.LayerNorm(project_dim),             
            nn.GELU(),             
            nn.Linear(project_dim, project_dim)
        )
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask,output_hidden_states=True,return_dict=True)
        last_hidden_states = torch.stack([output['hidden_states'][1], output['hidden_states'][2], output['hidden_states'][-1]]) # n_layer, batch, seqlen, emb_dim
        embed = last_hidden_states.permute(1,0,2,3).mean(2).mean(1) 
        embed = self.project_head(embed)
        return {'feature':output['hidden_states'],'project':embed}

class VisionModel(nn.Module):
    def __init__(self, vision_type, project_dim):
        super(VisionModel, self).__init__()
        self.model = AutoModel.from_pretrained(vision_type,output_hidden_states=True)   
        self.project_head = nn.Linear(768, project_dim)
        self.spatial_dim = 768

    def forward(self, x):
        output = self.model(x, output_hidden_states=True)
        embeds = output['pooler_output'].squeeze()
        project = self.project_head(embeds)
        return {"feature":output['hidden_states'], "project":project}

class FFBI(nn.Module):
    def __init__(self, dim, num,batchf):
        super(FFBI, self).__init__()
        self.cross_attnh = nn.MultiheadAttention(embed_dim=dim,num_heads=num,batch_first=batchf)
        self.cross_attnl = nn.MultiheadAttention(embed_dim=dim,num_heads=num,batch_first=batchf)
        self.norm_h = nn.LayerNorm(dim)
        self.norm_l = nn.LayerNorm(dim)

    def forward(self, x,y):
        x1, _=self.cross_attnl(query=x,key=y,value=y)
        x2 = self.norm_h(x1 + x)
        y1, _ = self.cross_attnh(query=y,key=x,value=x)
        y2 = self.norm_l(y1 + y)
        return x2,y2


class SegModel(nn.Module):
    def __init__(self, bert_type, vision_type, project_dim=512):
        super(SegModel, self).__init__()
        self.encoder = VisionModel(vision_type, project_dim)
        self.encoder2 = VisionModel(vision_type, project_dim)
        self.text_encoder = BERTModel(bert_type, project_dim)
        self.spatial_dim = [7,14,28,56]   
        feature_dim = [768,384,192,96]

        self.decoder16 = Decoder(feature_dim[0],feature_dim[1],self.spatial_dim[0],24)
        self.decoder8 = Decoder(feature_dim[1],feature_dim[2],self.spatial_dim[1],12)
        self.decoder4 = Decoder(feature_dim[2],feature_dim[3],self.spatial_dim[2],9)
        self.decoder1 = SubpixelUpsample(2,feature_dim[3],24,4)
        self.out = UnetOutBlock(2, in_channels=24, out_channels=1)

        self.decoder16_2 = Decoder(feature_dim[0],feature_dim[1],self.spatial_dim[0],24)
        self.decoder8_2 = Decoder(feature_dim[1],feature_dim[2],self.spatial_dim[1],12)
        self.decoder4_2 = Decoder(feature_dim[2],feature_dim[3],self.spatial_dim[2],9)
        self.decoder1_2 = SubpixelUpsample(2,feature_dim[3],24,4)
        self.out_2 = UnetOutBlock(2, in_channels=24, out_channels=1)
        self.ffbi = FFBI(feature_dim[0],4,True)
    def forward(self, data):

        high_image, low_image, text = data
        if high_image.shape[1] == 1:
            high_image = repeat(high_image,'b 1 h w -> b c h w',c=3)
            low_image = repeat(low_image,'b 1 h w -> b c h w',c=3)

        high_output = self.encoder(high_image)
        low_output = self.encoder2(low_image)
        high_features, _ = high_output['feature'], high_output['project']
        low_features, _ = low_output['feature'], low_output['project']
        text_output = self.text_encoder(text['input_ids'],text['attention_mask'])
        text_embeds, _ = text_output['feature'],text_output['project']

        if len(high_features[0].shape) == 4:
            high_features = high_features[1:]
            high_features = [rearrange(item,'b c h w -> b (h w) c') for item in high_features]
            low_features = low_features[1:]
            low_features = [rearrange(item,'b c h w -> b (h w) c') for item in low_features]
        high_os32 = high_features[3]
        low_os32 = low_features[3]

        high_fu32, low_fu32 = self.ffbi(high_os32, low_os32)
        text_tokens = text_embeds[-1]

        high_os16 = self.decoder16(high_fu32, high_features[2], text_tokens)
        low_os16 = self.decoder16_2(low_fu32, low_features[2], text_tokens)
        
        high_os8 = self.decoder8(high_os16, high_features[1], text_tokens)
        low_os8 = self.decoder8_2(low_os16, low_features[1], text_tokens)

        high_os4 = self.decoder4(high_os8, high_features[0], text_tokens)
        low_os4 = self.decoder4_2(low_os8, low_features[0], text_tokens)
        high_os4 = rearrange(high_os4, 'B (H W) C -> B C H W',H=self.spatial_dim[-1],W=self.spatial_dim[-1])
        low_os4 = rearrange(low_os4, 'B (H W) C -> B C H W',H=self.spatial_dim[-1],W=self.spatial_dim[-1])
        high_os1 = self.decoder1(high_os4)
        low_os1 = self.decoder1_2(low_os4)

        high_logits = self.out(high_os1)
        low_logits = self.out_2(low_os1)
        return high_logits, low_logits
    
