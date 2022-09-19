# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 19:31:50 2021

@author: Eashan S, Abhishek C

"""
import torch.nn as nn

class Attention_block(nn.Module):
    
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class conv_block(nn.Module):
    def __init__(self, in_map, out_map, kernel = 3, stride = 1, activation = True):
        super(conv_block, self).__init__()
        
        self._mconv = nn.Sequential(
            nn.Conv2d(in_map, out_map, kernel, stride, (kernel)//2),
            nn.BatchNorm2d(out_map)            
            )
        
        if(activation):
            self._mconv.add_module("conv_block_relu", nn.ReLU(inplace=True))
        
    def forward(self, x):        
        out = self._mconv(x)
        
        return out
    
class deconv_block(nn.Module):
    def __init__(self, in_map, out_map, kernel = 3, stride = 2, padding = 1):
        super(deconv_block, self).__init__()
                
        self._conv_trans_2d =  nn.ConvTranspose2d(in_map, out_map, kernel, stride, padding)
        self._batch_norm_2d = nn.BatchNorm2d(out_map)
        self._relu = nn.ReLU(inplace=True)        
        
    def forward(self, x, output_size):
        out = self._conv_trans_2d(x, output_size = output_size)
        
        return out
    
class res_block(nn.Module):
    def __init__(self, in_map, out_map, downsample = False):
        super(res_block, self).__init__()
                        
        self._mconv_2 = conv_block(out_map, out_map, 3, 1, False)        
        
        if downsample == True:
            stride = 2            
        else:
            stride = 1            
            
        self._mconv_1 = conv_block(in_map, out_map, 3, stride)
        self._mdownsample = nn.Sequential(
                nn.Conv2d(in_map, out_map, 1, stride),
                nn.BatchNorm2d(out_map)
                )
        self._relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        
        residual = x
        out = self._mconv_1(x)
        out = self._mconv_2(out)
        residual = self._mdownsample(x)
        #print("residual size,", residual.size())
        out = residual + out
        out = self._relu(out)
        
        return out
    
class encoder(nn.Module):
    def __init__(self, in_map, out_map):
        super(encoder, self).__init__()
        
        self._mres_1 = res_block(in_map, out_map, True)
        self._mres_2 = res_block(out_map, out_map)
        
        
    def forward(self, x):
        out = self._mres_1(x)
        out = self._mres_2(out)
        
        return out
    
class decoder(nn.Module):
    def __init__(self, in_map, out_map, padding = 1):
        super(decoder, self).__init__()
        
        self._mconv_1 = conv_block(in_map, in_map//4, 1)
        self._mdeconv_1 = deconv_block(in_map//4, in_map//4, 3, 2, padding)
        self._mconv_2 = conv_block(in_map//4, out_map, 1)        
        
    def forward(self, x, output_size):
        out = self._mconv_1(x)
        out = self._mdeconv_1(out, output_size = output_size)
        out = self._mconv_2(out)
        
        return out        
    
class link_net(nn.Module):
    def __init__(self, in_ch=3,category = 1):
        super(link_net, self).__init__()
        
        self._mconv_1 = conv_block(in_ch, 64, 7, 2)
        self._mmax_pool = nn.MaxPool2d(3, 2, padding=1)
        
        self._mencoder_1 = encoder(64, 64)
        self._mencoder_2 = encoder(64, 128)
        self._mencoder_3 = encoder(128, 256)
        self._mencoder_4 = encoder(256, 512)
        self._mencoder_5 = encoder(512,1024)
        
        self._mdecoder_1 = decoder(64, 64)
        self._mdecoder_2 = decoder(128, 64)
        self._mdecoder_3 = decoder(256, 128)
        self._mdecoder_4 = decoder(512, 256)
        self._mdeccoder_5 = decoder(1024,512)
        
        
        self._deconv_1 = deconv_block(64, 32)
        self._mconv_2 = conv_block(32, 32, 3)
        self._deconv_2 = deconv_block(32, category, 2, 2, 0)
        
        self._attblock_2 = Attention_block(64, 64, 32)
        self._attblock_3 = Attention_block(128,128,64)
        self._attblock_4 = Attention_block(256,256,128)
        self._attblock_5 = Attention_block(512,512,256)

    def forward(self, x):
                                        
        conv_down_out = self._mconv_1(x)
        max_pool_out = self._mmax_pool(conv_down_out)
                
        encoder_1_out = self._mencoder_1(max_pool_out)
        encoder_2_out = self._mencoder_2(encoder_1_out)                
        encoder_3_out = self._mencoder_3(encoder_2_out)        
        encoder_4_out = self._mencoder_4(encoder_3_out)
        encoder_5_out = self._mencoder_5(encoder_4_out)
        
        decoder_5_out = self._mdeccoder_5(encoder_5_out,encoder_4_out.size())+ encoder_4_out
        att_5_out     = self._attblock_5(g=decoder_5_out,x=encoder_4_out)
        decoder_5_out = decoder_5_out+att_5_out

        decoder_4_out = self._mdecoder_4(decoder_5_out, encoder_3_out.size()) + encoder_3_out
        att_4_out     = self._attblock_4(g=decoder_4_out,x=encoder_3_out)
        decoder_4_out = decoder_4_out+att_4_out
        
        decoder_3_out = self._mdecoder_3(decoder_4_out, encoder_2_out.size()) + encoder_2_out
        att_3_out     = self._attblock_3(g=decoder_3_out,x=encoder_2_out)
        decoder_3_out = att_3_out + decoder_3_out

        
        decoder_2_out = self._mdecoder_2(decoder_3_out, encoder_1_out.size()) + encoder_1_out
        att_2_out     = self._attblock_2(g=decoder_2_out,x=encoder_1_out)
        decoder_2_out = att_2_out + decoder_2_out
       
        
        decoder_1_out = self._mdecoder_1(decoder_2_out, max_pool_out.size())
        
        
        deconv_out = self._deconv_1(decoder_1_out, conv_down_out.size())
        conv_2_out = self._mconv_2(deconv_out)
        out = self._deconv_2(conv_2_out, x.size())
        
                
        return out
