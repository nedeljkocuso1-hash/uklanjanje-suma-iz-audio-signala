import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # ENCODER (Smanjivanje)
        self.enc1 = self.conv_block(1, 16)
        self.enc2 = self.conv_block(16, 32)
        self.enc3 = self.conv_block(32, 64)
        
        self.pool = nn.MaxPool2d(2)
        
        # BOTTLENECK (Dno slova U)
        self.bottleneck = self.conv_block(64, 128)
        
        # DECODER (Povećavanje)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(128, 64) # 128 jer spajamo sa enc3 (skip connection)
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(64, 32)
        
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(32, 16)
        
        # FINALNI SLOJ
        self.final = nn.Conv2d(16, 1, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # --- ENCODER ---
        e1 = self.enc1(x)
        p1 = self.pool(e1) # 257 -> 128
        
        e2 = self.enc2(p1)
        p2 = self.pool(e2) # 128 -> 64
        
        e3 = self.enc3(p2)
        p3 = self.pool(e3) # 64 -> 32
        
        # --- BOTTLENECK ---
        b = self.bottleneck(p3)
        
        # --- DECODER ---
        # Nivo 3
        d3 = self.up3(b)
        # Fix dimenzija: nateraj d3 da bude iste veličine kao e3
        d3 = torch.nn.functional.interpolate(d3, size=(e3.size(2), e3.size(3)))
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        # Nivo 2
        d2 = self.up2(d3)
        # Fix dimenzija: nateraj d2 da bude iste veličine kao e2
        d2 = torch.nn.functional.interpolate(d2, size=(e2.size(2), e2.size(3)))
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        # Nivo 1
        d1 = self.up1(d2)
        # Fix dimenzija: nateraj d1 da bude iste veličine kao e1
        d1 = torch.nn.functional.interpolate(d1, size=(e1.size(2), e1.size(3)))
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final(d1)

if __name__ == "__main__":
    m = UNet()
    test_input = torch.randn(1, 1, 257, 188)
    output = m(test_input)
    print(f"Ulazni oblik: {test_input.shape}")
    print(f"Izlazni oblik: {output.shape}")
    print("Model je uspešno definisan!")