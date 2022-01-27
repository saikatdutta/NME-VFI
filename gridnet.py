import torch
import torch.nn as nn


class lateral_block(nn.Module):

	def __init__(self, inChannels, outChannels, res_conn = True):

		super(lateral_block, self).__init__()

		self.block = nn.Sequential(
				nn.PReLU(),
				nn.Conv2d(inChannels,outChannels,3,1,1),
				nn.PReLU(),
				nn.Conv2d(outChannels,outChannels,3,1,1)


			)
		self.res_conn = res_conn

	def forward(self, x):
		x1 = self.block(x)
		if (self.res_conn):
			x1 = x+x1
		return x1

class down_block(nn.Module):

	def __init__(self, inChannels, outChannels):

		super(down_block, self).__init__()

		self.block = nn.Sequential(
				nn.PReLU(),
				nn.Conv2d(inChannels,outChannels,3,2,1),
				nn.PReLU(),
				nn.Conv2d(outChannels,outChannels,3,1,1)


			)

	def forward(self, x):
		x1 = self.block(x)
		return x1


class up_block(nn.Module):

	def __init__(self, inChannels, outChannels):

		super(up_block, self).__init__()

		self.block = nn.Sequential(
				nn.Upsample(scale_factor=2, mode= 'bilinear'),
				nn.PReLU(),
				nn.Conv2d(inChannels,outChannels,3,1,1),
				nn.PReLU(),
				nn.Conv2d(outChannels,outChannels,3,1,1)


			)

	def forward(self, x):
		x1 = self.block(x)
		return x1



class GridNet(nn.Module):

	def __init__(self, inChannels, outChannels, channel_list=[32,64,96]):

		super(GridNet, self).__init__()

		c1,c2,c3 = channel_list

		self.Lin = lateral_block(inChannels,c1 ,False)

		#row0
		self.L00 = lateral_block(c1 ,c1 )
		self.L01 = lateral_block(c1 ,c1 )
		self.L02 = lateral_block(c1 ,c1 )
		self.L03 = lateral_block(c1 ,c1 )
		self.L04 = lateral_block(c1 ,c1 )

		#row1
		self.L10 = lateral_block(c2,c2)
		self.L11 = lateral_block(c2,c2)
		self.L12 = lateral_block(c2,c2)
		self.L13 = lateral_block(c2,c2)
		self.L14 = lateral_block(c2,c2)

		#row2
		self.L20 = lateral_block(c3,c3)
		self.L21 = lateral_block(c3,c3)
		self.L22 = lateral_block(c3,c3)
		self.L23 = lateral_block(c3,c3)
		self.L24 = lateral_block(c3,c3)

		self.Lout = lateral_block(c1 ,outChannels,False)

		self.d00 = down_block(c1 ,c2)
		self.d01 = down_block(c1 ,c2)
		self.d02 = down_block(c1 ,c2)

		self.d10 = down_block(c2,c3)
		self.d11 = down_block(c2,c3)
		self.d12 = down_block(c2,c3)

		self.u00 = up_block(c2,c1 )
		self.u01 = up_block(c2,c1 )
		self.u02 = up_block(c2,c1 )

		self.u10 = up_block(c3,c2)
		self.u11 = up_block(c3,c2)
		self.u12 = up_block(c3,c2)

	def forward(self,x):

		out_Lin = self.Lin(x)
		out_L00 = self.L00(out_Lin)
		out_L01 = self.L01(out_L00)
		out_L02 = self.L02(out_L01)
		

		out_d00 = self.d00(out_Lin)
		out_d01 = self.d01(out_L00)
		out_d02 = self.d02(out_L01)

		out_L10 = self.L10(out_d00)
		out_L11 = self.L11(out_d01 + out_L10)
		out_L12 = self.L12(out_d02 + out_L11) 

		out_d10 = self.d10(out_d00)
		out_d11 = self.d11(out_L10 + out_d01)
		out_d12 = self.d12(out_L11 + out_d02)

		out_L20 = self.L20(out_d10)
		out_L21 = self.L21(out_d11 + out_L20)
		out_L22 = self.L22(out_d12 + out_L21)

		out_u10 = self.u10(out_L22)
		out_L23 = self.L23(out_L22)
		out_u11 = self.u11(out_L23)
		out_L24 = self.L24(out_L23)
		out_u12 = self.u12(out_L24)

		out_L13 = self.L13(out_u10 + out_L12)
		out_L14 = self.L14(out_u11 + out_L13)
		out_u00 = self.u00(out_u10 + out_L12)
		out_u01 = self.u01(out_u11 + out_L13)
		out_u02 = self.u02(out_u12 + out_L14)

		out_L03 = self.L03(out_u00 + out_L02)
		out_L04 = self.L04(out_u01 + out_L03)

		out_final = self.Lout(out_L04 + out_u02)

		return out_final, out_L04+out_u02


