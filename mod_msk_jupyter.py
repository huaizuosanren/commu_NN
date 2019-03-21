
# coding: utf-8

# In[82]:


import matplotlib.pyplot as plt
import numpy as np
 
from math import pi
from scipy import interpolate 
import matplotlib.pyplot as plt
 
import matplotlib
 
import scipy.signal as signal
 
import math
# generate symbol
data_len=10
sample_num=8
Rb=50
fc=20000
sampling_t=0.01
t=np.arange(0,data_len,sampling_t)
def symbol_out(data_len):
	a = np.random.randint(0, 2, data_len)
	m = np.zeros(len(t), dtype=np.float32)
	for i in range(len(t)):
		m[i] = a[math.floor(t[i])]
	for i in range(len(a)):
		if a[i]>0:
			a[i]=a[i]
		else:
			a[i]=-1
	return a,m
data,data_p=symbol_out(data_len)
data=[-1, 1, 1, 1, 1, 1, -1, -1, -1, 1]
# 解决set_title中文乱码
 
#zhfont1 = matplotlib.font_manager.FontProperties(fname = 'C:\Windows\Fonts\simsun.ttc')
#------plot figure----------
fig = plt.figure() 
ax1 = fig.add_subplot(3, 1, 1)
ax1.set_title('generate binary symbol', fontproperties = zhfont1, fontsize = 20)
 
plt.axis([0, 10, -0.5, 1.5])
 
plt.plot(t, data_p, 'b')
print (data)


# In[88]:


#---------- I and Q---------
def IQ_msk(data,data_len,sample_num,Rb):
    Tb=1/Rb
    #fs=Rb*sample_num
    # sampling
    data_sample=np.zeros(data_len*sample_num)
    count=0
    for i in range(data_len):
    	data_sample[count:(i+1)*sample_num+1]=data[i]
    	count=(i+1)*sample_num+1
    # phase
    phase=np.zeros(data_len*sample_num)
    phase[0]=data_sample[0]*pi/2/sample_num
    for i in range(1,data_len*sample_num):
    	phase[i]=phase[i-1]+data_sample[i-1]*pi/2/sample_num
    I_out=np.cos(phase)
    Q_out=np.sin(phase)
    print ("phase=",phase,end="")
    print("data_sample=",data_sample)
    return I_out,Q_out
I_out,Q_out=IQ_msk(data,data_len,sample_num,Rb)
#-------plot figure--------------
fig2 = plt.figure() 
#ax2=fig.add_subplot(3,1,2)
#fig2.set_title('I out', fontproperties = zhfont1, fontsize = 20)
plt.axis([0, 100, -1.5, 1.5])
plt.plot(I_out, 'b')
#ax3=fig.add_subplot(3,1,3)
#ax2.set_title('Q out', fontproperties = zhfont1, fontsize = 20)
plt.axis([0, 100, -1.5, 1.5])
plt.plot(Q_out, 'r')


# In[93]:


#---modulate to defined frequency-----
multi = fc/Rb
x_I=np.linspace(0,len(I_out)-1,len(I_out))
x_Itemp=np.linspace(0,len(I_out)-1,(len(I_out)*multi))
f_I = interpolate.interp1d(x_I,I_out,kind="slinear")
I_temp=f_I(x_Itemp)
x_Q=np.linspace(0,len(Q_out)-1,len(Q_out))
x_Qtemp=np.linspace(0,len(Q_out)-1,(len(Q_out)*multi))
f_Q = interpolate.interp1d(x_Q,Q_out,kind="slinear")
Q_temp=f_Q(x_Qtemp)
fig3 = plt.figure() 
plt.axis([0, 30000, -1.5, 1.5])
plt.plot(I_temp, 'b')
plt.axis([0, 30000, -1.5, 1.5])
plt.plot(Q_temp, 'r')

fs=fc*sample_num
ts = np.arange(0, (len(I_temp)* 1) / fs, 1 / fs)
signal_I=I_temp*np.cos(np.dot(2*pi*fc,ts))
signal_Q=Q_temp*np.sin(np.dot(2*pi*fc,ts))
fig4 = plt.figure() 
plt.axis([0, 30000, -1.5, 1.5])
plt.plot(signal_I, 'b')
plt.axis([0, 30000, -1.5, 1.5])
plt.plot(signal_Q, 'r')
signal_mod=signal_I-signal_Q
#---------plot figure--------
fig5 = plt.figure() 
plt.axis([0, 30000, -1.5, 1.5])
plt.plot(signal_mod, 'r')

