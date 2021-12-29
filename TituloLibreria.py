import cv2
import numpy as np
import tensorflow as tf
from operator import xor
from matplotlib import pyplot
import matplotlib.pyplot as plt


class HexImage:
    def __init__(self,image): 
        self.pic = image
        
    def sqr2hex(img,hexTam=1,tam=1):
        Copia =np.array(img)

        if(len(np.shape(Copia))==2):
            n=np.shape(img)[0]
            m=np.shape(img)[1]
            Amplificar = tf.keras.layers.UpSampling2D(size=(tam,tam))(Copia.reshape([1,n,m,1])).numpy()
            Restablecer = np.array(Amplificar.reshape([n*tam,m*tam]))
        else:
            n=np.shape(img)[0]
            m=np.shape(img)[1]
            l=np.shape(img)[2]
            Amplificar = tf.keras.layers.UpSampling2D(size=(tam,tam))(Copia.reshape([1,n,m,l])).numpy()
            Restablecer = np.array(Amplificar.reshape([n*tam,m*tam,l]))

        for i in range(n*tam):
            if i%(2*hexTam) < 1*hexTam :
                Restablecer[i,:] = 0

        for i in range(m*tam):
            if i%(2**hexTam) < 1*hexTam :
                Restablecer[:,i] = 0 

        for i in range(n*tam):
            if i%(4*hexTam) < 2*hexTam :
                Restablecer[i,int(1*hexTam):] = np.array(Restablecer[i,:-int(hexTam*1)])

        return Restablecer
    
    def HexConv(img,Filter):
        return cv2.filter2D(img, -1, cv2.flip(Filter,-1), borderType=cv2.BORDER_CONSTANT)

    def read(name):
        return cv2.imread(name)

    def write(img,name):
        cv2.imwrite(name, img)
    
    def hecx(sample):
        hexa=np.zeros((9,8))+1
        hexa[0][0]=0
        hexa[0][1]=0
        hexa[0][2]=0
        hexa[1][0]=0
        hexa[:2,-3:]=hexa[:2,2::-1]
        hexa[-2:]=hexa[:2][::-1]
        hexa2=(hexa-1)*(-1)+0
        res1,res2=sample*hexa,sample*hexa2

        mean=int(sum(sum(res1))/56)
        final=hexa*mean
        final=res2+final
        return final

    def recx(sample):
        hexa=np.ones((9,8))
        res1=sample*hexa
        mean=int(sum(sum(res1))/72)
        final=hexa*mean
        return final

    def HexSample(img,tam=4):
        Copia =np.array(img)
        n=np.shape(img)[0]
        m=np.shape(img)[1]
        if(len(np.shape(img))<3):
            l=1
        else:
            l=np.shape(img)[2]

        Amplificar = tf.keras.layers.UpSampling2D(size=(tam,tam))(Copia.reshape([1,n,m,l])).numpy()
        Restablecer = np.array(Amplificar.reshape([n*tam,m*tam,l]))
        Sep=np.array([Restablecer[:,:,i] for i in range(l)])

        for k in range(l):
            for i in range(0,4*m-9,9):
                for j in range(0,4*n-11,8):
                    if(i%2==0):
                        Sep[k][i:i+9,j:(j+8)]=HexImage.hecx(Sep[k][i:i+9,j:(j+8)])
                    else:
                        Sep[k][i:i+9,j+3:(j+11)]=HexImage.hecx(Sep[k][i:i+9,j+3:(j+11)])

        if(l==1):
            return Sep[0]
        else:
            A=Sep[0]
            B=Sep[1]
            C=Sep[2]
            resultado=np.array([np.array([i for i in zip(A[j],B[j],C[j])]) for j in range(len(A))])
            return resultado

    def RecSample(img,tam=4,affin=3):
        Copia =np.array(img)
        n=np.shape(img)[0]
        m=np.shape(img)[1]
        if(len(np.shape(img))<3):
            l=1
        else:
            l=np.shape(img)[2]

        Amplificar = tf.keras.layers.UpSampling2D(size=(tam,tam))(Copia.reshape([1,n,m,l])).numpy()
        Restablecer = np.array(Amplificar.reshape([n*tam,m*tam,l]))
        Sep=np.array([Restablecer[:,:,i] for i in range(l)])

        for k in range(l):
            for i in range(0,4*m-9,9):
                for j in range(0,4*n-11,8):
                    if(i%2==0):
                        Sep[k][i:i+9,j:(j+8)]=HexImage.recx(Sep[k][i:i+9,j:(j+8)])
                    else:
                        #Sep[k][i:i+9,j+3:(j+11)]=recx(Sep[k][i:i+9,j+3:(j+11)])
                        Sep[k][i:i+9,j+affin:(j+8+affin)]=HexImage.recx(Sep[k][i:i+9,j+affin:(j+8+affin)])

        if(l==1):
            return Sep[0]
        else:
            A=Sep[0]
            B=Sep[1]
            C=Sep[2]
            resultado=np.array([np.array([i for i in zip(A[j],B[j],C[j])]) for j in range(len(A))])
            return resultado
        
    def Cart2Hex(x,y,d=1):
        xs=2*x/d
        ys=2*y/(d*np.sqrt(3))
        xr=int(np.round(xs))
        yr=int(np.round(ys))
        m=1/3
        k=0

        if(xor(xr%2,yr%2)):
            if(xs>=xr):
                if(ys>yr): #Cuadrante1
                    if((ys-(yr+1/3)-m*(xs-xr))<=0):
                        xr=xr+1
                    else:
                        yr=yr+1
                else: #cuadrante2
                    if((ys-(yr-1/3)+m*(xs-xr))>=0):
                        xr=xr+1
                    else:
                        yr=yr-1
                        k=1
            else:
                if(ys>yr):#Cuadrante4
                    if(ys-(yr+1/3)+m*(xs-xr)<=0):
                        xr=xr-1
                    else:
                        yr=yr+1
                else:#Cuadrante3
                    if(ys-(yr-1/3)-m*(xs-xr)>=0):
                        xr=xr-1
                    else:
                        yr=yr-1
        a=yr%2
        r=(yr-a)/2
        c=(xr-a)/2
        if(k==1):
            a=1
        return a,r,c

    def Hex2Cart(a,r,c,d=1):
        y=d*(np.sqrt(3)*(a/2+r))
        if(a==1):
            x=c
        else:
            x=d*(a/2+c)
        return int(round(x)),int(round(y))
