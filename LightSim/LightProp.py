import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import scipy.fftpack as sfft

ROOTDIR = "C:/Users/Unimatrix Zero/Documents/Uni Masters/Project/"
class LightSim():
    def __init__(self):
        self.xDim = 500
        self.yDim = 500
        self.zObjects = []
        self.RealSpaceFactor = 1e-3
        self.wavelength = 700e-9
        x = np.array( range(-int(self.xDim/2),int(self.xDim/2),1) )
        x = x*self.RealSpaceFactor
        y = np.array( range(-int(self.yDim/2),int(self.yDim/2),1) )
        y = y*self.RealSpaceFactor

        g0,g1 = np.meshgrid(x,y)
        g0[np.where(g0==0)] = self.RealSpaceFactor
        g1[np.where(g1==0)] = self.RealSpaceFactor

        #self.bowl = 100000 * np.add(np.multiply(g0,g0), np.multiply(g1,g1))
        self.bowl = np.divide(2* np.pi, np.add(np.multiply(g0,g0), np.multiply(g1,g1)) )
        #self.bowl[np.isinf(self.bowl)] = 0

        #self.k = np.divide(2* np.pi, np.add(g0,g1))
        self.k = np.divide(2* np.pi, self.RealSpaceFactor * np.ones((self.xDim,self.yDim)) )
        print(self.k)
        print(self.bowl)
        cv2.imshow("K",self.k)

        #cv2.imshow("bowl",self.bowl/(self.RealSpaceFactor*50)) # show the bowl (s1**2 + s2**2)
        #cv2.waitKey(0)

        X = self.gaussian2d(x,0,0.005)
        cv2.imshow("Gaussian",X)

        X = self.PropagateFreeSpace(0.1,100,X,showResults=True,saveResults=True)

        #Grating = self.MakeGrating(len(x),len(x),3,10)

        #X = Grating*X

        #X = self.PropagateFreeSpace(0.1,1000000,X,showResults=True,saveResults=True)

    def gaussian1d(self,x, A, m, sigma):
        # The formula for a 1 dimensional gaussian
        return A*np.exp(-0.5*(x-m)**2/sigma**2)

    def gaussian2d(self,x,m,sigma,showResults=False):
        # to create a 2D gaussian, two 1D gaussians are combined
        s1 = self.gaussian1d(x,3,m,sigma)
        s1 = s1.reshape((len(s1),1))
        s2 = self.gaussian1d(x,3,m,sigma)
        s2 = s2.reshape(1,(len(s2)))
        if showResults:
            plt.imshow(np.multiply(s1,s2))
            plt.show()
        return np.multiply(s1,s2)
    
    def PropagateFreeSpace(self,d,D,X,showResults=False,saveResults=False):
        # step size is represented by d
        # total distance is represented by D
        
        for i in range(int(D/d)):
            # The fourier transform to get to k-space
            #F = np.fft.fft2(X)
            F = sfft.fft2(X)
            F = sfft.fftshift(F)
            # Applying the propagation multiplication to the fourier space
            exponentF = -1j*self.k*d + 1j*np.pi*self.wavelength*d*self.bowl
            #print(exponentF)
            F[np.isnan(F)] = 0
            KF = F*np.exp(exponentF)
            KF[np.isnan(KF)] = 0
            print(KF.shape)
            KF = sfft.fftshift(KF)
            # Inversing the result to get back to real space
            X = np.fft.ifft2(KF)

            X[np.isnan(X)] = 0

            #X = sfft.ifftshift(X)
            if showResults:
                cv2.imshow("Fourier of Gaussian",np.abs(F**2))
                cv2.waitKey(10)
            if showResults:
                cv2.imshow("KF",np.abs(KF))
                cv2.waitKey(10)
            if showResults:
                cv2.imshow("X",np.abs(X**2))
                cv2.waitKey(10)

            # Apply lens effects and other transforms
            #for zlength in self.zObjects:
            #    if (i + 1)*d > zlength:

        
        return X

    def MakeGrating(self,x_len,y_len,barrier_width,separation_width):
        Grating = np.ones((x_len,y_len))
        Period = barrier_width+separation_width
        for i in range(x_len):
            if i%Period <barrier_width:
                Grating[:,i] = 0
        return Grating


LS = LightSim()
