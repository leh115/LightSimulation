import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import math
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
import scipy.fftpack as sfft

ROOTDIR = "C:/Users/Unimatrix Zero/Documents/Uni Masters/Project/"
class LightSim():
    def __init__(self):
        self.xDim = 300
        self.yDim = 300
        self.RealSpaceFactor = 2e-3
        self.Movie_Frames = np.zeros((100,self.xDim,self.yDim))
        self.Movie_count = 0
        self.wavelength = 700e-9
        x = np.array( range(-int(self.xDim/2),int(self.xDim/2),1) )
        x = x*self.RealSpaceFactor
        y = np.array( range(-int(self.yDim/2),int(self.yDim/2),1) )
        y = y*self.RealSpaceFactor
        
        kx,ky = np.divide(2* np.pi, np.meshgrid(x,y) )
        print(x[0:100:5])
        #kx = 2*np.pi/x
        #ky = 2*np.pi/y
        kx[np.isinf(kx)] = 0
        ky[np.isinf(ky)] = 0
        #kx = np.reshape(kx,(self.xDim,1))
        #ky = np.reshape(ky,(1,self.yDim))
        #kxx,kyy = np.meshgrid(kx,ky)
        #self.k = np.add(kxx,kyy)
        self.k = np.add(kx,ky)
        self.bowl = np.add(kx**2, ky**2)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.scatter(kx[0:len(kx):100],ky[0:len(kx):100],kx[0:len(kx):100]**2+ky[0:len(kx):100]**2,marker=".")
        plt.show()


        Grating = self.MakeGrating(len(x),len(x),2,10)
        #cv2.imshow("Grating", Grating)

        X = self.gaussian2d(x,0,0.1,showResults=False)

        X = self.PropagateFreeSpace(1e-4,1,X,showResults=True,saveResults=True)
        X = Grating*X
        X = self.PropagateFreeSpace(1e-4,1,X,showResults=True,saveResults=True)
        
        self.saveAnimation(self.Movie_Frames,"GaussianPropagation")

    def gaussian1d(self,x, A, m, sigma):
        # The formula for a 1 dimensional gaussian
        return A*np.exp(-0.5*(x-m)**2/sigma**2)

    def gaussian2d(self,x,m,sigma,showResults=False):
        # to create a 2D gaussian, two 1D gaussians are combined
        s1 = self.gaussian1d(x,1,m,sigma)
        s1 = s1.reshape((len(s1),1))
        s2 = self.gaussian1d(x,1,m,sigma)
        s2 = s2.reshape(1,(len(s2)))
        if showResults:
            plt.imshow(np.multiply(s1,s2))
            plt.show()
        return np.multiply(s1,s2)
    
    def PropagateFreeSpace(self,d,D,X,showResults=False,saveResults=False):
        # step size is represented by d
        # total distance is represented by D
        kx = (math.pi*2)/self.wavelength
        #kx = 0.1
        for i in range(int(D/d)):
            # The fourier transform to get to k-space
            F = np.fft.fft2(X)

            F = sfft.fftshift(F)
            # Applying the propagation multiplication to the fourier space
            exponentF = -1j*self.k*d + 1j*np.pi*self.wavelength*d*self.bowl
            print(exponentF.shape)
            F[np.isnan(F)] = 0
            KF = F*np.exp(exponentF)
            KF[np.isnan(KF)] = 0
            print(KF.shape)
            # Inversing the result to get back to real space
            X = np.fft.ifft2(KF)

            #X = np.fft.ifftshift(X)
            X[np.isnan(X)] = 0

            #X = np.fft.ifftshift(X)
            if i%2==0:
                if showResults:
                    cv2.imshow("KF",np.real(KF))
                    cv2.waitKey(10)
                if showResults:
                    cv2.imshow("X",np.real(X**2))
                    cv2.waitKey(10)
            else:
                if showResults:
                    cv2.imshow("X2",np.real(X**2))
                    cv2.waitKey(10)
                #if showResults:
                #    cv2.imshow("X shift",np.real(Xs))
                #    cv2.waitKey(100)
                if saveResults:
                    self.Movie_Frames[self.Movie_count,:,:] = np.real(X)
                    self.Movie_count+=1
        
        return X

    def MakeGrating(self,x_len,y_len,barrier_width,separation_width):
        Grating = np.ones((x_len,y_len))
        Period = barrier_width+separation_width
        for i in range(x_len):
            if i%Period <barrier_width:
                Grating[:,i] = 0
        return Grating

    def saveAnimation(self,Images,name):
        Number_frames,width,height = Images.shape   
        #Images = Images/np.amax(Images) # normalisation step

        fourcc = VideoWriter_fourcc(*'mp4v')
        video = VideoWriter(ROOTDIR+"Results/"+name+".mp4", fourcc, 24, (width,height), False)

        for i in range(Number_frames):
            newImg = Images[i,:,:]*255
            video.write(newImg.astype('uint8'))

        video.release()


LS = LightSim()
