
#Q1 Grayscale to Binary
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

img=cv2.imread('quiet.png',cv2.IMREAD_GRAYSCALE)
mean_thresh=img.mean()
_,binary_mean=cv2.threshold(img,mean_thresh,255,cv2.THRESH_BINARY)

user_threshold=int(input("enter threshold value(0-255)"))
_,binary_user=cv2.threshold(img,user_threshold,255,cv2.THRESH_BINARY)

cv2_imshow(img)
cv2_imshow(binary_mean)
cv2_imshow(binary_user)

#Q2 RGB to Grayscale
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

img=cv2.imread('quiet.png')
b,g,r=cv2.split(img)
#case 1
gray1=((r+g+b)/3).astype(np.uint8)

#case 2
wr=float(input("red"))
wg=float(input("green"))
wb=float(input("blue"))
gray2=(wr*r+wg*g+wb*b).astype(np.uint8)

cv2_imshow(img)
cv2_imshow(gray1)
cv2_imshow(gray2)

#Q3 PUT BORDER
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
img=cv2.imread('quiet.png',cv2.IMREAD_GRAYSCALE)

border_width=int(input("border width:"))
border_colour=int(input("border color"))

bordered=cv2.copyMakeBorder(img,border_width,border_width,border_width,border_width,cv2.BORDER_CONSTANT,value=border_colour)
cv2_imshow(img)
cv2_imshow(bordered)

cv2.waitKey(0)
cv2.destroyAllWindows()

#Q4 Complement
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

img=cv2.imread('quiet.png')
complement=255-img
cv2_imshow(img)
cv2_imshow(complement)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Q5
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
img=cv2.imread('quiet.png',cv2.IMREAD_GRAYSCALE)

c=1
log_tf=c*np.log1p(img.astype(np.float32))
log_tf=cv2.normalize(log_tf,None,0,255,cv2.NORM_MINMAX)
log_tf=np.uint8(log_tf)

cv2_imshow(img)
cv2_imshow(log_tf)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Q6
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
img=cv2.imread('quiet.png',cv2.IMREAD_GRAYSCALE)

c=1
gamma=float(input("enter gamma vakue"))
power_tf=c*np.power(img/255.0,gamma)
power_tf=np.uint8(power_tf*255)
cv2_imshow(img)
cv2_imshow(power_tf)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Q7
img=cv2.imread('quiet.png',cv2.IMREAD_GRAYSCALE)

r1=int(input("enter r1: "))
s1=int(input("enter s1: "))
r2=int(input("enter r2: "))
s2=int(input("enter s2: "))

out=np.zeros_like(img,dtype=np.float32)
out=np.zeros_like(img,dtype=np.float32)
out = ((img - r1) * ((s2 - s1) / (r2 - r1))) + s1
out=np.clip(out,0,255).astype(np.uint8)
out=np.clip(out,1,255).astype(np.uint8)

cv2_imshow(img)
cv2_imshow(out)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Q8
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

img=cv2.imread('quiet.png',cv2.IMREAD_GRAYSCALE)
eq_hist=cv2.equalizeHist(img)
cv2_imshow(img)
cv2_imshow(eq_hist)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Q9
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
from skimage.exposure import match_histograms

inim=cv2.imread('quiet.png',cv2.IMREAD_GRAYSCALE)
refim=cv2.imread('independence day.jpg',cv2.IMREAD_GRAYSCALE)
output=match_histograms(inim,refim)
cv2_imshow(inim)
cv2_imshow(refim)
cv2_imshow(output.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()


#Q10
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
img = cv2.imread('quiet.png', cv2.IMREAD_GRAYSCALE)

#inuts
pads=int(input("enter padding: 1-zeropadding, 2-Replicatepadding"))
if pads==1:
  pad_type=cv2.BORDER_CONSTANT
else:
  pad_type=cv2.BORDER_REPLICATE

f=int(input("enter filter size in odd numbers"))
sigma=float(input("enter sigma for gaussian filter"))

#Averaging filter
avg_f=np.ones((f,f),np.float32)/(f*f)
avg_img=cv2.filter2D(img,-1,avg_f,borderType=pad_type)

#Custom filter
custom_filter=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],dtype=np.float32)
custom_img=cv2.filter2D(img,-1,custom_filter,borderType=pad_type)

#Gaussian filter
gaussian_img=cv2.GaussianBlur(img,(f,f),sigma,borderType=pad_type)


cv2_imshow(img)
cv2_imshow(avg_img)
cv2_imshow(custom_img)
cv2_imshow(gaussian_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

#Q11
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

img=cv2.imread('quiet.png',cv2.IMREAD_GRAYSCALE)
if img is None:
  print("error")
else:
  print("load")


#inputs
pad=int(input("enter padding 1-o,2-replicate"))
pad_type=cv2.BORDER_CONSTANT if pad==1 else cv2.BORDER_REPLICATE
k=int(input("enter kernel size(odd num)"))
sigma=float(input("enter sigma for Gaussian unmaksing"))

#laplacian
laplacian=cv2.Laplacian(img,cv2.CV_64F,ksize=k,borderType=pad_type)
laplacian=np.uint8(np.absolute(laplacian))

#sobel
sobel_x=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=k,borderType=pad_type)
sobel_x=np.uint8(np.absolute(sobel_x))

sobel_y=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=k,borderType=pad_type)
sobel_y=np.uint8(np.absolute(sobel_y))

#canny
canny_edges=cv2.Canny(img,100,200)

#unsharpmaksing
blur=cv2.GaussianBlur(img,(k,k),sigma,borderType=pad_type)
mask=img-blur
unsharp=img+1.5*mask
unsharp=np.clip(unsharp,0,255).astype(np.uint8)

cv2_imshow(img)
cv2_imshow(laplacian)
cv2_imshow(sobel_x)
cv2_imshow(sobel_y)
cv2_imshow(canny_edges)
cv2_imshow(unsharp)
cv2.waitKey(0)
cv2.destroyAllWindows()
[for gaussian blur,k=5,sigma =3]
