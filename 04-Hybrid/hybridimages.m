img1 = imread('este.jpg');
img2 = imread('javi.jpg');

 
w1 = 35; %win size
s1 = 21; % std dev
h1 = fspecial('gaussian',w1,s1); %gaussian filter with size window w1 and std dev s1

 
w2 = 25;
s2 = 25;
h2 = fspecial('gaussian',w2,s2); %gaussian filter with size window w2 and std dev s2

imgft1 = imfilter(img1,h1,'replicate'); %img filtered with gaussian: low pass image
figure;
imshow(imgft1);
title('low-pass image')
pause

imgft2 = img2 - imfilter(img2,h2,'replicate'); %high pass image
figure;
imshow(imgft2);
title('high-pass image')
pause

img = imgft1 + imgft2; %hybrid image is the sum of LP and HP images

figure
imshow(img)
title('hybrid image')
