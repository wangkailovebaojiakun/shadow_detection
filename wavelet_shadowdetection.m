clc;
background = imread('img/background.bmp');
frame = imread('img/357.jpg');
% Step 1. Represent the reference frame and the current frame in HSV colour model.
reference_frame_hsv = rgb2hsv(background);
current_frame_hsv =rgb2hsv(frame);
% ��ʾ������ͼƬ����
imshow_flag = 0;
if(imshow_flag)
    figure;
    subplot(1,2,1);imshow(reference_frame_hsv);title('HSV�ռ�reference frameͼ��');
    subplot(1,2,2);imshow(current_frame_hsv);title('HSV�ռ�current frameͼ��');
end 

% Step 2. Take absolute difference of reference frame and
% current frame with respect to Hue, saturation and value
% component (��H, ��V and ��S), respectively.
im_absdiff_hsv = imabsdiff(reference_frame_hsv,current_frame_hsv);
H = im_absdiff_hsv(:,:,1);
S = im_absdiff_hsv(:,:,2);
V = im_absdiff_hsv(:,:,3);
% 
% if(imshow_flag)
    figure;
    subplot(1,3,1);imshow(H);title('HSV�ռ�Hͨ��ͼ��');
    subplot(1,3,2);imshow(S);title('HSV�ռ�Sͨ��ͼ��');
    subplot(1,3,3);imshow(V);title('HSV�ռ�Vͨ��ͼ��');
% end 
% ��V��Sͨ��������ֵ����
 


% th_V = mean2(V);
% th_S = mean2(PSF_S);
% thed_V = im2bw(V,th_V);
% thed_S = im2bw(PSF_S,th_S);
% 
%     figure;
%     subplot(1,2,1);imshow(thed_V);title('Vͨ����ֵ��ֵͼ��');
%     subplot(1,2,2);imshow(thed_S);title('Sͨ����ֵ��ֵͼ��');

% % Step 3. Perform wavelet decomposition on difference image
% % of value and saturation component (��V and ��S) using
% % DWT. The wavelet coefficients are W��V and W��S.
%  
[cA_V,cH_V,cV_V,cD_V]= dwt2(V,'haar');
[cA_S,cH_S,cV_S,cD_S]= dwt2(S,'haar');

% �����ֵ�˲��Ͳ��Ӿ�ֵ�˲�
% 
PSF = fspecial('average',3);
% cA_V = imfilter(cA_V,PSF);
cA_S = imfilter(cA_S,PSF);
% ���
% iA = [];
%
figure;
subplot(1,4,1);imshow(cA_V);title('Vͨ��С��cAϵ��ͼ��');
subplot(1,4,2);imshow(cH_V);title('Vͨ��С��cHϵ��ͼ��');
subplot(1,4,3);imshow(cV_V);title('Vͨ��С��cVϵ��ͼ��');
subplot(1,4,4);imshow(cD_V);title('Vͨ��С��cDϵ��ͼ��'); 
figure;
subplot(1,4,1);imshow(cA_S);title('Sͨ��С��cAϵ��ͼ��');
subplot(1,4,2);imshow(cH_S);title('Sͨ��С��cHϵ��ͼ��');
subplot(1,4,3);imshow(cV_S);title('Sͨ��С��cVϵ��ͼ��');
subplot(1,4,4);imshow(cD_S);title('Sͨ��С��cDϵ��ͼ��'); 

W_V = [cA_V,cH_V,cV_V,cD_V];
W_S = [cA_S,cH_S,cV_S,cD_S];
A_V = cA_V;
H_V = cH_V;
V_V = cV_V; 
D_V = cD_V;
A_S = cA_S;
H_S = cH_S;
V_S = cV_S; 
D_S = cD_S;
% 
A_V2 = cA_V;
H_V2 = cH_V;
V_V2 = cV_V; 
D_V2 = cD_V;
A_S2 = cA_S;
H_S2 = cH_S;
V_S2 = cV_S; 
D_S2 = cD_S;
% 
% % Step 4. Compute coefficient of variation of wavelet
% % coefficients of ��V and ��S say (s/m)W��V
% % and (s/m)W��S.
% % ���������⣬û��˵����Ǽ����ȫ��С��ϵ������Ա�׼��ǲ�ͬ�Ӵ��ϵ���Ա�׼��
% % �������ȫ��С��ϵ������Ա�׼����м��㣬���ݹ�ʽ(1)�����жϺ�Ľ��ȫ����0
% % �������￼�ǣ�Ӧ���ǰ��ո��Ӵ��ֱ���м���
% % ����ȫ���Ӵ�
% stdW_V = std2(W_V);
% stdW_S = std2(W_S);
% meanW_V = mean2(W_V);
% meanW_S = mean2(W_S);
% RSD_V = stdW_V/meanW_V;
% RSD_S = stdW_S/meanW_S;
% �ֽ����4���Ӵ�
% V����4���Ӵ�
stdW_cA_V = std2(cA_V);
stdW_cH_V = std2(cH_V);
stdW_cV_V = std2(cV_V);
stdW_cD_V = std2(cD_V);
meanW_cA_V = mean2(cA_V);
meanW_cH_V = mean2(cH_V);
meanW_cV_V = mean2(cV_V);
meanW_cD_V = mean2(cD_V);
% RSD_cA_V = stdW_cA_V/meanW_cA_V;
% RSD_cH_V = stdW_cH_V/meanW_cH_V;
% RSD_cV_V = stdW_cV_V/meanW_cV_V;
% RSD_cD_V = stdW_cD_V/meanW_cD_V;
thcA_V = stdW_cA_V + meanW_cA_V;
% thcA_V = stdW_cA_V;
thcH_V = stdW_cH_V + meanW_cH_V;
thcV_V = stdW_cV_V + meanW_cV_V;
thcD_V = stdW_cD_V + meanW_cD_V;
% S����4���Ӵ�
stdW_cA_S = std2(cA_S);
stdW_cH_S = std2(cH_S);
stdW_cV_S = std2(cV_S);
stdW_cD_S = std2(cD_S);
meanW_cA_S = mean2(cA_S);
meanW_cH_S = mean2(cH_S);
meanW_cV_S = mean2(cV_S);
meanW_cD_S = mean2(cD_S);
% RSD_cA_S = stdW_cA_S/meanW_cA_S;
% RSD_cH_S = stdW_cH_S/meanW_cH_S;
% RSD_cV_S = stdW_cV_S/meanW_cV_S;
% RSD_cD_S = stdW_cD_S/meanW_cD_S;
thcA_S = stdW_cA_S + meanW_cA_S;
% thcA_S = stdW_cA_S;
thcH_S = stdW_cH_S + meanW_cH_S;
thcV_S = stdW_cV_S + meanW_cV_S;
thcD_S = stdW_cD_S + meanW_cD_S;
% 
% % Step 5. Check condition for shadow detection (detection of
% % foreground object with shadow) using (1).
% % ��V�����Ͻ��м���


[M,N] = size(cA_V);
disp([M,N]);
for x = 1:M
    for y = 1:N
        if(abs(cA_V(x,y)) >= thcA_V)
            cA_V(x,y) =1;
        else
            cA_V(x,y) = 0;
        end
    end
end
[M,N] = size(cH_V);
for x = 1:M
    for y = 1:N
        if(abs(cH_V(x,y)) >= thcH_V)
            cH_V(x,y) =1;
        else
            cH_V(x,y) = 0;
        end
    end
end
[M,N] = size(cV_V);
for x = 1:M
    for y = 1:N
        if(abs(cV_V(x,y)) >= thcV_V)
            cV_V(x,y) =1;
        else
            cV_V(x,y) = 0;
        end
    end
end
[M,N] = size(cD_V);

for x = 1:M
    for y = 1:N
        if(abs(cD_V(x,y)) >= thcD_V)
            cD_V(x,y) =1;
        else
            cD_V(x,y) = 0;
        end
    end
end

% ��ʾ������ͼƬ
figure;
subplot(1,4,1);imshow(cA_V);title('Vͨ��С��cAϵ����ֵ��ֵͼ��');
subplot(1,4,2);imshow(cH_V);title('Vͨ��С��cHϵ����ֵ��ֵͼ��');
subplot(1,4,3);imshow(cV_V);title('Vͨ��С��cVϵ����ֵ��ֵͼ��');
subplot(1,4,4);imshow(cD_V);title('Vͨ��С��cDϵ����ֵ��ֵͼ��'); 

% 
figure;
imshow(cA_V);title('Vͨ��С��cA&cVϵ����ֵ��ֵͼ��');
% ��S�����Ͻ��м���Ľ��

[M,N] = size(cA_S);
for x = 1:M
    for y = 1:N
        if(abs(cA_S(x,y)) >= thcA_S)
            cA_S(x,y) =1;
        else
            cA_S(x,y) = 0;
        end
    end
end
[M,N] = size(cH_S);
for x = 1:M
    for y = 1:N
        if(abs(cH_S(x,y)) >= thcH_S)
            cH_S(x,y) =1;
        else
            cH_S(x,y) = 0;
        end
    end
end
[M,N] = size(cV_S);
for x = 1:M
    for y = 1:N
        if(abs(cV_S(x,y)) >= thcV_S)
            cV_S(x,y) =1;
        else
            cV_S(x,y) = 0;
        end
    end
end
[M,N] = size(cD_S);
for x = 1:M
    for y = 1:N
        if(abs(cD_S(x,y)) >= thcD_S)
            cD_S(x,y) =1;
        else
            cD_S(x,y) = 0;
        end
    end
end
% ��ʾ������ͼƬ
figure;
subplot(1,4,1);imshow(cA_S);title('Sͨ��С��cAϵ����ֵ��ֵͼ��');
subplot(1,4,2);imshow(cH_S);title('Sͨ��С��cHϵ����ֵ��ֵͼ��');
subplot(1,4,3);imshow(cV_S);title('Sͨ��С��cVϵ����ֵ��ֵͼ��');
subplot(1,4,4);imshow(cD_S);title('Sͨ��С��cDϵ����ֵ��ֵͼ��'); 
% 
% % Step 6. Check condition for shadow removal (detection of
% % foreground object without shadow) using (2).
B = [0,1,0;
    1,1,1;
    0,1,0];

index_img = cH_S+cV_S+cD_S+cH_V+cV_V+cD_V;
std_index_img = std2(index_img);
mean_index_img = mean2(index_img);
th_index = mean_index_img + std_index_img;
cA_index = cA_S+cA_V;
figure;
imshow(cA_index);title('cA_index');
             
cA_index2 = imfill(cA_index);
index_img2 = imfill(index_img);

[M,N] = size(index_img);
for x = 1:M
    for y = 1:N
        if(index_img(x,y) <th_index)
            index_img(x,y) =0;
        end
    end
end
figure;
imshow(index_img);title('index_img');

% index_img = imfilter(index_img,PSF);

result_img = cA_index2&index_img;

figure;
imshow(index_img);title('��ֵ�˲���Ľ�� index_img');
figure;
subplot(1,3,1);imshow(cA_index2);title('cA_index2��ֵͼ��');
subplot(1,3,2);imshow(index_img);title('index_img��ֵͼ��');
subplot(1,3,3);imshow(result_img);title('result_img��ֵͼ��');

% ��ʾ������ͼƬ
% figure;
% subplot(1,4,1);imshow(D_1);title('�ϲ����С��cAϵ����ֵ��ֵͼ��');
% subplot(1,4,2);imshow(D_2);title('�ϲ����С��cHϵ����ֵ��ֵͼ��');
% subplot(1,4,3);imshow(D_3);title('�ϲ����С��cVϵ����ֵ��ֵͼ��');
% subplot(1,4,4);imshow(D_4);title('�ϲ����С��cDϵ����ֵ��ֵͼ��'); 

% % % Step 7. Reconstruct shadow detected image (foreground
% % % object with shadow) and shadow removed image
% % % (foreground object without shadow) using inverse wavelet
% % % transform.
% % % ����û��˵�����ô�ع�,�ع�����Ҫ���õ�ͼ����V��������S����
% % % �ع����ͼ����ʲô���͵�
% % % �ع���Ҫ�ڶ�ֵͼ���Ͻ��л��Ǳ���ԭ����ϵ������
% % % �Դ����Ĳ�����Ӱͼ������ع�
% [M,N] = size(A_V);
% for x = 1:M
%     for y = 1:N
%         if(D_1(x,y) == 0)
%             A_V(x,y) =0;
%             A_S(x,y) =0;
%         end
%     end
% end
% for x = 1:M
%     for y = 1:N
%         if(D_2(x,y) == 0)
%             H_V(x,y) =0;
%             H_S(x,y) =0;
%         end
%     end
% end
% for x = 1:M
%     for y = 1:N
%         if(D_3(x,y) == 0)
%             V_V(x,y) =0;
%             H_S(x,y) =0;
%         end
%     end
% end
% for x = 1:M
%     for y = 1:N
%         if(D_4(x,y) == 0)
%             D_V(x,y) =0;
%             D_S(x,y) =0;
%         end
%     end
% end
% % ��ʾ������ͼƬ
% figure;
% subplot(1,4,1);imshow(A_V);title('������Ӱ�ع�ǰV����С��cDϵ��ͼ��');
% subplot(1,4,2);imshow(H_V);title('������Ӱ�ع�ǰV����С��cHϵ��ͼ��');
% subplot(1,4,3);imshow(V_V);title('������Ӱ�ع�ǰV����С��cVϵ��ͼ��');
% subplot(1,4,4);imshow(D_V);title('������Ӱ�ع�ǰV����С��cDϵ��ͼ��'); 
% img_withoutshadow_V =  idwt2(A_V,H_V,V_V,D_V,'haar');
% figure;
% imshow(img_withoutshadow_V);title('������Ӱ�ع���V����ͼ��')
% figure;
% subplot(1,4,1);imshow(A_S);title('������Ӱ�ع�ǰS����С��cDϵ��ͼ��');
% subplot(1,4,2);imshow(H_S);title('������Ӱ�ع�ǰS����С��cHϵ��ͼ��');
% subplot(1,4,3);imshow(V_S);title('������Ӱ�ع�ǰS����С��cVϵ��ͼ��');
% subplot(1,4,4);imshow(D_S);title('������Ӱ�ع�ǰS����С��cDϵ��ͼ��');
% img_withoutshadow_S =  idwt2(A_S,H_S,V_S,D_S,'haar');
% figure;
% imshow(img_withoutshadow_S);title('������Ӱ�ع���S����ͼ��');
% % �԰�����Ӱ��ͼ���ع�
% for x = 1:M
%     for y = 1:N
%         if(cA_V(x,y) == 0)
%             A_V2(x,y) =0;
%             A_S2(x,y) =0;
%         end
%     end
% end
% for x = 1:M
%     for y = 1:N
%         if(cH_V(x,y) == 0)
%             H_V2(x,y) =0;
%             H_S2(x,y) =0;
%         end
%     end
% end
% for x = 1:M
%     for y = 1:N
%         if(cV_V(x,y) == 0)
%             V_V2(x,y) =0;
%             H_S2(x,y) =0;
%         end
%     end
% end
% for x = 1:M
%     for y = 1:N
%         if(cD_V(x,y) == 0)
%             D_V2(x,y) =0;
%             D_S2(x,y) =0;
%         end
%     end
% end
% % ��ʾ������ͼƬ
% figure;
% subplot(1,4,1);imshow(A_V2);title('����Ӱ�ع�ǰV����С��cDϵ��ͼ��');
% subplot(1,4,2);imshow(H_V2);title('����Ӱ�ع�ǰV����С��cHϵ��ͼ��');
% subplot(1,4,3);imshow(V_V2);title('����Ӱ�ع�ǰV����С��cVϵ��ͼ��');
% subplot(1,4,4);imshow(D_V2);title('����Ӱ�ع�ǰV����С��cDϵ��ͼ��'); 
% img_withshadow_V =  idwt2(A_V2,H_V2,V_V2,D_V2,'haar');
% figure;
% imshow(img_withshadow_V);title('����Ӱ�ع���V����ͼ��')
% figure;
% subplot(1,4,1);imshow(A_S2);title('����Ӱ�ع�ǰS����С��cDϵ��ͼ��');
% subplot(1,4,2);imshow(H_S2);title('����Ӱ�ع�ǰS����С��cHϵ��ͼ��');
% subplot(1,4,3);imshow(V_S2);title('����Ӱ�ع�ǰS����С��cVϵ��ͼ��');
% subplot(1,4,4);imshow(D_S2);title('����Ӱ�ع�ǰS����С��cDϵ��ͼ��');
% img_withshadow_S =  idwt2(A_S2,H_S2,V_S2,D_S2,'haar');
% figure;
% imshow(img_withshadow_S);title('����Ӱ�ع���S����ͼ��');
% % Step 8. Apply binary closing morphological operation on
% % shadow detected image and shadow removed image to
% % obtain smooth image.
% % ������ͼ����ж�ֵ��
% A1 = img_withshadow_V;
% A2 = img_withshadow_S;
% A3 = img_withoutshadow_V;
% A4 = img_withoutshadow_S;
% ALL = [A1,A2,A3,A4];

% ������ֵ����

 
% thresh = std2(ALL)+mean2(ALL);
% thresh = graythresh(A1);     %�Զ�ȷ����ֵ����ֵ
% I1 = im2bw(A1,thresh);       %��ͼ���ֵ��
% thresh = graythresh(A2);     %�Զ�ȷ����ֵ����ֵ
% I2 = im2bw(A2,thresh);       %��ͼ���ֵ��
% thresh = graythresh(A3);     %�Զ�ȷ����ֵ����ֵ
% I3 = im2bw(A3,thresh);       %��ͼ���ֵ��
% thresh = graythresh(A4);     %�Զ�ȷ����ֵ����ֵ
% I4 = im2bw(A4,thresh);       %��ͼ���ֵ����
% % ��ʾ������ͼƬ
% figure;
% subplot(1,4,1);imshow(I1);title('����ӰV������ֵ��');
% subplot(1,4,2);imshow(I2);title('����ӰS������ֵ��');
% subplot(1,4,3);imshow(I3);title('������ӰV������ֵ��');
% subplot(1,4,4);imshow(I4);title('������ӰS������ֵ��');
% 
% 
% B = [0,1,0;
%     1,1,1;
%     0,1,0];
% A1 = imclos0e(I1,B);%ͼ�񱻽ṹԪ��B����
% A2 = imclose(I2,B);
% A3 = imclose(I3,B);
% A4 = imclose(I4,B);
% % ��ʾ������ͼƬ
% figure;
% subplot(1,4,1);imshow(A1);title('����ӰV�����������');
% subplot(1,4,2);imshow(A2);title('����ӰS�����������');
% subplot(1,4,3);imshow(A3);title('������ӰV�����������');
% subplot(1,4,4);imshow(A4);title('������ӰS�����������');

% �����ر�����figure
pause;
 close all;
