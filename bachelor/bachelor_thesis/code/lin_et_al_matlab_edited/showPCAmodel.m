function showPCAmodel()

nPCA = 5;

[g0_paper, H_paper] = importPCA;

H_paper = H_paper(1:nPCA,:);



H_paper = H_paper';
g0 = g0_paper';
g_PCA = g0*H_paper;


Dim = length(g0_paper);
N = size(H_paper,1);
x = 0:1/(Dim-1):1;



fid = fopen('invdorf.txt','r');
NN = 201;
all_curves = zeros(NN,Dim);
figure;
hold;
for i=1:NN
    v = fscanf(fid,'%f ', Dim);
    if (i == 19 || i == 172)
        plot(v);
    end
    all_curves(i,:) = v';
end
fclose(fid);

c0_paper = all_curves*H_paper;
for i=1:NN
    c0 = c0_paper(i,:);
    g(i,:) = g_PCA + c0;
end

plot(g');
title '201 inverse response functions - representation'


fp = fopen('invdorf_pca.txt','w+');
fprintf(fp,'%d %d\r\n',NN,nPCA);
for i=1:NN
    for j=1:nPCA
        fprintf(fp, '%f ', c0_paper(i, j));
    end
    fprintf(fp, '\r\n');
end        
fclose(fp);