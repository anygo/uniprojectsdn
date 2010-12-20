clear all;
close all;
clc;

% signal from exercise
signal = [1 1 -1 0 0 -1 1 1];

subplot(2, 2, 1);
stem(0:7, signal);
title('signal'), axis([0 7 -1 1]);

subplot(2, 2, 2);
had = hadamard(8);
imagesc(had);
title('Hadamard matrix'), colormap gray;

subplot(2, 2, 3);
stem(0:7, had * signal');
title('Walsh Hadamard Transform'), axis([0 7 -6 6]);