set xrange [0:1]
set yrange [0:1]
set xlabel 'Intensity R'
set ylabel 'Irradiance r = g(R)'

set key top left
set key box

plot x t 'ground truth', \
	 'gamma_1_lambda_1_kernels_3_PCA_5.txt' u 1:3 w lines t 'lambda = 1', \
	 'gamma_1_lambda_5313.02_kernels_3_PCA_5.txt' u 1:3 w lines t 'lambda = 5000', \
	 'gamma_1_lambda_9412.34_kernels_3_PCA_5.txt' u 1:3 w lines t 'lambda = 10000', \
	 'gamma_1_lambda_101980_kernels_3_PCA_5.txt' u 1:3 w lines t 'lambda = 100000'