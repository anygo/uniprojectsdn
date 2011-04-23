set xrange [1:101000]
set log x

plot 'kernels3components6.txt' u 2:3 w l, \
	 'kernels4components6.txt' u 2:3 w l, \
	 'kernels5components6.txt' u 2:3 w l, \
	 'kernels6components6.txt' u 2:3 w l, \
	 'kernels7components6.txt' u 2:3 w l, \
	 'kernels8components6.txt' u 2:3 w l, \
	 'kernels9components6.txt' u 2:3 w l 