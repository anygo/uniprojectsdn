reset
set xlabel 'lambda'
set ylabel 'error function value'
set xrange [1:101980]
set log x


set key box
set key top left
#set key title 'gamma'

#set terminal pdfcairo font 'times, 8' size 8in,5in

#set output 'kernels5components5.pdf'
#set arrow from 332,0 to 332,80 nohead lt 9
#set label '332' at 245,-4.25

#f(x) = a + b*x + c*x*x + d*x*x*x + e*x*x*x*x
#fit f(x) file via a,b,c,d,e

plot  \
	 'kernels7components7crippled.txt'u 2:3 with l t '7 7', \
	 'kernels9components9crippled.txt'u 2:3 with l t '9 9', \
	 'kernels3components5crippled.txt'u 2:3 with l t '3 5', \
	 'kernels5components5crippled.txt'u 2:3 with l t '5 5', \
	 'kernels7components5crippled.txt'u 2:3 with l t '7 5'
#unset output
