reset
set xlabel 'lambda'
set ylabel 'error function value'
set yrange [0:0.078]
set xrange [1:101980]
set log x
set xtic offset -1

set key box
set key title 'gamma'

#set terminal svg fname 'Times New Roman' fsize 12 size 600,400 

file = 'kernels5components7.txt'
#set output 'kernels5components5.pdf'
#set arrow from 332,0 to 332,80 nohead lt 9
#set label '332' at 245,-4.25

#f(x) = a + b*x + c*x*x + d*x*x*x + e*x*x*x*x
#fit f(x) file via a,b,c,d,e

plot file every :::0::0 u 2:($3/1024.) with l t '0.2', \
	 file every :::1::1 u 2:($3/1024.) with l t '0.4', \
	 file every :::2::2 u 2:($3/1024.) with l t '0.6', \
	 file every :::3::3 u 2:($3/1024.) with l t '0.8', \
	 file every :::4::4 u 2:($3/1024.) with l t '1.0'
unset output
