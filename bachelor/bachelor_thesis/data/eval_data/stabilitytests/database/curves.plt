reset

#set term svg font 'Times New Roman' fsize 10 size 300,280

set xrange [0:1]
set yrange [0:1]
set xtic 0,0.2,1
set ytic 0,0.2,1
set xlabel 'Intensity'
set ylabel 'Irradiance'
set grid

set key box
set key top left
unset key
channel = 4

dorf = 'DoRF_mean_curve.txt'
#canon = 'mean_canon_eos_400d_digital_l100_k7_p7.txt'
kodak = 'mean_kodak_dx7590_l10000_k3_p3.txt'
kodakWorst = 'WORST.txt'


plot 'AVERAGE/1.txt' u 1:channel w l, \
	 'AVERAGE/2.txt' u 1:channel w l, \
	 'AVERAGE/3.txt' u 1:channel w l, \
	 'AVERAGE/4.txt' u 1:channel w l, \
	 'AVERAGE/5.txt' u 1:channel w l, \
	 'AVERAGE/6.txt' u 1:channel w l, \
	 'AVERAGE/7.txt' u 1:channel w l, \
	 'AVERAGE/8.txt' u 1:channel w l, \
	 'AVERAGE/9.txt' u 1:channel w l, \
	 'AVERAGE/10.txt' u 1:channel w l, \
	 'AVERAGE/11.txt' u 1:channel w l, \
	 'AVERAGE/12.txt' u 1:channel w l, \
	 'AVERAGE/13.txt' u 1:channel w l, \
	 'AVERAGE/14.txt' u 1:channel w l, \
	 'AVERAGE/15.txt' u 1:channel w l, \
	 'AVERAGE/16.txt' u 1:channel w l, \
	 'AVERAGE/17.txt' u 1:channel w l, \
	 'AVERAGE/18.txt' u 1:channel w l, \
	 'AVERAGE/20.txt' u 1:channel w l, \
	 'AVERAGE/21.txt' u 1:channel w l, \
	 'AVERAGE/22.txt' u 1:channel w l, \
	 'AVERAGE/19.txt' u 1:channel w l