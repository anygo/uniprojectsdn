set xrange [0:1]
set yrange [0:1]
set zrange [0:1]
unset xtic
unset ytic
unset ztic
set grid
set box 5

set view 59,26,1.09870,1.06299

set key box
set key title 'gamma'
set key at 0.3,0.3,-0.4

set arrow from 0.235294,0.823529,0.823529 to 0.560589,0.925277,0.925277
set arrow from 0.823529,0.235294,0.235294 to 0.925277,0.560589,0.560589


splot 'rgb.txt' w linesp pt 6 t '1.0', \
	  'rgb_gamma0.4.txt' w linesp pt 6 t '0.4'