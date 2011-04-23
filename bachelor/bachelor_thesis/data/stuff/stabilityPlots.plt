reset

set xrange [50:20000]
set yrange [0:0.008]
set log x
set key top left
set key box
set ytics 0,0.002
set xlabel 'lambda'
set ylabel 'mu_{joint}'


file = 'Canon EOS 400D Rebel XTi55100.txt'
#file = 'Sony CyberSHot DSC-W30055100.txt'

plot file w errorbars t 'measurements', \
	 file w lines t 'interpolation'