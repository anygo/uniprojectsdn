#reset
unset xlabel
unset ylabel
set label 'epsilon_{max}' at 0,2.9,0 rotate by -31
set label 'epsilon_{min}' rotate by 10 at -1.8,2.1,0
set label 'beta' rotate at 1.8,2.2,0.0

set cbrange [0:0.4]
set cbtics ("0%%" 0, "10%%" 0.1, "20%%" 0.2, "30%%" 0.3, "40%%" 0.4 )

set xrange [-0.05:sqrt(3)+0.05]
set yrange [-0.05:sqrt(3)+0.05]

set xtics 0,0.4,1.8
set ytics 0,0.4,1.8
set ztics 0,0.1,0.5
set grid
set dgrid3d 20,20,100
#set contour both
#set cntrparam levels incr 0.1,0.1
set pm3d


file = 'var_diff-stuff.txt'
#set output 'var_diff-klinikum.pdf'

set view 65,240,1,1
unset colorbox
#set multiplot
#set size 1,0.5
#set origin 0,0.5
set surface
#splot file u 1:2:4 w l t '' lt 7 
#set origin 0,0
unset key
unset label
unset surface
set view map
set colorbox
set cblabel 'beta'
set cblabel offset 0.8
set ylabel offset -0.8
set xlabel 'epsilon_{max}'
set ylabel 'epsilon_{min}'
set ytic rotate

unset grid

splot file u 1:2:4 w pm3d
#unset multiplot
#unset output
