#reset
set xlabel 'R'
set ylabel 'Q'
set xrange [0:1]
set yrange [0:1.5]
set dgrid3d 25,25,10
set pm3d
unset surface
unset key
unset grid
set view map


file = '0.2.txt'

splot file u 2:1:3 w pm3d

