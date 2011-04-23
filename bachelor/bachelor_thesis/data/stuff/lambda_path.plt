# set terminal png transparent nocrop enhanced font arial 8 size 420,320 
# set output 'random.3.png'
set dummy u,v
unset key
set parametric
#set view 68, 28, 1, 1
set samples 250, 250
set isosamples 250, 250
set contour base
unset clabel
set hidden3d offset 1 trianglepattern 3 undefined 1 altdiagonal bentover
set cntrparam levels incr 0, 0.03, 10
set style function dots
#set ticslevel 0
#set ztics border in scale 1,0.5 nomirror norotate  offset character 0, 0, 0 0.00000,0.05
#set title "50 random samples from a 2D Gaussian PDF with\nunit variance, zero mean and no dependence" 
set urange [ -25.00000 : 10.00000 ] noreverse nowriteback
set vrange [ -15.00000 : 15.00000 ] noreverse nowriteback
set xrange [ -25.00000 : 10.00000 ] noreverse nowriteback
set yrange [ -15.00000 : 15.00000 ] noreverse nowriteback
#set zrange [ -0.200000 : 0.200000 ] noreverse nowriteback
tstring(n) = sprintf("%d random samples from a 2D Gaussian PDF with\nunit variance, zero mean and no dependence", n)
nsamp = 50

set pm3d at b

unset key
unset label
unset surface
set view map
unset xlabel 
unset ylabel 
unset xtic
unset ytic
unset colorbox


splot u,v,( 1/(2*pi) * exp(-0.5 * ((0.4*u+0.1*v-2)**2 + (v-5)**2)) ) + \
		  ( 1/(2*pi) * exp(-0.09 * ((0.9*u)**2 + 0.5*(1.2*v+0.9*u)**2)) ) + \
		  ( 1/(2*pi) * exp(-0.04 * ((0.8*u-0.3*v+6)**2 + (5*v+u+10)**2)) ) + \
		  ( 1/(2*pi) * exp(-0.1 * ((0.6*u+8)**2 + (0.9*v+5)**2)) ) with line lc rgb "black"
		  
		  