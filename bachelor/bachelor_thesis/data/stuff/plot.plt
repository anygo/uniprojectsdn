file = 'means.txt'

set box

#plot file every 9::1 u 2 w lp, \
#	 file every 9::2 u 2 w lp, \
#	 file every 9::3 u 2 w lp, \
#	 file every 9::4 u 2 w lp, \
#	 file every 9::5 u 2 w lp, \
#	 file every 9::6 u 2 w lp, \
#	 file every 9::7 u 2 w lp, \
#	 file every 9::8 u 2 w lp, \
#	 file every 9::9 u 2 w lp
	 
	 
	 set key box
	 set key title 'Camera'
	 
plot file every :::0::0 u 2 w lp t 'Canon EOS 400D',\
	 file every :::1::1 u 2 w lp t 'Casio EX s-600', \
	 file every :::2::2 u 2 w lp t 'FujuFilm s5600', \
	 file every :::3::3 u 2 w lp t 'Kodak DX7590', \
	 file every :::4::4 u 2 w lp t 'Kodak Z740', \
	 file every :::5::5 u 2 w lp t 'Nikon D40', \
	 file every :::6::6 u 2 w lp t 'Nikon D80', \
	 file every :::7::7 u 2 w lp t 'Sony DSC w300', \
	 file every :::8::8 u 2 w lp t 'Funt Database'