set style fill solid 0.25 noborder

plot 'featureSynthetic0.txt' u 1:3 w boxes t 'non-LPIP', \
	 'featureSynthetic0.txt' u 1:2 w boxes t 'LPIP'