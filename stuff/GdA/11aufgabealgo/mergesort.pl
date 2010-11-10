#!/usr/bin/perl

use strict;
use warnings;


my @array=qw(9 8 6 98 43 12 59 52 4 5 14 2 92 3 32 54 22 41 7 34 15 3 1 13 99 42 63 34);

my $arraylength = scalar @array;

print "numbers before mergesort:";
for (my $i = 0; $i < $arraylength; $i++) {
	print " $array[$i]";
}
print "\n";

mergesort(\@array, 0, $arraylength);

print "numbers after mergesort:";
for (my $i = 0; $i < $arraylength; $i++) {
	print " $array[$i]";
}
print "\n";

############################################################################################################################

sub mergesort
{
	my ($aref, $begin, $end)=@_;

	my $size=$end-$begin;

	return unless($size>=2);
	my $half=$begin+int($size/2);

	mergesort($aref, $begin, $half);
	mergesort($aref, $half, $end);

	mergearray($aref, $half, $begin, $end);

}

############################################################################################################################

sub mergearray
{
	# die beiden Arrays sind in der ersten und zweiten Haelfte gespeichert
	my ($aref, $half, $begin, $end) = @_;

	for(my $i=$begin; $i<$half; ++$i) {
		if($$aref[$i] > $$aref[$half]) {
      
			($$aref[$i], $$aref[$half]) = ($$aref[$half], $$aref[$i]);

			# Sortieren der zweiten Haelfte
			my $j=$half;
			
			while($j<$end-1 && $$aref[$j] > $$aref[$j+1]) {
				($$aref[$j], $$aref[$j+1]) = ($$aref[$j+1], $$aref[$j]);
				++$j;
			}

			#########################################################################
			# Frage e											#
			#########################################################################
		}
	}
}
