#!/usr/bin/perl -w

@zahlen = (1, 2, 2, 5, 8, 4, 9);
print "Unsorted: ";
print @zahlen;
print "\nSorted: ";
@zahlen_sorted = sort @zahlen;
print @zahlen_sorted;
print "\n";

print "Und nochmal mit foreach:\n";
foreach $x (@zahlen_sorted) {
	print "$x ";
}
print "\n";
