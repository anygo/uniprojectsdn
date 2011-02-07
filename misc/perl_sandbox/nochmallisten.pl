#!/usr/bin/perl -w

$input = '1';
$cnt = 0;
$sum = 0;
@nums = ();

while ($input ne '') {
	chomp ($input = <STDIN>);
	if ($input ne '') {	
		$nums[$cnt] = $input;
		$cnt++;
		$sum += $input;
	}
}

@nums = sort @nums;
print @nums;
print "\n";
print "avg: ";
if ($cnt == 0) {print "OH NO";}
print $sum/$cnt;
print "\n";
