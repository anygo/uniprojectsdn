#!/usr/bin/perl -w

$kekse = "";

while ($kekse ne "KeKsE") {
	print "Sag KeKsE!!!\n";
	chomp($kekse = <STDIN>); # schneidet \n ab
}

print "Danke, Spast!\n";
