#!/bin/sh



quit() {
	rm -f $OUT_FILE
	exit
}



if [ ! -z $2 ]; then
	echo "Usage: $0 <executable>"
	exit
fi

WLIST_DIR=/proj/i4sp/pub/aufgabe2
OUT_FILE=/tmp/wsort-$USER-$RANDOM.out
SORTED_WLIST_DIR=/proj/ciptmp/sicherha/sorted-wlists

# Find wsort
if [ ! -z $1 ]; then
	WSORT=$1
else
	WSORT=./wsort
fi
if [ ! -x $WSORT ]; then
	echo "$WSORT: no such file"
	exit
fi

# Run wlist tests
for list in $WLIST_DIR/wlist[0-6]; do
	echo
	echo "Sorting $list:"
	$WSORT < $list > $OUT_FILE || quit
	diff -q $SORTED_WLIST_DIR/sorted-`basename $list` $OUT_FILE && echo "OK"
done

# Run test with border cases
echo
echo "Sorting bordercases:"
$WSORT < $SORTED_WLIST_DIR/bordercases.txt > $OUT_FILE || quit
diff -q $SORTED_WLIST_DIR/sorted-bordercases.txt $OUT_FILE && echo "OK"

# Run command line test
echo
echo "Sorting command line:"
$WSORT kann ich aber auch laufen > $OUT_FILE || quit
diff -q $SORTED_WLIST_DIR/sorted-cmdline $OUT_FILE && echo "OK"

# Remove output file
quit
