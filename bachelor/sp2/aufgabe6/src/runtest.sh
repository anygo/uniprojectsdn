#!/bin/sh

echo "Dieses Skript ruft myfind und das Unix-Programm find jeweils mit den"
echo "gleichen Parametern auf. Die Ausgabe von find und myfind wird nicht auf"
echo "dem Bildschirm angezeigt, sondern in eine Datei umgeleitet und"
echo "miteinander verglichen."
echo "Bei Ungleichheit wird der Unterschied grafisch dargestellt."
echo
echo "Dieser Test erhebt keinen Anspruch auf Vollstaendigkeit! Insbesondere ist"
echo "es empfehlenswert, sein Programm mit Kombinationen aus -maxdepth, -name,"
echo "-type und -links zu fuettern sowie (natuerlich) mit valgrind auf"
echo "Speicherzugriffsfehler zu ueberpruefen."



# GUI-Programm zur detaillierten Darstellung der Unterschiede in der Ausgabe zwischen find und myfind.
# Wer kein KDE mag, kann diese Variable auf ein anderes Programm wie z.B. "meld" setzen.
GUI_DIFF="kompare"

TOTAL_TESTS=0
PASSED_TESTS=0



# Gibt eine Ueberschriftenzeile aus.
printHeadline() {
	echo
	echo "---------------------------------"
	echo $@
	echo "---------------------------------"
}



# Startet das in der Variable $GUI_DIFF festgelegte Programm zur Anzeige der Unterschiede zweier Dateien.
# Falls kein X-Server laeuft (z.B. weil das Skript per SSH gestartet wurde), wird aufs "klassische" diff
# zurueckgegriffen.
guiDiff() {
	if [ -n "$DISPLAY" ]; then
		$GUI_DIFF $@ 2> /dev/null
	else
		diff $@
	fi
}



# Startet find und myfind mit den selben Aufrufparametern und vergleicht die Ausgaben miteinander.
testFind() {
	FIND_OUT="/tmp/find-$USER-$RANDOM.txt"
	MYFIND_OUT="/tmp/myfind-$USER-$RANDOM.txt"
	echo
	find "$@" > $FIND_OUT 2> /dev/null
	echo "./myfind $@"
	./myfind "$@" > $MYFIND_OUT
	echo -n " -> "
	diff -q $FIND_OUT $MYFIND_OUT
	if [ $? -eq 0 ]; then
		echo "OK"
		PASSED_TESTS=`expr $PASSED_TESTS + 1`
	else
		guiDiff $FIND_OUT $MYFIND_OUT
	fi
	TOTAL_TESTS=`expr $TOTAL_TESTS + 1`
	rm -f $FIND_OUT $MYFIND_OUT
}



printHeadline "1. Makefile"
echo
(make clean && make) || exit

printHeadline "2. Einfacher Test"
testFind .
testFind Makefile
testFind nonExistentPath

printHeadline "3. Mehrere Pfade"
testFind . ../..
testFind /proj/i4sp/pub/aufgabe2 nonExistentPath /proj/i4sp/pub/aufgabe4/bug.c /proj/ciptmp/sicherha

printHeadline "4. Maximale Pfadtiefe (-maxdepth)"
testFind /proj/i4sp/$USER -maxdepth 3
testFind . -maxdepth 0
testFind /proj/i4sp/pub/aufgabe2 nonExistentPath /proj/i4sp/pub/aufgabe4/bug.c /proj/ciptmp/sicherha \
         -maxdepth 2

printHeadline "5. Namensmuster (-name)"
testFind . -name runtest.sh
testFind . -name \*
testFind . -name nonExistentName
testFind . /proj/i4sp/pub/aufgabe3 nonExistentPath /proj/i4sp/pub/aufgabe4 -name \*.[ch]

printHeadline "6. Typ (-type)"
testFind /dev -type d
testFind /dev -type f
testFind /proj/i4sp/pub/aufgabe2 nonExistentPath /proj/i4sp/pub/aufgabe4/bug.c /proj/ciptmp/sicherha \
         -type d

printHeadline "7. Hardlinks (-links)"
testFind . -links 0
testFind . -links 1
testFind /usr/bin nonExistentPath /usr/lib -links 7

echo
echo "========================"
echo "Erfolgreich: $PASSED_TESTS/$TOTAL_TESTS Tests"
