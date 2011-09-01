#!/bin/bash

java_cmd="java -cp .:/proj/i4vs/pub/aufgabe4/jgroups-2.12.1.Final.jar"
start_class="vsue.distlock.TestCases"
log_name="distlock"
program_path="$PWD"
base_port=$(( 14000 + (UID * 337 % 977) * 13 ))

######################################################
##########                                 ###########
########## NO USER SERVICEABLE PARTS BELOW ###########
##########                                 ###########
######################################################

if [ ! -f "my_hosts" -o ! -f "stack.xml" ]; then
	echo "ERROR: my_hosts or stack.xml not found."
	echo "You need to copy those files to your current working directory."
	exit 1
fi

classfile=`printf "%s" "/$start_class" | sed -e "s?\\.?/?g"`.class
if [ ! -f "$program_path/$classfile" ]; then
	echo "Make sure the class file '$classfile' file exists."
	exit 1
fi

total_hosts=0
host_count=0
group_list=""

while read -r next_host; do
	test -z "$next_host" && continue;
	
	group_list="$group_list,$next_host[$((base_port+total_hosts))]"

	total_hosts=$((total_hosts+1))
done <my_hosts

if [ "$total_hosts" -le 1 -o "$total_hosts" -gt 10 ]; then
	echo "Found $total_hosts host entries, but must be between 2 and 10."
	exit 1;
fi

cat >__screenrc__ <<EOF
startup_message off
zombie kc
logfile ${log_name}-log.%n
hardstatus alwayslastline "%-Lw%{= bW}%50>%n%f* %t%{-}%+Lw%<"
EOF

while read -r next_host; do
	test -z "$next_host" && continue;

	printf "%s\n" \
"screen -t vs -L $host_count ssh -t $next_host /bin/bash -c \"\\\"\
cd '$program_path'; \
$java_cmd \
-Djg.bind_port=$((base_port+host_count)) \
-Djg.initial_hosts=${group_list:1} \
-Dbind.address=$next_host \
$start_class $total_hosts $1\\\"\"" >>__screenrc__
	
	host_count=$((host_count+1))
done <my_hosts

rm -f "${log_name}-log."{0..9}
screen -S locktest -c __screenrc__
rm -f __screenrc__
