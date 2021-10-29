#!/bin/bash

ques="$1"

if [ "$ques" == "1" ]
then 
	python3 ./Q1/ass2_q1.py $2 $3 $4;
fi
if [ "$ques" == "2" ]
then
	type="$4"
	if [ "$type" == "0" ]
	then
		python3 ./Q2/ass2_q2_bin.py $2 $3 $5;
	fi
	if [ "$type" == "1" ]
	then
		python3 ./Q2/ass2_q2_mul.py $2 $3 $5;
	fi
fi