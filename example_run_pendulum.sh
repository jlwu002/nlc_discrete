#!/bin/bash
for i in {0..9}
do
	python inverted_pendulum.py --seed $i
done
