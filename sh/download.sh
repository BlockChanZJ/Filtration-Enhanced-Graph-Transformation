#!/bin/bash


if [ $# != 1 ]; then
  echo 'Usage: ./[exe] [ds.txt]'
  exit 1;
fi

ds=$1

for d in `cat $ds`
do
  cd ../graph-kernels
  python3 dataset.py --dataset $d
  cd ../sh
done
