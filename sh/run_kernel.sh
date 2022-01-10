#!/bin/bash


if [ $# != 2 ]; then
  echo 'Usage: ./[exe] [edge-standard] [dir]'
  exit 1;
fi
edge_standard=$1
dir="$2"

intact_layers=(001 002 003 004 005)
partial_layers=(002 003 004 005 006 007 008 009 010 020 050)

ds_name='../kernel_ds.txt'
ds_name='../attr_ds.txt'
ds_name='../tmp_ds.txt'

for d in `cat $ds_name`
do
  printf "dataset: $d\n"
  cd ../graph-kernels
  for layer in ${intact_layers[@]}
  do
    timeout 1d python3 train_and_test_kernel_matrix.py --dataset $d --method intact --snapshot $layer --dir $dir --edge-standard $edge_standard
  done
  printf "\n===========================================================================================\n\n"
  for layer in ${partial_layers[@]}
  do
    timeout 1d python3 train_and_test_kernel_matrix.py --dataset $d --method partial --snapshot $layer --dir $dir --edge-standard $edge_standard
  done
  printf "\n===========================================================================================\n\n"
  cd ../sh
done

