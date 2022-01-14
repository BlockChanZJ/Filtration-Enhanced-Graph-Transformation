#!/bin/bash


if [ $# != 2 ]; then
  echo 'Usage: ./[exe] [edge-standard] [dir]'
  exit 1;
fi
edge_standard=$1
dir="$2"

intact_snapshot_layer=(002 003 004 005)
partial_snapshot_layer=(002 003 004 005 006 007 008 009 010 020 050)

ds_name='../attr_ds.txt'
# ds_name='../kernel_ds.txt'
ds_name='../tmp_ds.txt'

for d in `cat $ds_name`
do
  printf "dataset: $d\n"
  cd ../graph-kernels
  for layer in ${intact_snapshot_layer[@]}
  do
    timeout 1d python3 train_and_test_kernel_matrix_snapshots.py --dataset $d --snapshot $layer --method intact --dir $dir --edge-standard $edge_standard
  done
  printf "\n===========================================================================================\n\n"
  for layer in ${partial_snapshot_layer[@]}
  do
    timeout 1d python3 train_and_test_kernel_matrix_snapshots.py --dataset $d --snapshot $layer --method partial --dir $dir --edge-standard $edge_standard
  done
  printf "\n===========================================================================================\n\n"
  cd ../sh
done

