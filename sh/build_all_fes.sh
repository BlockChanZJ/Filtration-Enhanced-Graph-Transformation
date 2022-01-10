#!/bin/bash

intact_snapshot_layer=(002 003 004 005)
partial_snapshot_layer=(002 003 004 005 006 007 008 009 010 020 050)

if [ $# != 1 ]; then
  echo 'Usage: ./[exe] [edge_standard]'
  exit 1;
fi

#ds_name='../attr_ds.txt'
ds_name='../kernel_ds.txt'
ds_name='../tmp_ds.txt'
edge_standard=$1

for d in $(cat $ds_name); do
  printf "dataset: $d\n"
  cd ../graph-kernels
  for snapshot in ${intact_snapshot_layer[@]}; do
    python3 build_filtration_snapshot.py --dataset $d --snapshot $snapshot --method intact --node-feat default --edge-standard $edge_standard
  done
  printf "\n===========================================================================================\n\n"
  for snapshot in ${partial_snapshot_layer[@]}; do
    python3 build_filtration_snapshot.py --dataset $d --snapshot $snapshot --method partial --node-feat default --edge-standard $edge_standard
  done
  printf "\n===========================================================================================\n\n"
  cd ../sh
done

