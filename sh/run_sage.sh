#!/bin/bash

file='../gnn_ds.txt'
file='../attr_ds.txt'
file='../tmp_ds.txt'
layers=(2 3 4)
intact_snapshot=(001 002 003 004 005)
partial_snapshot=(002 003 004 005 010)
hidden_dim=64
lr=0.001

if [ $# != 3 ]; then
  echo 'Usage: ./[sh] [lr] [dropout] [edge-standard]'
  exit 1;
fi

prog='sage_v2.py'
if [ $2 != 0.5 ]; then
  prog='sage.py'
  echo 'sage.py'
fi

lr=$1
dropout=$2
edge_standard=$3
out_file=sage-$3-$1-$2.json


for d in $(cat $file); do
  printf "dataset: $d\n"
  cd ../gnns
  for snapshot in ${intact_snapshot[@]}; do
    for layer in ${layers[@]}; do
      for i in {0..9}; do
        python3 $prog --dataset $d --method intact --snapshot $snapshot --run $i --layer $layer --pooling sum --hidden-dim $hidden_dim --layer-feature --lr $lr --file $out_file --edge-standard $edge_standard
        python3 $prog --dataset $d --method intact --snapshot $snapshot --run $i --layer $layer --pooling mean --hidden-dim $hidden_dim --layer-feature --lr $lr --file $out_file --edge-standard $edge_standard
      done
    done
  done
  printf "\n===========================================================================================\n\n"
  for snapshot in ${partial_snapshot[@]}; do
    for layer in ${layers[@]}; do
      for i in {0..9}; do
        python3 $prog --dataset $d --method partial --snapshot $snapshot --run $i --layer $layer --pooling sum --hidden-dim $hidden_dim --layer-feature  --lr $lr --file $out_file --edge-standard $edge_standard
        python3 $prog --dataset $d --method partial --snapshot $snapshot --run $i --layer $layer --pooling mean --hidden-dim $hidden_dim --layer-feature --lr $lr --file $out_file --edge-standard $edge_standard
      done
    done
  done
  printf "\n===========================================================================================\n\n"
  cd ../sh
done
