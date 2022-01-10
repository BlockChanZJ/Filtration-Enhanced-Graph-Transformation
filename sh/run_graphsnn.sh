#!/bin/bash

file='../gnn_ds.txt'
file='../attr_ds.txt'
layers=(2 3 4)
intact_snapshot=(001 002 003 004 005)
partial_snapshot=(002 003 004 005 010)
hidden_dim=64

if [ $# != 3 ]; then
  echo 'Usage: ./[sh] [lr] [dropout] [edge-standard]'
  exit 1;
fi

prog='graphsnn'
lr=$1
dropout=$2
edge_standard=$3
out_file=snn-$3-$1-$2.json


for d in $(cat $file); do
  printf "dataset: $d\n"
  cd ../graphsnn
  for snapshot in ${intact_snapshot[@]}; do
    for layer in ${layers[@]}; do
      python3 $prog.py --dataset $d --method intact --snapshot $snapshot --n_layers $layer --pooling sum --hidden-dim $hidden_dim --dropout $dropout --lr $lr --edge-standard $edge_standard --file $out_file
      python3 $prog.py --dataset $d --method intact --snapshot $snapshot --n_layers $layer --pooling mean --hidden-dim $hidden_dim --dropout $dropout --lr $lr --edge-standard $edge_standard --file $out_file
    done
  done
  printf "\n===========================================================================================\n\n"
  for snapshot in ${partial_snapshot[@]}; do
    for layer in ${layers[@]}; do
      python3 $prog.py --dataset $d --method partial --snapshot $snapshot --n_layers $layer --pooling sum --hidden-dim $hidden_dim  --dropout $dropout --lr $lr --edge-standard $edge_standard --file $out_file
      python3 $prog.py --dataset $d --method partial --snapshot $snapshot --n_layers $layer --pooling mean --hidden-dim $hidden_dim --dropout $dropout --lr $lr --edge-standard $edge_standard --file $out_file
    done
  done
  printf "\n===========================================================================================\n\n"
  cd ../sh
done
