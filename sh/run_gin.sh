#!/bin/bash

file='../gnn_ds.txt'
file='../attr_ds.txt'
layers=(2 3 4)
intact_snapshot=(001 002 003 004 005)
partial_snapshot=(002 003 004 005 010)
hidden_dim=64

if [ $# != 4 ]; then
  echo 'Usage: ./[sh] [gin] [lr] [dropout] [edge-standard]'
  exit 1;
fi

prog=$1
lr=$2
dropout=$3
edge_standard=$4
out_file=$1-$4-$2-$3.json


for d in $(cat $file); do
  printf "dataset: $d\n"
  cd ../gnns
  for snapshot in ${intact_snapshot[@]}; do
    for layer in ${layers[@]}; do
      for i in {0..9}; do
        python3 $prog.py --dataset $d --method intact --snapshot $snapshot --run $i --layer $layer --pooling sum --hidden-dim $hidden_dim --layer-feature  --dropout $dropout --lr $lr --edge-standard $edge_standard --file $out_file
        python3 $prog.py --dataset $d --method intact --snapshot $snapshot --run $i --layer $layer --pooling mean --hidden-dim $hidden_dim --layer-feature --dropout $dropout --lr $lr --edge-standard $edge_standard --file $out_file
      done
    done
  done
  printf "\n===========================================================================================\n\n"
  for snapshot in ${partial_snapshot[@]}; do
    for layer in ${layers[@]}; do
      for i in {0..9}; do
        python3 $prog.py --dataset $d --method partial --snapshot $snapshot --run $i --layer $layer --pooling sum --hidden-dim $hidden_dim --layer-feature  --dropout $dropout --lr $lr --edge-standard $edge_standard --file $out_file
        python3 $prog.py --dataset $d --method partial --snapshot $snapshot --run $i --layer $layer --pooling mean --hidden-dim $hidden_dim --layer-feature --dropout $dropout --lr $lr --edge-standard $edge_standard --file $out_file
      done
    done
  done
  printf "\n===========================================================================================\n\n"
  cd ../sh
done
