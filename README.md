# Filtration-Enhanced Graph Transformation

## Dependencies

- python (==3.8)
- torch (==1.10.0)
- pyg (==2.0.2)
- torch-scatter (==1.5.9)
- torch-sparse (==0.6.12)
- networkx (==2.6.2)
- GraphRicciCurvature (==0.5.3)
- igraph (==0.9.8)
- tabulate (==0.8.9)
- GraKeL (==0.1.8)
- numpy (==1.21.0)
- pandas (==1.2.5)
- sklearn (==0.24.2)
- scipy (==1.7.0)

## Data
Add dataset name  in `tmp_ds.txt` which is available in `https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldataset`. And we have provided `build_all_feg.sh` and `build_all_fes.sh`in `sh` to download all data and build filtration-enhanced graphs and snapshots.
To download dataset and build FEG with filtration of **native edge weight(or native vertex attributes)**, navigate to `sh` folder and type the following command into the terminal:

```bash
$ ./build_all_feg.sh attr
$ ./build_all_feg.sh vattr
```

## Run Graph Kernels for FEG
To run Weisfeiler-Leman Subtree kernel, ShortestPath Kernel and GraphLet Kernel for datasets with filtration of **native edge weight**, navigate to `sh` folder and type the following command into the terminal:
```bash
$ ./run_kernel.sh attr [directory name]
```

## Run Graph Kernels for FES
To run Weisfeiler-Leman Subtree kernel, ShortestPath Kernel and GraphLet Kernel for datasets with filtration of **native vertex attributes**, navigate to `sh` folder and type the following command into the terminal:
```bash
$ ./run_kernel_snapshot.sh vattr [directory name]
```


## Run GIN
To run GIN for datasets with filtration of **core number**, navigate to `sh` folder and type the following command into the terminal:
```bash
$ ./run_gin.sh gin [lr] [dropout] degeneracy
```

## Run GraphSAGE
To run GIN for datasets with filtration of **ricci-curvature**, navigate to `sh` folder and type the following command into the terminal:
```bash
$ ./run_sage.sh [lr] [dropout] curvature
```

## Run GraphSNN
To run GraphSNN for datasets with filtration of **ricci-curvature**, navigate to `sh` folder and type the following command into the terminal:
```bash
$ ./run_graphsnn.sh [lr] [dropout] curvature
```
