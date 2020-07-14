# matrix_mult_omp
This project covers the optimization, algorithms, and parallelization used to solve matrix chain multiplication to show how it can be scaled up. Based on knowledge gained from using various tools to parallelize code, the matrix chain multiplication was parallelized using OpenMP. This project was implemented and tested on the Foundry, owned by Missouri University of Science and Technology. Testing consisted of randomized randomized matrix sets not shown in the code. Data gathered from the results show how the speed of the algorithm is affected.

## To Run on PuTTY
Can be run on basic cmd/ssh terminal with openmp
To run faster and with more power, use a service with allows for more than standard computer cores.

Compile:
```bash
g++ matrix_chain_omp -fopenmp -o mco
```

Specify number of threads:
```bash
export OMP_NUM_THREADS=<integer>
```

## To Run on The Foundry:
Need access to Missouri S&T Foundry.
Can set up files in account or mount already exisiting data from another drive.

Start an interactive job:
```bash
sinteractive --time=2:00:00 --ntasks=<integer> --nodes=<integer>
```

Compile and set threads as you would to run on PuTTY, seen above.

