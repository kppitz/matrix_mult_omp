//Parallel Project - Matrix Chain Multiplication
//Parallel Version - OpenMPI
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <iostream>
#include <limits.h>
using namespace std;

//function to multiply two matrices
//matrix a with dimensions a1 and a2
//matrix b with dimensions b1 and b2
vector<vector<long>> multiply(const vector<vector<long>> &a, const vector<vector<long>> &b)
{
    //easier to initialize loop invariants here
    int i, j, k;
    //get matrix dimensions
    
    int a1 = a.size();
    //cout << "multiply" << endl;
    int a2 = a[1].size();
    int b1 = b.size();
    int b2 = b[1].size();
    //initialize solution
    vector<vector<long>> sol(a1, vector<long>(b2));

    //parallelize the actual multiplication of matrices
    #pragma omp parallel for
    for(i=0; i<a1; i++)
    {
        for(j=0; j<b2; j++)
        {
            sol[i][j] = 0;
            for(k = 0; k < a2; k++)
            {
                sol[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return sol;
}

//Function to carry out Matrix Chain Multiplication
vector<vector<long>> matrixMultiply(const vector<vector<vector<long>>> &matrices, const vector<vector<long>> &order, int i, int j)
{
    vector<vector<long>> a;
    vector<vector<long>> b;
    int split = order[i][j];

    if (i == j)
    {
        //cout << "return matrix" << endl;
        return matrices[i-1];
    }
    else
    {
        //cout << "a split: " << i<<" "<<split<<endl;
        a = matrixMultiply(matrices, order, i, split);
        //cout << "b split: " << split + 1 <<" "<< j<< endl;
        b = matrixMultiply(matrices, order, split + 1, j);
        //cout << "multiplying matrices in order" << endl;
        return multiply(a,b);
    }
}

void printSolution(const vector<vector<long>> &sol)
{
    int dim1 = sol.size();
    int dim2 = sol[0].size();

    cout<<"Matrix Multiplication Solution: "<<endl;
    cout<<" { "<<endl;;
    for(int i=0; i<dim1; i++)
    {
        cout<<" [ ";

        for(int j=0; j<dim2; j++)
        {
            cout<<" "<<sol[i][j]<<" ";
        }
        cout<<" ] "<<endl;
    }
    cout<<" } "<<endl;

    return;
}

//Function to find optimal Matrix Chain Multiplication solution
long matrixChain(const vector<vector<vector<long>>> &matrices, const vector<int> &p, const int n, const int nthreads)
{
    //iterators
    int i, j, k;
    //length of chain, cost
    int l;
    int cost = 0;
    int tid = omp_get_thread_num();

    vector<vector<long>> m(n, vector<long>(n));
    vector<vector<long>> order(n, vector<long>(n));
    vector<vector<long>> sol;

    //cout<<"intitialize matrix"<<endl;
    //intitialize matrix
    for (i = 1; i < n; i++)
    {
        m[i][i] = 0;
    }

    //cout<<"cost of chain"<<endl;
    //calculate cost of this multiplication chain
    #pragma omp parallel for
        for (l = 2; l < n; l++)
        {
            for (i = 1; i < n - l + 1; i++)
            {
                j = i + l - 1;
                m[i][j] = INT_MAX;
                for (k = i; k <= j - 1; k++)
                {
                    //cout<<"calculating cost"<<endl;
                    cost = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j];

                    if (cost < m[i][j])
                    {
                        //cout<<"add cost to matrix"<<endl;
                        m[i][j] = cost;
                        //where chain is split
                        order[i][j] = k;
                        //cout << "added to order" << endl;
                    }
                }
            }
        }
        //cout << "Thread number: " << tid << endl;

    //call multiply chain to calculate    
    #pragma omp parallel shared(matrices, order)
    sol = matrixMultiply(matrices, order, 1, n - 1);

    return m[1][n-1];
}

int main()
{
    vector<vector<vector<long>>> matrices;
    vector<vector<long>> matrix;
    vector<int> dims;

    int i, j, k, dim1, dim2, prevDim;
    int size, total;
    double start, end, time_diff;
    int num_op;

    srand(time(NULL));

    int nthreads = omp_get_num_threads();

    cout<<"Enter number of matrices: ";
    cin>>total;

    start = omp_get_wtime();

    matrices.resize(total);
    dims.resize(total+1);

    prevDim = rand() % 500;

    //get matrix dimensions
    #pragma omp for
    for(i=0; i<total; i++)
    {
        //Constant Dimensions
        // dim1 = dims[i];
        // dim2 = dims[i+1];

        //Randomized dimensions
        dim1 = prevDim;
        dim2 = rand() % 500;
        prevDim = dim2;
        dims[i] = dim1;
        if (i == total - 1)
        {
            dims[i+1] = dim2;
        }

        //fill matrix with random entries
        matrix.resize(dim1, vector<long>(dim2));
        for(j=0; j<dim1; j++)
        {
            for(k=0; k<dim2; k++)
            {
                matrix[j][k] = rand()%100;
            }
        }
        matrices[i] = matrix;
        matrix.clear();
    }
    size = dims.size();

    //output matrix dimensions if randomized
    cout<<"Matrix Dimensions"<<endl;
    for(i=0;i<size; i++)
    {
        cout << dims[i] << " ";
    }
    cout<<endl;

#pragma omp parallel shared(nthreads, matrices, dims, size)
    num_op = matrixChain(matrices, dims, size, nthreads);

    #pragma omp barrier
    end = omp_get_wtime();
    time_diff = end-start;

    cout << "Minimum number of multiplications: " << num_op << endl;
    cout<<"Time: "<<time_diff<<" s"<<endl;

    return 0;
}