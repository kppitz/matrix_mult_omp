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

    //cout<<"Calculating cost of multiplication chain"<<endl;
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

    //printSolution(sol);

    return m[1][n-1];
}

int main()
{
    vector<vector<vector<long>>> matrices;
    vector<vector<long>> matrix;
    vector<int> dims;
    //Randomized data for 100 count matrices, dimensions 1-500
    vector<int> constDims = {171, 187, 374, 167, 358, 451, 290, 235,
                             154, 65, 411, 269, 262, 496, 306, 31, 203,
                             145, 282, 373, 264, 384, 100, 384, 338, 193,
                             340, 278, 351, 248, 109, 315, 76, 422, 409,
                             289, 427, 293, 405, 404, 228, 116, 117, 132,
                             220, 29, 304, 102, 194, 185, 477, 332, 140,
                             55, 61, 187, 383, 431, 398, 165, 59, 179, 460,
                             366, 265, 276, 133, 13, 170, 400, 298, 172,
                             104, 163, 270, 72, 276, 102, 89, 285, 467,
                             438, 29, 268, 329, 454, 173, 351, 132, 121,
                             316, 43, 99, 49, 429, 385, 200, 267, 101, 458, 296};

    int i, j, k, dim1, dim2, prevDim;
    int size, total;
    string dimType;
    double start, end, time_diff;
    int num_op;

    srand(time(NULL));

    int nthreads = omp_get_num_threads();

    cout << "Matrix Chain Multiplication in Parallel" << endl;
    cout << "Enter number of matrices: ";
    cin >> total;
    while (total < 3)
    {
        cout << "Invalid number of matrices. Please enter valid number(3+): ";
        cin >> total;
    }

    cout << "Constant or randomized dimensions? (c/r): ";
    cin >> dimType;
    while (tolower(dimType[0]) != 'c' && tolower(dimType[0]) != 'r')
    {
        cout << "Invalid type of dimensions. Please enter constant or randomized(c/r): ";
        cin >> dimType;
    }
    dimType = tolower(dimType[0]);

    start = omp_get_wtime();

    matrices.resize(total);
    dims.resize(total + 1);

    if (dimType == "c")
        dims = constDims;

    prevDim = rand() % 500;

//get matrix dimensions
#pragma omp for
    for (i = 0; i < total; i++)
    {
        //Code for Constant Dimensions
        if (dimType == "c")
        {
            dim1 = dims[i];
            dim2 = dims[i + 1];
        }
        //Code for randomized dimensions
        else if (dimType == "r")
        {
            dim1 = prevDim;
            dim2 = rand() % 500;
            prevDim = dim2;
            dims[i] = dim1;
            if (i == total - 1)
            {
                dims[i + 1] = dim2;
            }
        }
        //fill matrix with random entries
        matrix.resize(dim1, vector<long>(dim2));
        for (j = 0; j < dim1; j++)
        {
            for (k = 0; k < dim2; k++)
            {
                matrix[j][k] = rand() % 100;
            }
        }
        matrices[i] = matrix;
        matrix.clear();
    }
    size = dims.size();

    //output matrix dimensions
    cout << "Matrix Dimensions" << endl;
    for (i = 0; i < size; i++)
    {
        cout << dims[i] << " ";
    }
    cout << endl;

#pragma omp parallel shared(nthreads, matrices, dims, size)
    num_op = matrixChain(matrices, dims, size, nthreads);

#pragma omp barrier
    end = omp_get_wtime();
    time_diff = end - start;

    cout << "Minimum number of multiplications: " << num_op << endl;
    cout << "Time: " << time_diff << " s" << endl;

    return 0;
}