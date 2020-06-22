//Parallel Project - Matrix Chain Multiplication
//Vanilla Version
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <vector>
#include <iostream>
#include <limits.h>
using namespace std;

//function to multiply two matrices
//matrix a with dimensions a1 and a2
//matrix b with dimensions b1 and b2
vector<vector<int>> multiply(const vector<vector<int>> &a, const vector<vector<int>> &b)
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
    vector<vector<int>> sol(a1, vector<int>(b2));

    
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
vector<vector<int>> matrixMultiply(const vector<vector<vector<int>>> &matrices, const vector<vector<int>> &order, int i, int j)
{
    vector<vector<int>> a;
    vector<vector<int>> b;
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

void printSolution(const vector<vector<int>> &sol)
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
void matrixChain(const vector<vector<vector<int>>> &matrices, const vector<int> &p, int n)
{
    //iterators
    int i, j, k;
    //length of chain, cost
    int l;
    int cost = 0;

    vector<vector<int>> m(n, vector<int>(n));
    vector<vector<int>> order(n, vector<int>(n));
    vector<vector<int>> sol;

    cout<<"intitialize matrix"<<endl;
    //intitialize matrix
    for (i = 1; i < n; i++)
        m[i][i] = 0;

    cout<<"cost of chain"<<endl;
    //calculate cost of this multiplication chain
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
    cout<<"finished calculations"<<endl;
    //call multiply chain to calculate
    cout<<"Minimum number of multiplications: "<<m[1][n - 1]<<endl;

    sol = matrixMultiply(matrices, order, 1, n-1);

    printSolution(sol);

    return;
}

int main()
{
    vector<vector<vector<int>>> matrices;
    vector<vector<int>> matrix; 
    vector<int> dims;
    int i, j, k, dim1, dim2, prevDim;
    int size, total;
    double start, end, time_diff;

    srand(time(NULL));
    cout<<"Enter number of matrices: ";
    cin>>total;

    start = omp_get_wtime();

    matrices.resize(total);
    dims.resize(total+1);
    prevDim = rand() % 1000;

    //get matrix dimensions
    for(i=0; i<total; i++)
    {

        dim1 = prevDim;
        dim2 = rand() % 1000;
        prevDim = dim2;
        dims[i] = dim1;
        if (i == total - 1)
        {
            dims[i + 1] = dim2;
        }

        //fill matrix with random entries
        matrix.resize(dim1, vector<int>(dim2));
        //int matrix[dim1][dim2];
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

    matrixChain(matrices, dims, dims.size());

    end = omp_get_wtime();
    time_diff = end - start;

    cout << "Time: " << time_diff << " s" << endl;

    return 0;
}