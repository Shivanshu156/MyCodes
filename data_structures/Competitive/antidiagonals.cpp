vector<vector<int> > Solution::diagonal(vector<vector<int> > &A) {
    int n = A.size();
    vector<vector<int>> Arr(2*n -1);
    for(int k=0; k<n; k++)
        {   Arr[k] = vector<int> (k+1);
            for(int i=0, j=k; j>=0, i<=k; i++, j--)
                Arr[k][i] = A[i][j];
        }        
    for(int k=1; k<n; k++)
        {   Arr[k+n-1] = vector<int> (n-k);
            for(int i=k, j=n-1; j>=k, i<=n-1; j--, i++)
                Arr[k+n-1][n-j-1] = A[i][j];
        }
    return Arr;
 
    
}