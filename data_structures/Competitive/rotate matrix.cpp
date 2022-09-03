// If the array is

// [
//     [1, 2],
//     [3, 4]
// ]
// Then the rotated array becomes:

// [
//     [3, 1],
//     [4, 2]
// ]



void Solution::rotate(vector<vector<int> > &A) {
    int n= A.size();
    for(int i=0;i<n/2;i++)
    for(int j=i;j<n-1;j++)
    {   int temp = A[n-1-j][i];
        
        A[n-1-j][i] = A[n-1-i][n-1-j];
        A[n-1-i][n-1-j] = A[j][n-1-i];
        A[j][n-1-i] = A[i][j];
        A[i][j] = temp;
    }
    

}
