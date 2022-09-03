int Solution::searchMatrix(vector<vector<int> > &A, int B) {
    int m = A.size(), n = A[0].size(), start = 0, end = m*n-1, mid;
    // pair<int,int> mid;
    
    while(start <= end)
    {   mid = (start+end)/2;
        
        
        if(A[mid/n][mid%n]==B)
        return 1;
        
        if(A[mid/n][mid%n]>B)
        end = mid-1;
        
        if(A[mid/n][mid%n]<B)
        start = mid+1;
                    
        
    }
    return 0;
    
    