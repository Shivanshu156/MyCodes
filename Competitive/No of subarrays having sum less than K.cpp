int subarray_count(vector<int> A, int k)
{
    int sum = A[0], start = 0, end = 0, n = A.size(), count = 0;
    while(start<n && end<n)
    {   
         
        if(sum<k)
        {   end++;
            
            if(end>=start)
                count += end-start;
            if(end<n)
                sum+=A[end];
        }
        else{   sum -= A[start];
            start++;
        }
    }
    
    return count;
}
int Solution::numRange(vector<int> &A, int B, int C) {
    
    
    int lower, upper;
    lower = subarray_count(A,B);
    upper = subarray_count(A,C+1);
    // cout<<"upper is "<<upper;
    return upper-lower;
}