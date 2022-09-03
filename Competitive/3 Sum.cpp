// Given an array S of n integers, find three integers in S such that the sum is closest to a given number, target. 
// Return the sum of the three integers.

// Assume that there will only be one solution

// Example: 
// given array S = {-1 2 1 -4}, 
// and target = 1.

// The sum that is closest to the target is 2. (-1 + 2 + 1 = 2)


int Solution::threeSumClosest(vector<int> &A, int B) {
    
    sort(A.begin(), A.end());
    
    int k=0,i, j, diff = INT_MAX, sum, n=A.size();
    
    while(k<n-2)
    {   i=k+1;
        j=n-1;
        while(i<j)
        {   if(abs(A[i]+A[j]+A[k] - B)<diff)
            {diff = abs(A[i]+A[j]+A[k] - B);
            //  cout<<"diff is "<<diff<<endl;
             sum = A[i]+A[j]+A[k]; }
            if(A[i]+A[j]+A[k] - B<0) i++;
            else if(A[i]+A[j]+A[k] - B>0) j--;
            else if(A[i]+A[j]+A[k] - B==0) {return A[i]+A[j]+A[k];}
            // cout<<"i, j, k is "<<i<<" "<<j<<" "<<k<<endl;
        }
        k++;
    }
    return sum;
}
