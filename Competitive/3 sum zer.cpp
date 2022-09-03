// Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? 
// Find all unique triplets in the array which gives the sum of zero.

// Note:

//  Elements in a triplet (a,b,c) must be in non-descending order. (ie, a ≤ b ≤ c)
// The solution set must not contain duplicate triplets. For example, given array S = {-1 0 1 2 -1 -4}, A solution set is:
// (-1, 0, 1)
// (-1, -1, 2) 
// See Expected Output


vector<vector<int> > Solution::threeSum(vector<int> &A) {
    sort(A.begin(), A.end());
    int i, j, k = 0, sum, n=A.size();
    vector<vector<int>> ans_set; 
    while(k<n)
    { if(k>0)
        if(A[k]==A[k-1]) {k++; continue;}
      i = k+1;
      j = n-1;   
      
        while(i<j)
        { 
            if(i>k+1)if(A[i]==A[i-1]) {i++; continue;}
            if(j<n-1)if(A[j]==A[j+1]) {j--; continue;}
            sum = A[i]+A[j]+A[k];
            if(sum<0) i++;
            else if(sum>0) j--;
            else if(sum==0){ 
                // cout<<"here "<<A[k]<<" "<<A[i]<<" "<< A[j]<<endl;
                // ans.push_back(k); ans.push_back(i); ans.push_back(j);
                ans_set.push_back({A[k],A[i], A[j]}); i++;j--;
                // break;
            }
            
            
        }
        k++;
        
    }
    
    return ans_set;
}