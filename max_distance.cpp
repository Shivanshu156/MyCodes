// Given an array A of integers, find the maximum of j - i subjected to the constraint of A[i] <= A[j].

// If there is no solution possible, return -1.

// Example :

// A : [3 5 4 2]

// Output : 2 
// for the pair (3, 4)

int Solution::maximumGap(const vector<int> &A) {
    int n = A.size();
    if(n==0)
    return -1;
    if(n==1)
    return 0;
    
    
    vector<int> Arr;
    vector<pair<int, int>> vp;
    for (int i = 0; i < n; ++i) 
        vp.push_back(make_pair(A[i], i)); 
    
    sort(vp.begin(), vp.end());
    
    for(int i=0; i<n; i++)
    Arr.push_back(vp[i].second);
    
    int diff = 0, max_index=Arr[n-1];
    for(int i=n-2; i>=0; i--)
    {
        diff = max(diff, max_index - Arr[i]);
        max_index = max(max_index, Arr[i]);
        
        
        
    }

    return diff;

    
}
