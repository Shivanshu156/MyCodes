// Find the kth smallest element in an unsorted array of non-negative integers.

// Definition of kth smallest element

//  kth smallest element is the minimum possible n such that there are at least k elements in the array <= n.
// In other words, if the array A was sorted, then A[k - 1] ( k is 1 based, while the arrays are 0 based ) 
// NOTE
// You are not allowed to modify the array ( The array is read only ). 
// Try to do it using constant extra space.

// Example:

// A : [2 1 4 3 2]
// k : 3

// answer : 2

// Method 1 : Heap

int Solution::kthsmallest(const vector<int> &A, int B) {
    vector<int> K;
    for(int i=0;i<B;i++)
    K.push_back(A[i]);
    make_heap(K.begin(),K.end()); 
    
    for(int i=B;i<A.size();i++)
    if(A[i]<K.front())
    {   
        pop_heap(K.begin(), K.end());
        K.pop_back();
        K.push_back(A[i]);
        push_heap(K.begin(), K.end());
    }
 return K.front();
}



// Method 2:  O(1) space 



int Solution::kthsmallest(const vector<int> &A, int B) {
    int hi=INT_MIN, lo = INT_MAX, n = A.size(), mid, k=B;
    for(int i=0;i<n;i++)
    if(A[i]>hi) hi = A[i];
    for(int i=0;i<n;i++)
    if(A[i]<lo) lo = A[i];
    
    while(lo<=hi)
    {   int countless=0, countequal = 0;
        mid = (hi+lo)/2;
        for(int i=0;i<n;i++)    
        {   if(A[i]<mid)    countless++;
            if(A[i]==mid)    countequal++;
            if(countless>=k) break;
            
        }
        if(countless<k && countless+countequal>=k) return mid;
        else if(countless<k)    lo = mid+1;
        else if(countless>=k)   hi = mid-1;
    }
    
}
