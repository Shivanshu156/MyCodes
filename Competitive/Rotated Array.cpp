int Solution::findMin(const vector<int> &A) {
    
    int start = 0, end = A.size()-1, mid, N = A.size();
    if(start == end) return A[start];
    if(A[start]<A[end]) return A[start];
    while(start<=end)
    {   mid = (start+end)/2;
        // cout<<"start is "<<start<<" end is "<<end<<" mid is "<<mid<<endl;
        // cout<<"A[mid] is "<<A[mid]<<endl;
        if(A[mid-1]>A[mid])
            { int ans = A[mid];
                // cout<<"here "<<A[mid];
                return ans;
            }
        else if(A[mid] > A[mid+1])
        { int ans = A[mid+1];
                // cout<<"here "<<A[mid+1];
                return ans;
            }
        else if(A[mid]<A[end])   end = (mid-1+N)%N;
        else if(A[mid]>A[start]) start = (mid+1)%N;  
    
    }
    return A[0];
}
