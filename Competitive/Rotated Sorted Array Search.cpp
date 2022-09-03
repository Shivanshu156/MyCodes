int binary(const vector<int> &A,int start, int end, int B)
{
    int mid;
    while(start<=end)
    {   mid = (start+end)/2;
        if(A[mid]==B)   return mid;
        if(A[mid]>B)    end = mid-1;
        if(A[mid]<B)    start = mid+1;
        
        
    }
    
   return -1; 
}

int Solution::search(const vector<int> &A, int B) {
   
    int n = A.size(), start = 0, end =n-1, mid, pivot = n-1, r1, r2=-1;
    while(start<=end)
    {   mid = (start+end)/2;
        if(A[mid]>A[mid+1]){ 
            // cout<<"here in pivot"<<endl;
        pivot =mid; break; }
        if(A[mid]>A[start]) start = mid+1;
        if(A[mid]<A[end])   end = mid-1;
    }
    // cout<<"pivot is "<<pivot<<endl;
    r1 = binary(A, 0, pivot, B);
    if(pivot!=n-1)
    r2 = binary(A,pivot+1,n-1, B);
    
    return max(r1,r2);
    
    
}
